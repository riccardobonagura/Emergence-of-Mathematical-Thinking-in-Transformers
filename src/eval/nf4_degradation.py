"""
nf4_degradation.py — Baseline control for NF4 quantization (T16).
Measures the representational drift introduced purely by 4-bit quantization,
isolating it from the actual QLoRA fine-tuning drift of RQ3.
"""

import logging
import tempfile
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.extraction.extract_states import load_stimuli
from src.probing.io_utils import _atomic_write_csv, _atomic_write_json

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("nf4_eval")

def extract_nf4_native(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stimuli: list[dict],
    out_dir: Path,
    n_layers: int,
    batch_size: int = 32
) -> None:
    """
    Extracts representations using native PyTorch hooks to avoid TransformerLens
    dequantizing the NF4 weights back into persistent FP16 arrays.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_accumulators: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    activations: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # output[0] is hidden state tensor [batch, seq_len, d_model]
            activations[layer_idx] = output[0][:, -1, :].detach().cpu()
        return hook

    # Register native HF hooks on GPT-NeoX specific architecture paths
    handles = []
    for l in range(n_layers):
        handle = model.gpt_neox.layers[l].register_forward_hook(make_hook(l))
        handles.append(handle)

    for i in tqdm(range(0, len(stimuli), batch_size), desc="NF4 Forward Passes"):
        batch = stimuli[i : i + batch_size]
        texts = [s["text"] for s in batch]
        
        tokens_out = tokenizer(
            texts, 
            padding=True, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(model.device)
        
        with torch.no_grad():
            model(**tokens_out)
            
        for l in range(n_layers):
            layer_accumulators[l].append(activations[l])
            
    for handle in handles:
        handle.remove()

    for l in range(n_layers):
        H = torch.cat(layer_accumulators[l], dim=0).half()
        torch.save(H, out_dir / f"layer_{l:02d}.pt")


def main() -> None:
    logger = setup_logger()
    
    # Paths and configurations
    base_model_id = "EleutherAI/pythia-1.4b"
    stimuli_path = Path("data/processed/dataset_master_v5.jsonl")
    baseline_dir = Path("data/processed/pythia-1.4b")
    out_dir = Path("results/nf4_degradation")
    
    if not stimuli_path.exists() or not baseline_dir.exists():
        logger.error("Missing dependencies. Ensure dataset generation and T04 baseline extraction are complete.")
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and slice deterministic subset (10% of 3000)
    logger.info("Loading first 300 stimuli for statistical estimation...")
    stimuli = load_stimuli(stimuli_path)[:300]
    
    # Initialize NF4 Quantization
    logger.info("Loading Pythia-1.4B in NF4 via bitsandbytes...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_cfg, 
        device_map="cuda"
    )
    
    # Tokenizer setup for native HF extraction
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    n_layers = hf_model.config.num_hidden_layers

    # Temporary directory ensures we don't pollute the disk with control tensors
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        
        logger.info(f"Extracting NF4 representations natively to temporary directory: {tmp_dir} ...")
        extract_nf4_native(hf_model, tokenizer, stimuli, tmp_dir, n_layers, batch_size=32)
        
        # Free GPU memory before computing metrics on CPU
        del hf_model
        torch.cuda.empty_cache()
        
        logger.info("Computing geometric degradation metrics...")
        metrics = []
        
        for l in range(n_layers):
            base_file = baseline_dir / f"layer_{l:02d}.pt"
            nf4_file = tmp_dir / f"layer_{l:02d}.pt"
            
            if not base_file.exists() or not nf4_file.exists():
                logger.warning(f"Missing tensor for layer {l:02d}. Skipping.")
                continue
                
            # Load baseline and slice it to match the 300 stimuli subset
            # Cast to float32 to prevent precision overflow during metric calculation
            H_fp16 = torch.load(base_file, map_location="cpu", weights_only=True)[:300].float()
            H_nf4 = torch.load(nf4_file, map_location="cpu", weights_only=True).float()
            
            diff = H_nf4 - H_fp16
            
            # 1. Relative Frobenius Distance (Updated to linalg API)
            frob_dist = float(torch.linalg.norm(diff, "fro") / torch.linalg.norm(H_fp16, "fro"))
            
            # 2. Mean Cosine Similarity (row-wise) (Renamed)
            mean_cos = float(F.cosine_similarity(H_nf4, H_fp16, dim=1).mean())
            
            # 3. Max Absolute Difference
            max_abs = float(diff.abs().max())
            
            metrics.append({
                "layer": l,
                "frobenius_dist": frob_dist,
                "mean_cosine_similarity": mean_cos,
                "max_abs_diff": max_abs
            })
            
    if not metrics:
        logger.error("No metrics computed. Aborting.")
        sys.exit(1)
        
    df = pd.DataFrame(metrics)
    csv_path = out_dir / "per_layer_stats.csv"
    _atomic_write_csv(csv_path, df.to_dict("records"), df.columns.tolist())
    
    # Compute aggregates for summary
    mean_frobenius = float(df["frobenius_dist"].mean())
    max_frobenius = float(df["frobenius_dist"].max())
    
    # Threshold interpretation
    if mean_frobenius < 0.01:
        interpretation = "negligible — NF4 quantization noise is below 1%"
    elif mean_frobenius < 0.05:
        interpretation = "minor — document as limitation, results valid"
    else:
        interpretation = "significant — NF4 degradation confounds RQ3 drift analysis"
        
    summary = {
        "max_frobenius": max_frobenius,
        "mean_frobenius": mean_frobenius,
        "interpretation": interpretation
    }
    
    json_path = out_dir / "summary.json"
    _atomic_write_json(json_path, summary)
    
    logger.info(f"NF4 Degradation baseline complete.")
    logger.info(f"Mean Frobenius: {mean_frobenius:.4f} -> {interpretation}")
    logger.info(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()