"""
checkpoint_loop.py — Hardware orchestration for RQ3 dynamic evaluation.
Iterates over saved LoRA checkpoints, merges weights, wraps in TransformerLens,
extracts representations, and triggers the static probing validation.
"""

import gc
import logging
import subprocess
import yaml
from pathlib import Path

import torch
import transformers

# ENV-02: Protect against GPT-NeoX vmap/SDPA bug in newer transformers
assert transformers.__version__ < "4.49", (
    f"transformers {transformers.__version__} has a vmap/SDPA bug with GPT-NeoX. "
    "Pin to <4.49: pip install 'transformers>=4.46,<4.49'"
)

from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformer_lens import HookedTransformer

from src.extraction.extract_states import extract_from_model, load_stimuli
from src.config.models import get_model_profile

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("checkpoint_loop")

def process_checkpoint(
    ckpt_dir: Path, 
    base_model_id: str, 
    extract_batch_size: int,
    stimuli: list[dict], 
    config_path: Path,
    logger: logging.Logger
) -> None:
    """Merges LoRA adapter, extracts states, and triggers RQ3 metric computation."""
    logger.info(f"Processing checkpoint: {ckpt_dir.name}")
    
    # 1. Load Base Model (FP16, CPU first to avoid VRAM spikes during merge)
    logger.info("Loading base HF model...")
    base_hf = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu" 
    )
    
    # 2. Merge LoRA Weights
    logger.info("Merging LoRA weights (in-memory)...")
    peft_model = PeftModel.from_pretrained(base_hf, str(ckpt_dir))
    merged_hf = peft_model.merge_and_unload()
    
    # 3. Wrap in HookedTransformer
    logger.info("Wrapping in HookedTransformer...")
    hooked_model = HookedTransformer.from_pretrained(
        base_model_id,
        hf_model=merged_hf,
        device="cuda",
        dtype=torch.float16,
        fold_ln=True
    )
    
    # 4. Extract Hidden States
    out_dir = Path("data/processed/checkpoints_extracted") / ckpt_dir.name
    logger.info(f"Extracting hidden states to {out_dir} (Batch: {extract_batch_size})...")
    extract_from_model(hooked_model, stimuli, out_dir, batch_size=extract_batch_size)
    
    # Free VRAM strictly before launching subprocess
    del hooked_model
    del merged_hf
    del peft_model
    del base_hf
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Execute RQ3 Orchestrator via Subprocess
    logger.info("Triggering run_rq3.py evaluation...")
    cmd = [
        "python", "run_rq3.py",
        "--config", str(config_path),
        "--checkpoint_dir", str(out_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"run_rq3.py failed for {ckpt_dir.name}:\n{result.stderr}")
    else:
        logger.info(f"Evaluation complete for {ckpt_dir.name}.")


def main() -> None:
    logger = setup_logger()
    config_path = Path("configs/config.yaml")
    stimuli_path = Path("data/processed/dataset_master_v5.jsonl")
    checkpoints_base = Path("data/processed/checkpoints")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config missing: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_name = cfg.get("model_name", "pythia-1.4b")
    profile = get_model_profile(model_name)
    base_model_id = profile["hf_path"]
    extract_batch_size = profile["extract_batch_size"]

    if not stimuli_path.exists():
        raise FileNotFoundError(f"Dataset missing: {stimuli_path}")
    
    stimuli = load_stimuli(stimuli_path)
    
    checkpoints = sorted(
        [d for d in checkpoints_base.iterdir() if d.is_dir() and "checkpoint" in d.name],
        key=lambda x: int(x.name.split("-")[-1]) if "-" in x.name else 0
    )
    
    if not checkpoints:
        logger.warning(f"No checkpoints found in {checkpoints_base}")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoints for {model_name}.")
    
    for ckpt in checkpoints:
        process_checkpoint(ckpt, base_model_id, extract_batch_size, stimuli, config_path, logger)
        
    logger.info("Checkpoint extraction loop completed.")

if __name__ == "__main__":
    main()