"""
nf4_degradation.py — Baseline control for NF4 quantization stability (T16).
Measures representational degradation introduced purely by 4-bit compression,
comparing an unquantized BF16 reference against the double-quantized NF4 model
(both BF16 compute) so the metric reflects 4-bit weight quantization alone.

Aligns the two passes (shared attention masks, stratified evaluation set,
config-driven model identity) and reports relative + dimension-normalized
Frobenius distance per layer.
"""

import argparse
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config.models import get_model_profile
from src.config.categories import ALL_CATS
from src.extraction.extract_states import load_stimuli
from src.probing.io_utils import _atomic_write_csv, _atomic_write_json, setup_logging
from src.probing.seeds import get_seed

logger = logging.getLogger("nf4_eval")


def extract_native_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stimuli: list[dict],
    out_dir: Path,
    n_layers: int,
    batch_size: int = 32
) -> None:
    """Extract last-token activations per layer using native PyTorch forward hooks.

    Forwards an explicit attention_mask so both passes (reference and NF4) see
    identical masking.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_accumulators: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    activations: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # output[0] holds the hidden state block [batch, seq_len, d_model]
            activations[layer_idx] = output[0][:, -1, :].detach().cpu()
        return hook

    # Register hooks on the native HF GPT-NeoX layers.
    handles = []
    for l in range(n_layers):
        handle = model.gpt_neox.layers[l].register_forward_hook(make_hook(l))
        handles.append(handle)

    for i in tqdm(range(0, len(stimuli), batch_size), desc="Forward Passes Loop"):
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
        # Store in float32: FP16 storage overflows on Pythia's deep-layer rogue
        # dimensions (values > 65504 -> inf -> NaN Frobenius ratios at layers 12+).
        H = torch.cat(layer_accumulators[l], dim=0).float()
        torch.save(H, out_dir / f"layer_{l:02d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="NF4 quantization degradation baseline (T16)")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to operational config file (e.g., configs/config_rq2.yaml)"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/nf4_degradation")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    # Model identity from the config / profile registry.
    model_name = config["model_name"]
    model_profile = get_model_profile(model_name)
    base_model_id = model_profile["hf_path"]
    global_seed = int(config["seed"])

    stimuli_path = Path("data/processed/dataset_master_v5.jsonl")
    if not stimuli_path.exists():
        logger.error(f"Execution blocked: master dataset missing at {stimuli_path}.")
        sys.exit(1)

    full_stimuli = load_stimuli(stimuli_path)
    df_stimuli = pd.DataFrame(full_stimuli)

    # Stratified subsample: 75 items per category (300 total) to avoid the
    # single-category slice bias of a contiguous draw.
    logger.info("Drawing balanced stratified subsample across categories...")
    sampling_seed = get_seed(global_seed, "nf4_degradation_sampling", 0)
    rng_strat = np.random.default_rng(sampling_seed)

    selected_stimuli = []
    for cat in ALL_CATS:
        cat_slice = df_stimuli[df_stimuli["category"] == cat]
        if len(cat_slice) < 75:
            raise ValueError(f"Category target '{cat}' possesses insufficient rows (found {len(cat_slice)}).")
        chosen_indices = rng_strat.choice(cat_slice.index, size=75, replace=False)
        selected_stimuli.extend(cat_slice.loc[chosen_indices].to_dict("records"))

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with tempfile.TemporaryDirectory() as tmp_ref_str, tempfile.TemporaryDirectory() as tmp_nf4_str:
        tmp_ref = Path(tmp_ref_str)
        tmp_nf4 = Path(tmp_nf4_str)

        # Pass 1: unquantized BF16 reference. BF16 shares FP32's exponent range,
        # so Pythia's large deep-layer activations do not overflow, and it
        # matches NF4's bnb_4bit_compute_dtype=bfloat16 — isolating the pure
        # 4-bit weight-quantization effect.
        logger.info(f"Instantiating baseline {model_name} in unquantized BF16 reference...")
        hf_ref = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=model_profile.get("attn_implementation", "eager")
        )
        n_layers = hf_ref.config.num_hidden_layers

        logger.info("Caching BF16 reference activations...")
        extract_native_states(hf_ref, tokenizer, selected_stimuli, tmp_ref, n_layers, batch_size=32)
        del hf_ref
        torch.cuda.empty_cache()

        # Pass 2: double-quantized NF4 model. double_quant matches the
        # quantization used during training.
        logger.info(f"Instantiating {model_name} in double-quantized NF4...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        hf_nf4 = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=model_profile.get("attn_implementation", "eager")
        )

        logger.info("Caching NF4 activations...")
        extract_native_states(hf_nf4, tokenizer, selected_stimuli, tmp_nf4, n_layers, batch_size=32)
        del hf_nf4
        torch.cuda.empty_cache()

        # Per-layer distance between reference and NF4 representations.
        logger.info("Computing per-layer representational distance...")
        metrics = []
        for l in range(n_layers):
            H_ref = torch.load(tmp_ref / f"layer_{l:02d}.pt", map_location="cpu", weights_only=True).float()
            H_nf4 = torch.load(tmp_nf4 / f"layer_{l:02d}.pt", map_location="cpu", weights_only=True).float()
            diff = H_nf4 - H_ref
            N, d = H_ref.shape

            # Report both relative Frobenius and a dimension-normalized distance
            # for comparison against run_rq3.py trajectories.
            frob_dist_relative = float(torch.linalg.norm(diff, "fro") / torch.linalg.norm(H_ref, "fro"))
            frob_dist_normalized_dim = float(torch.linalg.norm(diff, "fro") / (N * d))

            mean_cos = float(F.cosine_similarity(H_nf4, H_ref, dim=1).mean())
            max_abs = float(diff.abs().max())

            metrics.append({
                "layer": l,
                "frobenius_dist_relative": round(frob_dist_relative, 6),
                "frobenius_dist_normalized_dim": round(frob_dist_normalized_dim, 7),
                "mean_cosine_similarity": round(mean_cos, 6),
                "max_abs_diff": round(max_abs, 6)
            })

        df_metrics = pd.DataFrame(metrics)
        _atomic_write_csv(out_dir / "per_layer_stats.csv", df_metrics.to_dict("records"), df_metrics.columns.tolist())

        mean_frob_rel = float(df_metrics["frobenius_dist_relative"].mean())
        mean_frob_norm = float(df_metrics["frobenius_dist_normalized_dim"].mean())

        # Interpretation bands from Dettmers et al. (2023): <3% degradation for
        # 1B-7B scales is the empirical baseline.
        if mean_frob_rel < 0.03:
            interpretation = "negligible — quantization noise is under the 3% boundary established by Dettmers et al. (2023)"
        elif mean_frob_rel < 0.05:
            interpretation = "minor — structural traits preserved with acceptable degradation artifacts"
        else:
            interpretation = "significant — quantization noise exceeds baseline bounds, threatening RQ3 trajectory tracking validity"

        summary = {
            "model_name": model_name,
            "reference_dtype": "bfloat16",
            "quantized_dtype": "nf4 (double-quant, bf16 compute)",
            "mean_frobenius_relative": round(mean_frob_rel, 6),
            "mean_frobenius_normalized_dim": round(mean_frob_norm, 7),
            "interpretation": interpretation,
            "reference": "Dettmers et al. 2023 (QLoRA empirical limits framework)",
            "timestamp": datetime.now().isoformat()
        }
        _atomic_write_json(out_dir / "summary.json", summary)

        logger.info("NF4 degradation analysis complete.")
        logger.info(f"Mean Relative Frobenius: {mean_frob_rel * 100:.2f}% -> {interpretation}")
        logger.info(f"Summaries saved to {out_dir}")


if __name__ == "__main__":
    main()
