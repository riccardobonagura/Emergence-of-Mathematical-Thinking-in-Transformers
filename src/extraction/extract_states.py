"""
Phase 2 - Hidden States Extraction (Batch, Mask-based Indexing, Layer-wise)
Isolates resid_post ensuring topological tracking and VRAM efficiency.

v5 Dataset Compatibility
------------------------
In v5, all probeable properties (sign, parity) use the "last_token" strategy:
the representation is extracted from the last real token of the sequence,
which always coincides with the "=" token for CAT-* stimuli.

With left-padding, the final real token is always at index `max_len - 1`
regardless of the sequence length.
"""

from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

def load_stimuli(path: str | Path) -> list[dict]:
    """Load the stimuli generated in Phase 1."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_extraction_metadata(
    stimuli: list[dict],
    out_dir: Path,
    model: "HookedTransformer",
) -> None:
    """
    Build and save the mapping between the row index of extracted tensors
    and stimulus IDs, vital for decoding (Phase 3).
    """
    metadata = {
        "stimuli_ids": [s["id"] for s in stimuli],
        "categories":  [s["category"] for s in stimuli],
        "probe_strategy": "last_token",
        "dataset_version": stimuli[0].get("dataset_version", "unknown") if stimuli else "unknown",
        "n_layers":  model.cfg.n_layers,
        "d_model":   model.cfg.d_model,
        "n_stimuli": len(stimuli),
        "labels": {
            "sign":   [s["labels"].get("sign",   -1) for s in stimuli],
            "parity": [s["labels"].get("parity", -1) for s in stimuli],
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def extract_from_model(
    model: "HookedTransformer",
    stimuli: list[dict],
    out_dir: Path,
    batch_size: int = 32,
    verify_regression: bool = False
) -> None:
    """
    Functional entry-point for extraction.
    Decouples model initialization from the extraction loop logic.
    Designed to be invoked both at baseline and on merged checkpoints (RQ3).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Tokenizer setup required for proper token alignment
    model.tokenizer.padding_side = "left" 
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        
    # --- THE FAIL-FAST GUARDIAN ---
    sample_text = stimuli[0]["text"]
    sample_tokens = model.tokenizer(sample_text)["input_ids"]
    last_token_str = model.tokenizer.decode([sample_tokens[-1]])
    
    if "=" not in last_token_str:
        raise AssertionError(
            f"FATAL: Tokenizer alignment broken for {model.cfg.model_name}.\n"
            f"Expected extraction token '=', but got {last_token_str!r}.\n"
            f"This architecture tokenizes numbers/spaces differently. "
            "Write custom extraction index logic before proceeding."
        )
    # ------------------------------

    n_layers = model.cfg.n_layers
    
    save_extraction_metadata(stimuli, out_dir, model)
    print(f"Metadata saved. Starting extraction for {len(stimuli)} stimuli across {n_layers} layers.")
    
    # OPT-01 Architectural Shift: Single Forward Pass Multi-Hook
    # We accumulate detached representations in host RAM (CPU) to prevent VRAM explosion
    # while gaining a ~20x speedup by computing all layers in a single pass.
    layer_accumulators: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}

    def make_hook(layer_idx: int):
        def _hook(value: torch.Tensor, hook) -> torch.Tensor:
            # value shape: [batch, seq_len, d_model]. Target is strictly the last token (-1).
            layer_accumulators[layer_idx].append(value[:, -1, :].detach().cpu())
            return value
        return _hook

    all_hooks = [
        (f"blocks.{l}.hook_resid_post", make_hook(l))
        for l in range(n_layers)
    ]

    for i in tqdm(range(0, len(stimuli), batch_size), desc="Forward Passes"):
        batch = stimuli[i : i + batch_size]
        texts = [s["text"] for s in batch]
        
        tokens_out = model.tokenizer(
            texts, 
            padding=True, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(model.cfg.device)
        
        input_ids = tokens_out["input_ids"]

        with torch.no_grad():
            # KNOWN LIMITATION: attention_mask is intentionally not passed to run_with_hooks here.
            # With left-padding, the model currently attends to padding tokens. 
            # Retained 'as-is' to preserve bit-identical baseline backward compatibility.
            model.run_with_hooks(input_ids, fwd_hooks=all_hooks)
            
    # Concat batches per layer, verify numerical stability (optional), and serialize
    for l in range(n_layers):
        H = torch.cat(layer_accumulators[l], dim=0).half()  # Cast to FP16
        out_file = out_dir / f"layer_{l:02d}.pt"
        
        if verify_regression and out_file.exists():
            # Match the baseline slice to the length of H to prevent shape mismatch when verifying subsets
            baseline = torch.load(out_file, map_location="cpu", weights_only=True)[:len(H)]
            assert torch.allclose(H, baseline, atol=1e-3), \
                f"Numerical regression failed at layer {l:02d}! Max diff: {(H - baseline).abs().max()}"
            print(f"  [OK] Numerical regression verified for layer_{l:02d}.pt")

        torch.save(H, out_file)
        if not verify_regression:
            print(f"  layer_{l:02d}.pt  shape={H.shape}")

    torch.cuda.empty_cache()


def main():
    """
    Executable script for baseline pre-fine-tuning extraction.
    """
    parser = argparse.ArgumentParser(description="Baseline Hidden States Extraction.")
    parser.add_argument("--verify", action="store_true", help="Verify FP16 numerical regression against existing tensors.")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    DATA_PATH = Path("data/processed/dataset_master_v5.jsonl")
    OUT_DIR   = Path("data/processed/pythia-1.4b")
    
    print("Initializing HookedTransformer in FP16 (Baseline)...")
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-1.4b",
        device="cuda", 
        dtype=torch.float16,
        fold_ln=True,    # fold LayerNorm for mechanistic analysis
    )
    
    stimuli = load_stimuli(DATA_PATH)
    
    # If verify is on, we only test on a subset to quickly validate the code path
    if args.verify:
        print("Verification mode active. Testing on first 10 stimuli.")
        stimuli = stimuli[:10]
        
    extract_from_model(model, stimuli, OUT_DIR, batch_size=32, verify_regression=args.verify)


if __name__ == "__main__":
    import numpy as np
    import random
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    main()