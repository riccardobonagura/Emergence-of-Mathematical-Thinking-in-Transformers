#!/usr/bin/env python
"""
run_rq3.py — RQ3 orchestrator: dynamic probing on MetaMath/QLoRA checkpoints.

Applies frozen probe weights (trained in RQ2) to checkpoint hidden states;
computes geometric drift and probing accuracy per step to track geometry evolution.

Usage: run_rq3.py --config configs/config.yaml --checkpoint_dir data/processed/checkpoints/ckpt_500
"""

import argparse
import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score

from src.probing.io_utils import (MetadataHandler, setup_logging,
                                   load_hidden_states, load_test_indices,
                                   _atomic_write_csv)
from src.probing.seeds import get_seed


def compute_geometric_drift(H_ckpt: np.ndarray, H_base: np.ndarray) -> float:
    """Normalised Frobenius distance: ||H_ckpt - H_base||_F / (N * d)."""
    N, d = H_ckpt.shape
    return float(np.linalg.norm(H_ckpt - H_base, ord="fro") / (N * d))


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 dynamic probing")
    parser.add_argument("--config",         required=True)
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory of per-layer .pt tensors for this checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    logger     = setup_logging(output_dir)
    ckpt_path  = Path(args.checkpoint_dir)

    # Infer training step from directory name (e.g. "ckpt_500" → 500)
    stem     = ckpt_path.stem
    step_num = int(stem.split("_")[-1]) if "_" in stem else 0
    logger.info("Dynamic evaluation | checkpoint step: %d", step_num)

    model_dir = Path("data/processed") / config["model_name"]
    meta      = MetadataHandler(model_dir / "metadata.json")   # robust key access
    n_layers  = meta.get_n_layers()
    n_stimuli = meta.get_n_stimuli()

    # Pre-load all label arrays from metadata (avoids JSONL re-read at runtime)
    label_arrays: dict[str, np.ndarray] = {
        field: meta.get_labels(field)
        for field in {pc["label_field"] for pc in config["properties"].values()}
    }

    # Global drift sample: all categories (incl. CTRL), deterministic subsample
    eval_size  = config.get("eval_subset_size", 200)
    rng_drift  = np.random.default_rng(get_seed(config["seed"], "global_drift_sampling"))
    drift_idx  = rng_drift.choice(n_stimuli, size=min(eval_size, n_stimuli), replace=False)

    results = []

    for l in range(n_layers):
        H_base = load_hidden_states(model_dir / f"layer_{l:02d}.pt")
        H_ckpt = load_hidden_states(ckpt_path  / f"layer_{l:02d}.pt")

        geom_delta = compute_geometric_drift(H_ckpt[drift_idx], H_base[drift_idx])

        for prop_name, prop_cfg in config["properties"].items():
            w_path = output_dir / "weights" / f"layer_{l:02d}_{prop_name}.npy"
            b_path = output_dir / "weights" / f"layer_{l:02d}_{prop_name}_bias.npy"
            if not w_path.exists():
                continue

            w_orig   = np.load(w_path)
            b_orig   = np.load(b_path)
            test_idx = load_test_indices(output_dir, prop_name)
            X_test   = H_ckpt[test_idx]

            # Geometric inference: no scaler — weights are already in original space
            if w_orig.ndim == 1:
                # Binary: sign of the dot product with the probe hyperplane
                y_pred = (np.dot(X_test, w_orig) + b_orig > 0).astype(int)
            else:
                # Multiclass: argmax over class scores
                y_pred = np.argmax(np.dot(X_test, w_orig.T) + b_orig, axis=1)

            label_field = prop_cfg["label_field"]
            y_true      = label_arrays[label_field][test_idx]

            acc = float(np.round(accuracy_score(y_true, y_pred), 4))
            results.append({
                "step":       step_num,
                "layer":      l,
                "property":   prop_name,
                "probing_acc": acc,
                "geom_delta": float(np.round(geom_delta, 6)),
            })

    # Append this checkpoint to the running trajectory file (idempotent on re-run)
    df      = pd.DataFrame(results)
    dyn_dir = output_dir / "dynamic"
    dyn_dir.mkdir(parents=True, exist_ok=True)
    traj    = dyn_dir / "trajectories.csv"

    if traj.exists():
        old = pd.read_csv(traj)
        df  = pd.concat([old[old["step"] != step_num], df], ignore_index=True)

    _atomic_write_csv(traj, df.to_dict("records"), df.columns.tolist())
    logger.info(
        "Trajectories saved. Max geometric drift: %.4f", df["geom_delta"].max()
    )


if __name__ == "__main__":
    main()