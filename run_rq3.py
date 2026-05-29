#!/usr/bin/env python
"""
run_rq3.py — RQ3 orchestrator: dynamic probing on MetaMath/QLoRA checkpoints.
Applies frozen probe weights to checkpoint hidden states and isolates geometric drifts.

Enforces fixes RQ3-01 to RQ3-05: resolve step-zero flattening bug, separate math/ctrl
drift manifolds, decouple probing tables from GSM8K columns, and verify config seeds hashes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

from src.config.categories import MATH_CATS, CTRL_CATS
from src.probing.io_utils import (MetadataHandler, setup_logging,
                                   load_hidden_states, load_test_indices,
                                   _atomic_write_csv)
from src.probing.seeds import get_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_rq3")


def compute_geometric_drift(H_ckpt: np.ndarray, H_base: np.ndarray) -> float:
    """Normalised Frobenius distance: ||H_ckpt - H_base||_F / (N * d)."""
    N, d = H_ckpt.shape
    if N == 0:
        return 0.0
    return float(np.linalg.norm(H_ckpt - H_base, ord="fro") / (N * d))


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 dynamic probing")
    parser.add_argument("--config",         required=True)
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory of per-layer .pt tensors for this checkpoint")
    args = parser.parse_args()

    # RQ3-04: Explicit UTF-8 configuration reading forced
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    logger     = setup_logging(output_dir)
    ckpt_path  = Path(args.checkpoint_dir)

    # ── RQ3-01 & RQ3-01b: EXPLICIT STEP-PARSING RESOLUTION LOOP ───────────────
    # Neutralizes the underscore splitting defect which flattened checkpoint steps to zero.
    stem = ckpt_path.name
    if "final_checkpoint" in stem or "final_adapter" in stem:
        # Map to total training length boundary if terminal checkpoint is active
        step_num = config.get("total_training_steps", 2000)
        logger.info(f"Terminal checkpoint detected. Mapping step identifier to final marker: {step_num}")
    else:
        try:
            normalized_stem = stem.replace("_", "-")
            step_num = int(normalized_stem.split("-")[-1])
        except (ValueError, IndexError):
            step_num = 0
            logger.warning(f"Could not parse checkpoint step from string value '{stem}'. Defaulting index to 0.")

    logger.info("Dynamic evaluation engine active | checkpoint step tracking field: %d", step_num)

    model_dir = Path("data/processed") / config["model_name"]
    meta = MetadataHandler(model_dir / "metadata.json")
    n_layers = meta.get_n_layers()

    # ── RQ3-05: REPRODUCTION CONFIGURATION SEED INTEGRITY GUARD ───────────────
    # Verifies if frozen weights are aligned with current evaluation matrices setups
    config_hash_file = output_dir / "weights/rq2_config_hash.json"
    if config_hash_file.exists():
        with open(config_hash_file, "r", encoding="utf-8") as hf:
            saved_hash = json.load(hf).get("seed", -1)
            if saved_hash != config["seed"]:
                logger.error(f"Fatal verification mismatch: Current evaluation seed ({config['seed']}) "
                             f"differs from the configuration seed used to train weights ({saved_hash}).")
                sys.exit(1)

    label_arrays: dict[str, np.ndarray] = {
        field: meta.get_labels(field)
        for field in {pc["label_field"] for pc in config["properties"].values()}
    }

    # ── RQ3-02: REPRESENTATIONAL DRIFT MANIFOLD SEPARATION (B-09) ─────────────
    # Isolate mathematical sub-spaces from linguistic variants to prevent noise contamination
    categories = np.array(meta.data.get("categories", []))
    math_idx_global = np.where(np.isin(categories, list(MATH_CATS)))[0]
    ctrl_idx_global = np.where(np.isin(categories, list(CTRL_CATS)))[0]

    eval_size  = config.get("eval_subset_size", 200)
    rng_drift  = np.random.default_rng(get_seed(config["seed"], "global_drift_sampling"))

    drift_idx_math = rng_drift.choice(math_idx_global, size=min(eval_size, len(math_idx_global)), replace=False)
    drift_idx_ctrl = rng_drift.choice(ctrl_idx_global, size=min(eval_size, len(ctrl_idx_global)), replace=False)

    results = []

    for l in range(n_layers):
        H_base = load_hidden_states(model_dir / f"layer_{l:02d}.pt")
        H_ckpt = load_hidden_states(ckpt_path  / f"layer_{l:02d}.pt")

        # Compute isolated geometric displacement paths
        geom_delta_math = compute_geometric_drift(H_ckpt[drift_idx_math], H_base[drift_idx_math])
        geom_delta_ctrl = compute_geometric_drift(H_ckpt[drift_idx_ctrl], H_base[drift_idx_ctrl])

        for prop_name, prop_cfg in config["properties"].items():
            w_path = output_dir / "weights" / f"layer_{l:02d}_{prop_name}.npy"
            b_path = output_dir / "weights" / f"layer_{l:02d}_{prop_name}_bias.npy"
            if not w_path.exists():
                continue

            w_orig   = np.load(w_path)
            b_orig   = np.load(b_path)
            test_idx = load_test_indices(output_dir, prop_name)
            X_test   = H_ckpt[test_idx]

            if w_orig.ndim == 1:
                y_pred = (np.dot(X_test, w_orig) + b_orig > 0).astype(int)
            else:
                y_pred = np.argmax(np.dot(X_test, w_orig.T) + b_orig, axis=1)

            label_field = prop_cfg["label_field"]
            y_true      = label_arrays[label_field][test_idx]

            acc = float(np.round(accuracy_score(y_true, y_pred), 4))
            results.append({
                "step":             step_num,
                "layer":            l,
                "property":         prop_name,
                "probing_acc":      acc,
                "geom_delta_math":  float(np.round(geom_delta_math, 6)),
                "geom_delta_ctrl":  float(np.round(geom_delta_ctrl, 6))
            })

    # ── RQ3-03: DECOUPLED ACCURACY TRAJECTORY LOGS TARGETING (B-11) ───────────
    # Outputs strictly to trajectories_probing.csv to remove pd.NA unaligned column corruptions
    df      = pd.DataFrame(results)
    dyn_dir = output_dir / "dynamic"
    dyn_dir.mkdir(parents=True, exist_ok=True)
    traj    = dyn_dir / "trajectories_probing.csv"

    if traj.exists():
        old = pd.read_csv(traj)
        df  = pd.concat([old[old["step"] != step_num], df], ignore_index=True)

    _atomic_write_csv(traj, df.to_dict("records"), df.columns.tolist())
    logger.info(f"Probing trajectories committed. Max Math Drift: {df['geom_delta_math'].max():.4f}")


if __name__ == "__main__":
    main()
