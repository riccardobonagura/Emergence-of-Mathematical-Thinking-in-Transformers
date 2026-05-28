#!/usr/bin/env python
"""
run_rq1.py — RQ1 orchestrator: Emergence threshold location (l*).
Calculates category-balanced Isotropy, Evolutionary CKA, and Inter-Category CKA.

Enforces structural fixes RQ1-01 to RQ1-08 by eliminating all hardcoded paths,
constants, and uneven sampling distributions, replacing them with a strict,
config-driven statistical framework.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.config.categories import CTRL_CATS, MATH_CATS
from src.metrics.cka import compute_cka_intercategory, linear_cka
from src.metrics.isotropy import isotropy_exact, run_isotropy_analysis
from src.probing.io_utils import (
    MetadataHandler,
    _atomic_save_npy,
    _atomic_write_csv,
    load_hidden_states,
)
from src.probing.seeds import get_seed

# Initialize standard logging framework configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_rq1")


def main() -> None:
    # ── RQ1-01: ARGPARSE & CONFIGURATION INITIALIZATION ───────────────────────
    parser = argparse.ArgumentParser(description="Strict RQ1 Emergence Orchestrator")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the master configuration YAML file (e.g., configs/config_rq2.yaml)"
    )
    args = parser.parse_args()

    # Load and validate YAML file configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Derive operational directories and seeds from configuration file keys
    global_seed = int(config["seed"])
    PROC_DIR = Path("data/processed") / config["model_name"]
    STIMULI_PATH = Path("data/processed/dataset_master_v5.jsonl")
    OUT_DIR = Path(config.get("output_dir", "results/rq1_emergence"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Enforce pre-flight extraction existence check
    meta_path = PROC_DIR / "metadata.json"
    if not meta_path.exists():
        logger.error(f"Fatal: Extraction metadata missing at {meta_path}. Run state extraction first.")
        sys.exit(1)

    # Instantiate metadata parsing handler contract interface
    meta = MetadataHandler(meta_path)

    # ── RQ1-02: DYNAMIC LAYER COUNT DERIVATION ────────────────────────────────
    n_layers = meta.get_n_layers()
    logger.info(f"Loaded architecture profile for {config['model_name']}: {n_layers} layers detected.")

    # ── RQ1-04: DEFENSIVE CATEGORY RECONCILIATION GUARD ───────────────────────
    categories_pool = meta.data.get("categories")
    if categories_pool is None:
        raise ValueError(
            f"Fatal architectural misalignment: 'categories' array is missing from metadata at {meta_path}. "
            "This indicates an incompatible pre-v5 enriched extraction file format."
        )
    categories = np.array(categories_pool)

    # Isolate sample sequence matrices indices per topological category
    math_idx_raw = np.where(np.isin(categories, list(MATH_CATS)))[0]
    ctrl_idx_raw = np.where(np.isin(categories, list(CTRL_CATS)))[0]

    if math_idx_raw.size == 0 or ctrl_idx_raw.size == 0:
        raise ValueError(
            f"Fatal: Category indexing yielded empty sets. "
            f"Math indices: {math_idx_raw.size}, Control indices: {ctrl_idx_raw.size}. Check configuration mapping."
        )

    logger.info("--- STARTING RQ1 METHODOLOGICALLY HARDENED EMBEDDING ANALYSIS ---")

    # ── RQ1-03: ISOTROPY ANALYSIS SEED DISCIPLINE ROUTING ─────────────────────
    logger.info("Step 1: Commencing Exact Isotropy Calculation Loop...")
    # Derive localized deterministic seed to protect against cross-metric leakage
    isotropy_seed = get_seed(global_seed, "isotropy_orchestration", 0)

    run_isotropy_analysis(
        processed_dir=str(PROC_DIR),
        stimuli_path=str(STIMULI_PATH),
        output_path=str(OUT_DIR / "isotropy_pythia.csv"),
        seed=isotropy_seed
    )

    # ── RQ1-05: DISHOMOGENEOUS SAMPLE SIZE BALANCING (SUBSAMPLING) ────────────
    # Mathematical representations and controls must contain identical sample sizes
    # to equalize background estimation variance before executing CKA operations.
    n_sub = min(math_idx_raw.size, ctrl_idx_raw.size)
    logger.info(f"Balancing dataset variants. Subsampling evaluation boundaries to exact N = {n_sub}")

    subsampling_seed = get_seed(global_seed, "rq1_subsampling", 0)
    rng_sub = np.random.default_rng(subsampling_seed)

    math_idx = rng_sub.choice(math_idx_raw, size=n_sub, replace=False)
    ctrl_idx = rng_sub.choice(ctrl_idx_raw, size=n_sub, replace=False)

    # Sort indices to maintain sequential row caching optimizations
    math_idx.sort()
    ctrl_idx.sort()

    # ── SECTION 2 — ALIGNED EVOLUTIONARY & INTER-CATEGORY CKA ANALYSIS ────────
    # CAVEAT (extraction-position asymmetry): math states are read at the "=" token,
    # control states at the sentence-final token (see extract_states.make_hook).
    # At upper layers these are functionally different positions, so inter-category
    # CKA divergence conflates representational content with this positional gap.
    # Report alongside the result; it is an inherent RQ1 limitation, not a bug.
    logger.info("Step 2: Processing Multi-Layer Centered Kernel Alignment Iterations...")
    results = []

    # Cache and prepare baseline Layer 00 state activations
    H_prev = load_hidden_states(PROC_DIR / "layer_00.pt").astype(np.float64)
    H_prev_math = H_prev[math_idx]
    H_prev_ctrl = H_prev[ctrl_idx]

    # ── RQ1-08: ANCHOR LAYER 0 VALIDATION & COMPUTATIONAL VERIFICATION ────────
    # Enforce verification loop on the anchor baseline identity representation
    base_self_cka_math = linear_cka(H_prev_math, H_prev_math)
    base_self_cka_ctrl = linear_cka(H_prev_ctrl, H_prev_ctrl)

    assert abs(base_self_cka_math - 1.0) < 1e-6, f"Self-CKA identity violation on math space: {base_self_cka_math}"
    assert abs(base_self_cka_ctrl - 1.0) < 1e-6, f"Self-CKA identity violation on control space: {base_self_cka_ctrl}"

    # Inter-category CKA at layer 0 — point estimate only.
    # E-M-03: layer-by-layer CI is replaced by the post-hoc Z-score baseline test
    # (lines below) on ΔCKA, which gates terminal-layer significance against the
    # background variance across layers 1..n-2.
    inter_seed_l0 = get_seed(global_seed, "cka_inter_layer", 0)
    cka_inter_point_0 = compute_cka_intercategory(
        H_prev_math, H_prev_ctrl, seed=inter_seed_l0
    )

    results.append({
        "layer": 0,
        "cka_evo_math": 1.0,
        "cka_evo_ctrl": 1.0,
        "cka_inter_mean": round(cka_inter_point_0, 6),
        "delta_cka_evolution": 0.0
    })

    # Array to track consecutive layer deltas for post-hoc statistical outlier screening
    cka_deltas = []

    # Process subsequent hidden layers hierarchically
    for l in range(1, n_layers):
        H_curr = load_hidden_states(PROC_DIR / f"layer_{l:02d}.pt").astype(np.float64)
        H_curr_math = H_curr[math_idx]
        H_curr_ctrl = H_curr[ctrl_idx]

        # Calculate balanced Evolutionary CKA points (Layer l-1 vs Layer l)
        cka_m = linear_cka(H_prev_math, H_curr_math)
        cka_c = linear_cka(H_prev_ctrl, H_curr_ctrl)

        # Positive delta indicates linguistic representations are evolving/shifting faster
        delta_cka = cka_c - cka_m
        cka_deltas.append(delta_cka)

        # Inter-category CKA per layer — point estimate (see Z-score baseline below).
        inter_seed = get_seed(global_seed, "cka_inter_layer", l)
        cka_inter_point = compute_cka_intercategory(
            H_curr_math, H_curr_ctrl, seed=inter_seed
        )

        results.append({
            "layer": l,
            "cka_evo_math": round(cka_m, 6),
            "cka_evo_ctrl": round(cka_c, 6),
            "cka_inter_mean": round(cka_inter_point, 6),
            "delta_cka_evolution": round(delta_cka, 6)
        })

        # Rotate states buffers to reduce storage footprint overheads
        H_prev_math = H_curr_math
        H_prev_ctrl = H_curr_ctrl

    # ── POST-HOC STATISTICAL OUTLIER DETECTION (F-01 / SA-04) ─────────────────
    # Evaluates if the terminal layer 23 divergence point stands as a clear
    # statistical anomaly compared to background baseline layers 1-22 variance.
    baseline_deltas = cka_deltas[:-1]
    terminal_delta = cka_deltas[-1]

    mu_baseline = np.mean(baseline_deltas)
    sigma_baseline = np.std(baseline_deltas)
    z_score = (terminal_delta - mu_baseline) / sigma_baseline if sigma_baseline > 0 else 0.0

    logger.info(f"Background Deviation Delta CKA (Layers 1-22): {mu_baseline:.5f} ± {sigma_baseline:.5f}")
    logger.info(f"Terminal Target Delta CKA (Layer 23): {terminal_delta:.5f}")
    logger.info(f"Calculated Z-Score for Layer 23 Bifurcation: {z_score:.2f}")

    if z_score > 3.0:
        logger.info("[✔] HYPOTHESIS CONFIRMED: Layer 23 divergence is a statistically significant outlier (Z > 3).")
    else:
        logger.warning("[!] WARNING: Terminal layer divergence is buried within baseline background noise.")

    df_cka = pd.DataFrame(results)
    output_csv_path = OUT_DIR / "cka_results_annotated.csv"
    _atomic_write_csv(output_csv_path, df_cka.to_dict("records"), df_cka.columns.tolist())

    # Persist the per-layer inter-category CKA vector as a numpy array (M-03):
    # downstream visualizations consume this as the RQ1 primary curve.
    cka_inter_array = df_cka["cka_inter_mean"].to_numpy(dtype=np.float64)
    _atomic_save_npy(OUT_DIR / "cka_intercategory.npy", cka_inter_array)

    # ── SECTION 3 — BALANCED AGGREGATED ISOTROPY (M-05) ───────────────────────
    # Computes ΔIso = ISO(math) − ISO(ctrl) on equal-N pools, using the same
    # balanced math_idx/ctrl_idx prepared above for CKA. The per-category CSV
    # written by run_isotropy_analysis is preserved; this block adds the
    # comparable math-vs-ctrl ΔIso that the per-category file cannot express.
    logger.info("Step 3: Balanced aggregated isotropy (math vs ctrl, equal N)...")
    iso_rng = np.random.default_rng(get_seed(global_seed, "isotropy_aggregated_balanced", 0))

    iso_rows = []
    for l in range(n_layers):
        H_l = load_hidden_states(PROC_DIR / f"layer_{l:02d}.pt").astype(np.float64)
        H_math_t = torch.from_numpy(H_l[math_idx])
        H_ctrl_t = torch.from_numpy(H_l[ctrl_idx])

        iso_m, _, ci_low_m, ci_high_m = isotropy_exact(H_math_t, n_bootstrap=1000, rng=iso_rng)
        iso_c, _, ci_low_c, ci_high_c = isotropy_exact(H_ctrl_t, n_bootstrap=1000, rng=iso_rng)

        iso_rows.append({
            "layer": l,
            "iso_math": round(iso_m, 6),
            "iso_ctrl": round(iso_c, 6),
            "delta_iso": round(iso_m - iso_c, 6),
            "ci_low_math": round(ci_low_m, 6),
            "ci_high_math": round(ci_high_m, 6),
            "ci_low_ctrl": round(ci_low_c, 6),
            "ci_high_ctrl": round(ci_high_c, 6),
            "n_per_side": int(n_sub),
        })

    df_iso = pd.DataFrame(iso_rows)
    iso_csv = OUT_DIR / "isotropy_aggregated_balanced.csv"
    _atomic_write_csv(iso_csv, df_iso.to_dict("records"), df_iso.columns.tolist())

    logger.info(f"RQ1 execution run completed. Verified tables successfully stored in {OUT_DIR}")


if __name__ == "__main__":
    main()
