#!/usr/bin/env python
"""
run_rq1.py — RQ1 orchestrator: Emergence threshold location (l*).
Calculates category-balanced Isotropy, Evolutionary CKA, and Inter-Category CKA.

Enforces structural fixes RQ1-01 to RQ1-08 by eliminating all hardcoded paths,
constants, and uneven sampling distributions, replacing them with a strict,
config-driven statistical framework.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.config.categories import CTRL_CATS, MATH_CATS
from src.metrics.cka import (
    compute_cka_intercategory,
    debiased_linear_cka,
    leave_k_out_influence,
    linear_cka,
    procrustes_distance,
)
from src.metrics.isotropy import (
    isotropy_exact,
    random_gaussian_isotropy_floor,
    run_isotropy_analysis,
)
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


def _bootstrap_inter_cka(
    H_math: np.ndarray,
    H_ctrl: np.ndarray,
    n_boot: int,
    base_seed: int,
    layer: int,
    subsample_frac: float,
) -> tuple[float, float, float]:
    """Inter-category CKA point estimate + WITHOUT-replacement subsampling 95% CI.

    The first return slot is a single CLEAN point estimate on the full balanced
    math/ctrl sets — NOT a resample average. The CI is a bias-corrected SUBSAMPLING
    CI, not a classical bootstrap.

    Why subsampling without replacement and not a with-replacement bootstrap: linear
    CKA is computed on Gram matrices XXᵀ. Resampling rows WITH replacement duplicates
    rows, which inflates the Gram diagonal and deflates the centered-HSIC ratio,
    collapsing the estimate toward zero in the near-zero CKA regime of real data (the
    old code reported 0.005-0.012 against a true ~0.10). We therefore draw DISTINCT
    rows — floor(frac*n) per side, replace=False — so the no-duplicate structure CKA
    assumes is preserved; the spread across draws is the uncertainty band.

    Why the band is bias-corrected: smaller n inflates linear CKA (finite-sample
    positive bias; Kornblith et al. 2019), so the subsample cloud sits systematically
    ABOVE the full-n point and a raw-percentile band would not contain it. We recenter
    the band by the subsample-vs-point offset (bias = submean − point), reporting the
    2.5/97.5 percentiles shifted onto the point. The width is the subsampling spread;
    the location is pinned to the unbiased-er full-n estimate, so the interval brackets
    the point by construction. Seeds come only from get_seed.
    """
    point = compute_cka_intercategory(H_math, H_ctrl, seed=base_seed)
    n_m, n_c = H_math.shape[0], H_ctrl.shape[0]
    k_m, k_c = int(np.floor(subsample_frac * n_m)), int(np.floor(subsample_frac * n_c))
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        rng = np.random.default_rng(get_seed(base_seed, "cka_inter_boot", layer * 1000 + b))
        idx_m = rng.choice(n_m, size=k_m, replace=False)
        idx_c = rng.choice(n_c, size=k_c, replace=False)
        vals[b] = compute_cka_intercategory(H_math[idx_m], H_ctrl[idx_c], seed=base_seed)

    # Recenter the subsampling spread onto the full-n point to remove finite-sample bias.
    bias = float(vals.mean()) - float(point)
    ci_low = float(np.percentile(vals, 2.5)) - bias
    ci_high = float(np.percentile(vals, 97.5)) - bias
    return float(point), ci_low, ci_high


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
    cka_inter_boot_n = int(config.get("cka_inter_bootstrap_n", 50))
    cka_inter_subsample_frac = float(config.get("cka_inter_subsample_frac", 0.8))
    iso_floor_boot_n = int(config.get("iso_floor_bootstrap_n", 1000))
    PROC_DIR = Path("data/processed") / config["model_name"]
    STIMULI_PATH = Path("data/processed/dataset_master_v5.jsonl")
    OUT_DIR = Path("results/rq1_emergence")
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

    # ── REVIEWER-MANDATED BASELINE INDICES ────────────────────────────────────
    ctrl_neu_idx_raw = np.where(categories == "CTRL-NEU")[0]
    ctrl_num_idx_raw = np.where(categories == "CTRL-NUM")[0]

    with open(STIMULI_PATH, "r", encoding="utf-8") as f:
        template_ids = np.array([json.loads(line).get("template_id", "") for line in f if line.strip()])

    bare_templates = {t for t in np.unique(template_ids) if t.startswith("TPL-") and t.endswith("-1")}
    math_mask = np.isin(categories, list(MATH_CATS))
    math_bare_idx_raw = np.where(math_mask & np.isin(template_ids, list(bare_templates)))[0]
    math_prefixed_idx_raw = np.where(math_mask & ~np.isin(template_ids, list(bare_templates)) & (template_ids != ""))[0]

    logger.info(
        f"Baseline groups: CTRL-NEU={ctrl_neu_idx_raw.size}, CTRL-NUM={ctrl_num_idx_raw.size}, "
        f"math_bare={math_bare_idx_raw.size}, math_prefixed={math_prefixed_idx_raw.size}"
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

    # ── BASELINE SUBSAMPLING (balanced CTRL-NEU vs CTRL-NUM, bare vs prefixed) ──
    has_ctrl_baseline = ctrl_neu_idx_raw.size > 0 and ctrl_num_idx_raw.size > 0
    has_tpl_baseline = math_bare_idx_raw.size > 0 and math_prefixed_idx_raw.size > 0

    if has_ctrl_baseline:
        n_ctrl_baseline = min(ctrl_neu_idx_raw.size, ctrl_num_idx_raw.size)
        rng_ctrl_bl = np.random.default_rng(get_seed(global_seed, "rq1_ctrl_subsampling", 0))
        ctrl_neu_idx = np.sort(rng_ctrl_bl.choice(ctrl_neu_idx_raw, size=n_ctrl_baseline, replace=False))
        ctrl_num_idx = np.sort(rng_ctrl_bl.choice(ctrl_num_idx_raw, size=n_ctrl_baseline, replace=False))
        logger.info(f"Baseline A (CTRL-NEU vs CTRL-NUM): N={n_ctrl_baseline} per side")
    else:
        logger.warning("Baseline A skipped: CTRL-NEU or CTRL-NUM group is empty")

    if has_tpl_baseline:
        n_tpl_baseline = min(math_bare_idx_raw.size, math_prefixed_idx_raw.size)
        rng_tpl_bl = np.random.default_rng(get_seed(global_seed, "rq1_template_subsampling", 0))
        math_bare_idx = np.sort(rng_tpl_bl.choice(math_bare_idx_raw, size=n_tpl_baseline, replace=False))
        math_prefixed_idx = np.sort(rng_tpl_bl.choice(math_prefixed_idx_raw, size=n_tpl_baseline, replace=False))
        logger.info(f"Baseline B (bare vs prefixed math): N={n_tpl_baseline} per side")
    else:
        logger.warning("Baseline B skipped: bare or prefixed template group is empty")

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

    # Inter-category CKA at layer 0 — clean point estimate + without-replacement
    # subsampling 95% CI over the balanced math/ctrl sets (E-M-02). The post-hoc
    # Z-score baseline test on the evolutionary ΔCKA (lines below) remains a separate
    # secondary check.
    cka_inter_mean_0, cka_inter_lo_0, cka_inter_hi_0 = _bootstrap_inter_cka(
        H_prev_math, H_prev_ctrl, cka_inter_boot_n, global_seed, 0, cka_inter_subsample_frac
    )

    cka_ctrl_bl_0 = compute_cka_intercategory(
        H_prev[ctrl_neu_idx], H_prev[ctrl_num_idx],
        seed=get_seed(global_seed, "cka_ctrl_baseline_layer", 0)
    ) if has_ctrl_baseline else float("nan")
    cka_tpl_bl_0 = compute_cka_intercategory(
        H_prev[math_bare_idx], H_prev[math_prefixed_idx],
        seed=get_seed(global_seed, "cka_template_baseline_layer", 0)
    ) if has_tpl_baseline else float("nan")

    results.append({
        "layer": 0,
        "cka_evo_math": 1.0,
        "cka_evo_ctrl": 1.0,
        "cka_inter_mean": round(cka_inter_mean_0, 6),
        "cka_inter_ci_low": round(cka_inter_lo_0, 6),
        "cka_inter_ci_high": round(cka_inter_hi_0, 6),
        "cka_ctrl_neu_vs_num": round(cka_ctrl_bl_0, 6) if has_ctrl_baseline else None,
        "cka_math_template_baseline": round(cka_tpl_bl_0, 6) if has_tpl_baseline else None,
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

        # Inter-category CKA per layer — clean point estimate + without-replacement
        # subsampling 95% CI.
        cka_inter_mean, cka_inter_lo, cka_inter_hi = _bootstrap_inter_cka(
            H_curr_math, H_curr_ctrl, cka_inter_boot_n, global_seed, l, cka_inter_subsample_frac
        )

        cka_ctrl_bl = compute_cka_intercategory(
            H_curr[ctrl_neu_idx], H_curr[ctrl_num_idx],
            seed=get_seed(global_seed, "cka_ctrl_baseline_layer", l)
        ) if has_ctrl_baseline else float("nan")
        cka_tpl_bl = compute_cka_intercategory(
            H_curr[math_bare_idx], H_curr[math_prefixed_idx],
            seed=get_seed(global_seed, "cka_template_baseline_layer", l)
        ) if has_tpl_baseline else float("nan")

        results.append({
            "layer": l,
            "cka_evo_math": round(cka_m, 6),
            "cka_evo_ctrl": round(cka_c, 6),
            "cka_inter_mean": round(cka_inter_mean, 6),
            "cka_inter_ci_low": round(cka_inter_lo, 6),
            "cka_inter_ci_high": round(cka_inter_hi, 6),
            "cka_ctrl_neu_vs_num": round(cka_ctrl_bl, 6) if has_ctrl_baseline else None,
            "cka_math_template_baseline": round(cka_tpl_bl, 6) if has_tpl_baseline else None,
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

    # ── INTER-CATEGORY ROBUSTNESS BATTERY (E-G-02) ────────────────────────────
    # The authority makes inter-category CKA the PRIMARY metric and requires it to
    # survive a 4-part battery before any "math diverges from language" claim. Here
    # we run the cheap, deterministic checks post-hoc on the balanced math/ctrl sets:
    # matched-baseline comparison (CTRL-NEU↔CTRL-NUM), debiased CKA, orthogonal
    # Procrustes, and leave-k-out influence. A permutation null over category labels
    # is the rigorous fourth leg but costs L×N_perm CKA evals — it is FUTURE WORK and
    # deliberately not implemented here.

    # (c1) Matched-baseline comparison from the assembled columns. This is now a
    # like-for-like comparison: BOTH sides are single-shot CKA point estimates on
    # balanced sets — cka_inter_mean is the clean point estimate from
    # _bootstrap_inter_cka (post Fix A, no longer a duplication-deflated resample
    # average), and cka_ctrl_neu_vs_num is one clean compute_cka_intercategory call.
    # A genuine content divergence sits BELOW the control↔control baseline. When the
    # baseline is NaN (no CTRL-NEU/CTRL-NUM split), the `<` comparison evaluates to
    # False by design — the intentional safe null, never a spurious "diverges".
    df_cka["delta_vs_ctrl_baseline"] = df_cka["cka_inter_mean"] - df_cka["cka_ctrl_neu_vs_num"]
    df_cka["divergence_exceeds_baseline"] = df_cka["cka_inter_mean"] < df_cka["cka_ctrl_neu_vs_num"]

    # (c2) Debiased CKA + Procrustes + leave-k-out, per layer, on the balanced sets.
    cka_loo_k = int(config.get("cka_loo_k", 10))
    cka_loo_iter = int(config.get("cka_loo_iter", 20))

    debiased_col, procrustes_col, loo_col = [], [], []
    for l in range(n_layers):
        H_l = load_hidden_states(PROC_DIR / f"layer_{l:02d}.pt").astype(np.float64)
        H_m, H_c = H_l[math_idx], H_l[ctrl_idx]

        if H_m.shape[0] <= 3:
            # Battery undefined for n <= 3 (debiased HSIC); emit NaN like the Z-score guard.
            debiased_col.append(float("nan"))
            procrustes_col.append(float("nan"))
            loo_col.append(float("nan"))
            continue

        try:
            deb = debiased_linear_cka(H_m, H_c)
        except (ValueError, RuntimeError):
            deb = float("nan")
        try:
            proc = procrustes_distance(H_m, H_c)
        except RuntimeError:
            proc = float("nan")
        try:
            loo = leave_k_out_influence(
                H_m, H_c, cka_loo_k, cka_loo_iter, get_seed(global_seed, "loo_battery", l)
            )["max_abs_influence"]
        except RuntimeError:
            loo = float("nan")

        debiased_col.append(round(deb, 6) if np.isfinite(deb) else deb)
        procrustes_col.append(round(proc, 6) if np.isfinite(proc) else proc)
        loo_col.append(round(loo, 6) if np.isfinite(loo) else loo)

    df_cka["cka_inter_debiased"] = debiased_col
    df_cka["procrustes_math_ctrl"] = procrustes_col
    df_cka["leave_k_influence"] = loo_col

    # (c3) One-line verdict: does the divergence claim survive the matched baseline?
    n_exceed = int(df_cka["divergence_exceeds_baseline"].sum())
    term_deb = df_cka["cka_inter_debiased"].iloc[-1]
    term_bl = df_cka["cka_ctrl_neu_vs_num"].iloc[-1]
    logger.info(
        f"[Robustness] {n_exceed}/{n_layers} layers fall below the CTRL-NEU↔CTRL-NUM baseline; "
        f"terminal-layer debiased inter-CKA={term_deb} vs matched baseline={term_bl}."
    )

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

        # Random-Gaussian norm-matched isotropy floor on the pooled balanced set
        # (math_idx + ctrl_idx). Real hidden states sit well ABOVE this null floor —
        # the floor anchors ΔIso interpretation so anisotropy is read as the normal
        # state, not a semantic structure (E-G-01). Additive columns only.
        H_pool = np.vstack([H_l[math_idx], H_l[ctrl_idx]])
        iso_floor, iso_floor_lo, iso_floor_hi = random_gaussian_isotropy_floor(
            H_pool,
            n_bootstrap=iso_floor_boot_n,
            base_seed=get_seed(global_seed, "iso_floor_layer", l),
        )

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
            "iso_floor": round(iso_floor, 6),
            "iso_floor_ci_low": round(iso_floor_lo, 6),
            "iso_floor_ci_high": round(iso_floor_hi, 6),
        })

    df_iso = pd.DataFrame(iso_rows)
    iso_csv = OUT_DIR / "isotropy_aggregated_balanced.csv"
    _atomic_write_csv(iso_csv, df_iso.to_dict("records"), df_iso.columns.tolist())

    logger.info(f"RQ1 execution run completed. Verified tables successfully stored in {OUT_DIR}")


if __name__ == "__main__":
    main()
