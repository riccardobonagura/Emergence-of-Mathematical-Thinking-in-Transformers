#!/usr/bin/env python
"""run_rq3.py — RQ3: dynamics of the RQ1 geometry across fine-tuning.

Recomputes RQ1's geometry (ΔIso math vs ctrl, inter-category CKA) on the QLoRA
checkpoints, plus a cross-temporal CKA(base → checkpoint) drift — tracking how the
static RQ1 geometry evolves over the fine-tuning trajectory (thesis RQ3).

Reuse-only: imports the existing metric functions; does not modify run_rq1.py, run_rq4.py,
cka.py, or isotropy.py. Output: results/rq3_ft_dynamics/rq3_dynamics.csv.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.config.categories import CTRL_CATS, MATH_CATS
from src.metrics.cka import compute_cka_intercategory, linear_cka, subsample_indices
from src.metrics.isotropy import isotropy_exact
from src.probing.io_utils import (
    MetadataHandler,
    _atomic_write_csv,
    load_hidden_states,
)
from src.probing.seeds import get_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_rq3")

# Cross-temporal CKA subsample size (matches cka.py defaults; CKA stable for n >= 256).
CT_SUBSAMPLE = 512


def stimuli_hash(metadata_path: Path) -> str:
    """12-char digest of the ordered stimuli_ids — used to assert checkpoint alignment."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        ids = json.load(f).get("stimuli_ids", [])
    return hashlib.md5(str(ids).encode()).hexdigest()[:12]


def resolve_step(dir_name: str, config: dict) -> int:
    """Parse the training step from a checkpoint dir name.

    Mirrors run_rq4.py's convention: terminal adapter -> total_training_steps,
    otherwise the trailing integer of the (underscore-normalized) name.
    """
    if "final_adapter" in dir_name or "final_checkpoint" in dir_name:
        # Derive the final step from the config, falling back to run_rq4's default.
        return int(config.get("total_training_steps", 2000))
    try:
        return int(dir_name.replace("_", "-").split("-")[-1])
    except (ValueError, IndexError):
        logger.warning(f"Could not parse step from '{dir_name}'; defaulting to 0.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 fine-tuning geometry dynamics")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to the master configuration YAML (e.g. configs/config_rq2.yaml)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    global_seed = int(config["seed"])
    base_dir = Path("data/processed") / config["model_name"]
    ckpt_root = Path("data/processed/checkpoints_extracted")
    out_dir = Path("results/rq3_ft_dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_meta_path = base_dir / "metadata.json"
    if not base_meta_path.exists():
        logger.error(f"Base extraction metadata missing at {base_meta_path}.")
        sys.exit(1)

    meta = MetadataHandler(base_meta_path)
    n_layers = meta.get_n_layers()
    categories = np.array(meta.data.get("categories"))
    if categories is None or categories.size == 0:
        logger.error("Base metadata lacks a 'categories' array (pre-v5 format?).")
        sys.exit(1)

    # ── Checkpoint enumeration (runtime, not hardcoded) ───────────────────────
    base_hash = stimuli_hash(base_meta_path)
    steps = [(0, base_dir)]
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        if not d.is_dir() or not (d / "metadata.json").exists():
            continue
        steps.append((resolve_step(d.name, config), d))
    steps.sort(key=lambda x: x[0])

    n = len(steps)
    logger.info(
        f"n = {n} extracted states (design specifies ~25 at save-every-500); resolution "
        "is bound by what is on disk. Extracting the intermediate saved checkpoints would "
        "raise n with no methodological change."
    )
    logger.info(f"Base stimuli hash: {base_hash}")

    # Alignment guard: every checkpoint must share the base stimuli order.
    for step, d in steps:
        h = stimuli_hash(d / "metadata.json")
        if h != base_hash:
            logger.error(f"Stimuli hash mismatch at step {step} ({d}): {h} != base {base_hash}.")
            sys.exit(1)

    # ── Balanced math/ctrl indices — reuse run_rq1.py seed discipline ─────────
    math_idx_raw = np.where(np.isin(categories, list(MATH_CATS)))[0]
    ctrl_idx_raw = np.where(np.isin(categories, list(CTRL_CATS)))[0]
    n_sub = min(math_idx_raw.size, ctrl_idx_raw.size)
    rng_sub = np.random.default_rng(get_seed(global_seed, "rq1_subsampling", 0))
    math_idx = np.sort(rng_sub.choice(math_idx_raw, size=n_sub, replace=False))
    ctrl_idx = np.sort(rng_sub.choice(ctrl_idx_raw, size=n_sub, replace=False))
    logger.info(f"Balanced isotropy/inter-CKA pools: N={n_sub} per side.")

    # ── Cross-temporal subsample — new distinct purpose, shared across steps ──
    N_total = categories.size
    ct_idx = subsample_indices(
        n_total=N_total, n_sub=CT_SUBSAMPLE,
        # frozen: get_seed purpose string — renaming would change the derived seed, not an RQ label
        seed=get_seed(global_seed, "rq1_dynamics_crosstemporal", 0),
    )
    # Preload base cross-temporal subsamples once (float64, matching cka.py).
    base_ct = [
        load_hidden_states(base_dir / f"layer_{l:02d}.pt").astype(np.float64)[ct_idx]
        for l in range(n_layers)
    ]

    # ── Per (step, layer) metrics ─────────────────────────────────────────────
    rows = []
    drift_by_step = {}
    for step, d in steps:
        # Fresh bootstrap rng per step, same purpose/order as run_rq1 Section 3,
        # so the step-0 (base) rows reproduce results/rq1_emergence/ exactly.
        iso_rng = np.random.default_rng(get_seed(global_seed, "isotropy_aggregated_balanced", 0))
        drifts = []
        for l in range(n_layers):
            H_l = load_hidden_states(d / f"layer_{l:02d}.pt").astype(np.float64)

            H_math_t = torch.from_numpy(H_l[math_idx])
            H_ctrl_t = torch.from_numpy(H_l[ctrl_idx])
            iso_m, _, clm, chm = isotropy_exact(H_math_t, n_bootstrap=1000, rng=iso_rng)
            iso_c, _, clc, chc = isotropy_exact(H_ctrl_t, n_bootstrap=1000, rng=iso_rng)

            cka_inter = compute_cka_intercategory(
                H_l[math_idx], H_l[ctrl_idx],
                seed=get_seed(global_seed, "cka_inter_layer", l),
            )
            cka_vs_base = linear_cka(base_ct[l], H_l[ct_idx])
            drifts.append(1.0 - cka_vs_base)

            rows.append({
                "step": step,
                "layer": l,
                "iso_math": round(iso_m, 6),
                "iso_ctrl": round(iso_c, 6),
                "delta_iso": round(iso_m - iso_c, 6),
                "ci_low_math": round(clm, 6),
                "ci_high_math": round(chm, 6),
                "ci_low_ctrl": round(clc, 6),
                "ci_high_ctrl": round(chc, 6),
                "cka_inter": round(cka_inter, 6),
                "cka_vs_base": round(cka_vs_base, 6),
            })
        drift_by_step[step] = float(np.mean(drifts))

    out_csv = out_dir / "rq3_dynamics.csv"
    _atomic_write_csv(out_csv, rows, list(rows[0].keys()))
    logger.info(f"Wrote {len(rows)} rows to {out_csv}.")

    # ── Step-0 consistency backstop ───────────────────────────────────────────
    step0 = [r for r in rows if r["step"] == 0]
    d_iso_l15 = next((r["delta_iso"] for r in step0 if r["layer"] == 15), None)
    ct0 = [r["cka_vs_base"] for r in step0]
    logger.info(f"[check] step-0 delta_iso@L15 = {d_iso_l15} (expect ~ -0.1056)")
    logger.info(f"[check] step-0 cka_vs_base all == 1.0: {all(abs(v - 1.0) < 1e-9 for v in ct0)}")
    assert all(abs(v - 1.0) < 1e-9 for v in ct0), "Step-0 cross-temporal CKA must be 1.0 by construction."

    # ── Cross-temporal drift vs the T16 NF4 non-learning floor (read live) ────
    nf4_path = Path("results/nf4_degradation/summary.json")
    t16_ref = None
    if nf4_path.exists():
        with open(nf4_path, "r", encoding="utf-8") as f:
            nf4 = json.load(f)
        # Key confirmed as mean_frobenius_relative; fall back gracefully if renamed.
        t16_ref = nf4.get("mean_frobenius_relative")
    logger.info("Mean cross-temporal CKA drift (1 - CKA) per step:")
    for step in sorted(drift_by_step):
        logger.info(f"  step {step:>6}: {drift_by_step[step]:.4f}")
    logger.info(
        f"[T16 reference] NF4 mean relative Frobenius = {t16_ref} "
        "(non-learning floor, E-F-03; DIFFERENT metric/unit — Frobenius vs 1-CKA; "
        "no apples-to-apples CKA noise band exists for the cross-temporal comparison)."
    )


if __name__ == "__main__":
    main()
