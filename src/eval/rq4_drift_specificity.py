"""
rq4_drift_specificity.py — Standalone diagnostic: math-vs-ctrl drift specificity (post-RQ4).

Runs after RQ4, analogous to nf4_degradation.py and run_confound_checks.py: a standalone
diagnostic, NOT invoked by the orchestrators. It reads the RQ4 trajectory CSV and, at a chosen
training step, contrasts the RELATIVE geometric drift of the math subset against the ctrl subset
per layer, then issues a specificity verdict — math-specific reorganization vs a global
representational shift.

Why the relative columns (geom_delta_math_rel / geom_delta_ctrl_rel) and NOT the dimension-
normalized ones: each *_rel value is normalized by its OWN base-representation Frobenius norm, so
it reads as "the fraction of its own representation that moved". That makes math and ctrl directly
comparable even when their absolute magnitudes differ, and it is the only formulation comparable
to the NF4 quantization floor reported by T16 (nf4_degradation.py).

Dedup: in run_rq4.py the geometric drift is computed once per (step, layer) and copied onto both
the sign and parity rows. Those columns are therefore identical across the two properties of the
same (step, layer); we keep a single row per layer via drop_duplicates(['step', 'layer']).
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, TypedDict

import pandas as pd
import yaml

from src.probing.io_utils import _atomic_write_json, setup_logging

logger = logging.getLogger("probing")

REQUIRED_COLUMNS = ["step", "layer", "geom_delta_math_rel", "geom_delta_ctrl_rel"]

# Fraction of layers with math > ctrl required for the "predominantly math-specific" verdict.
PREDOMINANT_THRESHOLD = 0.75


class LayerDriftRow(TypedDict):
    """Per-layer math-vs-ctrl relative-drift comparison at one training step (ARCH-03)."""
    layer: int
    math_rel: float
    ctrl_rel: float
    ratio: Optional[float]   # math_rel / ctrl_rel; None when ctrl_rel == 0 (n/a)
    math_gt_ctrl: bool


def classify_specificity(
    per_layer: List[LayerDriftRow],
    mean_math_rel: float,
    mean_ctrl_rel: float,
) -> str:
    """Issue a deliberately conservative specificity verdict (no over-claiming).

    Ladder (first match wins):
      - math > ctrl at ALL layers AND mean math > mean ctrl -> clean math-specific signal
      - math > ctrl in >= 75% of layers                     -> predominantly math-specific
      - mean math > mean ctrl but not at every layer        -> mixed, report per-layer only
      - otherwise                                           -> global representational shift
    """
    n_layers = len(per_layer)
    n_math_gt_ctrl = sum(1 for r in per_layer if r["math_gt_ctrl"])
    mean_gt = mean_math_rel > mean_ctrl_rel

    if n_math_gt_ctrl == n_layers and mean_gt:
        return (
            "math-specific: relative drift exceeds ctrl at every layer and on average — "
            "consistent with math-specific reorganization"
        )
    if n_math_gt_ctrl >= PREDOMINANT_THRESHOLD * n_layers:
        return (
            "predominantly math-specific (note the exceptions): math drift exceeds ctrl in "
            f"{n_math_gt_ctrl}/{n_layers} layers — report the layers where it does not"
        )
    if mean_gt:
        return (
            "mixed: mean math drift exceeds ctrl but not at every layer — report per-layer, "
            "no clean specificity claim"
        )
    return (
        "global drift, not math-specific: ctrl drift matches or exceeds math — "
        "frame as a global representational shift, not math-specific reorganization"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ4 math-vs-ctrl relative-drift specificity diagnostic (post-RQ4)."
    )
    parser.add_argument("--config", required=True, type=str,
                        help="Path to operational config file (e.g., configs/config_rq2.yaml)")
    parser.add_argument("--step", type=int, default=None,
                        help="Training step to analyze (default: the terminal step = max step).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    traj_path = Path(config.get(
        "rq4_trajectory_csv", "results/rq4_drift/trajectories_probing.csv"
    ))
    out_dir = traj_path.parent
    setup_logging(out_dir)

    if not traj_path.exists():
        raise FileNotFoundError(
            f"RQ4 trajectory CSV missing: {traj_path}. Run run_rq4.py first to produce it."
        )

    df = pd.read_csv(traj_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"RQ4 trajectory CSV {traj_path} is missing required columns: {missing}. "
            f"Present columns: {df.columns.tolist()}."
        )

    # Dedup: drift is identical across sign/parity rows of the same (step, layer).
    df = df.drop_duplicates(subset=["step", "layer"], keep="first")

    available_steps = sorted(int(s) for s in df["step"].unique())
    target_step = args.step if args.step is not None else max(available_steps)

    if target_step not in available_steps:
        raise ValueError(
            f"Requested step {target_step} not present in {traj_path}. "
            f"Available steps: {available_steps}."
        )

    total_steps = config.get("total_training_steps")
    if total_steps is not None and target_step == int(total_steps):
        logger.info(f"Step {target_step} is the terminal training step (total_training_steps).")

    step_df = df[df["step"] == target_step].sort_values("layer")

    per_layer: List[LayerDriftRow] = []
    for _, row in step_df.iterrows():
        math_rel = float(row["geom_delta_math_rel"])
        ctrl_rel = float(row["geom_delta_ctrl_rel"])
        ratio = round(math_rel / ctrl_rel, 4) if ctrl_rel != 0.0 else None
        per_layer.append({
            "layer": int(row["layer"]),
            "math_rel": round(math_rel, 6),
            "ctrl_rel": round(ctrl_rel, 6),
            "ratio": ratio,
            "math_gt_ctrl": math_rel > ctrl_rel,
        })

    n_layers = len(per_layer)
    mean_math_rel = round(sum(r["math_rel"] for r in per_layer) / n_layers, 6) if n_layers else 0.0
    mean_ctrl_rel = round(sum(r["ctrl_rel"] for r in per_layer) / n_layers, 6) if n_layers else 0.0
    n_math_gt_ctrl = sum(1 for r in per_layer if r["math_gt_ctrl"])

    max_row = max(per_layer, key=lambda r: r["math_rel"])
    max_math_drift = max_row["math_rel"]
    max_math_layer = max_row["layer"]

    verdict = classify_specificity(per_layer, mean_math_rel, mean_ctrl_rel)

    # ── Readable per-layer table ──
    logger.info(f"RQ4 drift specificity @ step {target_step} ({n_layers} layers)")
    logger.info(f"{'layer':>5} | {'math_rel':>10} | {'ctrl_rel':>10} | {'math/ctrl':>10} | math>ctrl")
    logger.info("-" * 58)
    for r in per_layer:
        ratio_str = f"{r['ratio']:.4f}" if r["ratio"] is not None else "n/a"
        marker = " <-- max math" if r["layer"] == max_math_layer else ""
        logger.info(
            f"{r['layer']:>5} | {r['math_rel']:>10.6f} | {r['ctrl_rel']:>10.6f} | "
            f"{ratio_str:>10} | {str(r['math_gt_ctrl']):>5}{marker}"
        )

    # ── Aggregates ──
    logger.info("-" * 58)
    logger.info(f"mean math_rel      : {mean_math_rel:.6f}")
    logger.info(f"mean ctrl_rel      : {mean_ctrl_rel:.6f}")
    logger.info(f"layers math>ctrl   : {n_math_gt_ctrl}/{n_layers}")
    logger.info(f"max math drift     : {max_math_drift:.6f} @ layer {max_math_layer}")
    logger.info(f"VERDICT            : {verdict}")

    summary = {
        "step": target_step,
        "per_layer": per_layer,
        "mean_math_rel": mean_math_rel,
        "mean_ctrl_rel": mean_ctrl_rel,
        "n_math_gt_ctrl": n_math_gt_ctrl,
        "n_layers": n_layers,
        "max_math_drift": max_math_drift,
        "max_math_layer": max_math_layer,
        "verdict": verdict,
    }
    _atomic_write_json(out_dir / "drift_specificity_summary.json", summary)
    logger.info(f"Summary written to {out_dir / 'drift_specificity_summary.json'}")


if __name__ == "__main__":
    main()
