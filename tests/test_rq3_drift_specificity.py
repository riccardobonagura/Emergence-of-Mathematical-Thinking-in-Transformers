"""CPU-only tests for the RQ3 drift-specificity diagnostic.

Covers the per-layer dedup (drift is duplicated across sign/parity rows) and the four
branches of the specificity verdict ladder. No GPU, no RNG.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.eval.rq3_drift_specificity import classify_specificity, main


def _layer(layer: int, math_rel: float, ctrl_rel: float) -> dict:
    """Build one LayerDriftRow-shaped dict for verdict-branch tests."""
    ratio = round(math_rel / ctrl_rel, 4) if ctrl_rel != 0.0 else None
    return {
        "layer": layer,
        "math_rel": math_rel,
        "ctrl_rel": ctrl_rel,
        "ratio": ratio,
        "math_gt_ctrl": math_rel > ctrl_rel,
    }


# ── Verdict ladder: four branches ─────────────────────────────────────────────

def test_verdict_math_specific_all_layers() -> None:
    rows = [_layer(l, 0.6, 0.3) for l in range(4)]  # math > ctrl everywhere
    verdict = classify_specificity(rows, mean_math_rel=0.6, mean_ctrl_rel=0.3)
    assert verdict.startswith("math-specific")


def test_verdict_predominantly_math_specific() -> None:
    # 3/4 layers math > ctrl (>= 75%) but not all -> predominantly.
    rows = [_layer(0, 0.6, 0.3), _layer(1, 0.6, 0.3), _layer(2, 0.6, 0.3), _layer(3, 0.2, 0.5)]
    verdict = classify_specificity(rows, mean_math_rel=0.5, mean_ctrl_rel=0.35)
    assert verdict.startswith("predominantly math-specific")


def test_verdict_mixed() -> None:
    # 2/4 layers math > ctrl (< 75%) but mean math > mean ctrl -> mixed.
    rows = [_layer(0, 0.9, 0.1), _layer(1, 0.9, 0.1), _layer(2, 0.1, 0.4), _layer(3, 0.1, 0.4)]
    mean_math = sum(r["math_rel"] for r in rows) / 4  # 0.5
    mean_ctrl = sum(r["ctrl_rel"] for r in rows) / 4  # 0.25
    verdict = classify_specificity(rows, mean_math, mean_ctrl)
    assert verdict.startswith("mixed")


def test_verdict_global_drift() -> None:
    rows = [_layer(l, 0.2, 0.5) for l in range(4)]  # ctrl >= math everywhere
    verdict = classify_specificity(rows, mean_math_rel=0.2, mean_ctrl_rel=0.5)
    assert verdict.startswith("global drift")


# ── Per-layer dedup through main() on a synthetic CSV ─────────────────────────

def test_dedup_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two rows (sign + parity) per (step, layer) must collapse to one layer row."""
    dyn_dir = tmp_path / "dynamic"
    dyn_dir.mkdir()
    csv_path = dyn_dir / "trajectories_probing.csv"

    rows = []
    for layer in range(3):
        math_rel = 0.6 - 0.1 * layer   # 0.6, 0.5, 0.4 -> max at layer 0
        ctrl_rel = 0.3
        for prop in ("sign", "parity"):
            rows.append({
                "step": 100, "layer": layer, "property": prop,
                "probing_acc": 0.8,
                "geom_delta_math": 1.0, "geom_delta_ctrl": 0.5,
                "geom_delta_math_rel": math_rel, "geom_delta_ctrl_rel": ctrl_rel,
                "gsm8k_acc": 0.0, "gsm8k_ci_lower": 0.0, "gsm8k_ci_upper": 0.0,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"rq3_trajectory_csv: {csv_path}\ntotal_training_steps: 100\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])
    main()

    summary = json.loads((dyn_dir / "drift_specificity_summary.json").read_text(encoding="utf-8"))
    assert summary["step"] == 100
    assert summary["n_layers"] == 3                 # dedup: 6 rows -> 3 layers
    assert len(summary["per_layer"]) == 3
    assert summary["n_math_gt_ctrl"] == 3
    assert summary["max_math_layer"] == 0
    assert summary["max_math_drift"] == pytest.approx(0.6)
    assert summary["verdict"].startswith("math-specific")


def test_missing_step_lists_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dyn_dir = tmp_path / "dynamic"
    dyn_dir.mkdir()
    csv_path = dyn_dir / "trajectories_probing.csv"
    pd.DataFrame([{
        "step": 100, "layer": 0, "property": "sign",
        "geom_delta_math_rel": 0.6, "geom_delta_ctrl_rel": 0.3,
    }]).to_csv(csv_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"rq3_trajectory_csv: {csv_path}\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path), "--step", "999"])
    with pytest.raises(ValueError, match=r"Available steps: \[100\]"):
        main()
