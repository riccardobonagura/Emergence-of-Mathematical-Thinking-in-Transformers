"""Smoke tests: each touched viz entry point renders HTML from synthetic CSVs."""

import sys
from pathlib import Path

import pandas as pd
import pytest


def _write_rq1_csvs(root: Path) -> None:
    rq1 = root / "results/rq1_emergence"
    rq1.mkdir(parents=True, exist_ok=True)
    layers = list(range(5))
    pd.DataFrame({
        "layer": layers,
        "cka_evo_math": [1.0, 0.9, 0.85, 0.8, 0.78],
        "cka_evo_ctrl": [1.0, 0.92, 0.88, 0.83, 0.8],
        "cka_inter_mean": [0.7, 0.6, 0.55, 0.5, 0.45],
        "cka_inter_ci_low": [0.66, 0.56, 0.5, 0.45, 0.4],
        "cka_inter_ci_high": [0.74, 0.64, 0.6, 0.55, 0.5],
        "cka_ctrl_neu_vs_num": [0.8, 0.75, 0.7, 0.68, 0.66],
        "cka_math_template_baseline": [None, None, None, None, None],
        "delta_cka_evolution": [0.0, 0.02, 0.03, 0.03, 0.02],
    }).to_csv(rq1 / "cka_results_annotated.csv", index=False)

    pd.DataFrame({
        "layer": layers,
        "iso_math": [0.5, 0.55, 0.6, 0.62, 0.63],
        "iso_ctrl": [0.55, 0.6, 0.66, 0.69, 0.7],
        "delta_iso": [-0.05, -0.05, -0.06, -0.07, -0.07],
        "ci_low_math": [0.48, 0.53, 0.58, 0.6, 0.61],
        "ci_high_math": [0.52, 0.57, 0.62, 0.64, 0.65],
        "ci_low_ctrl": [0.53, 0.58, 0.64, 0.67, 0.68],
        "ci_high_ctrl": [0.57, 0.62, 0.68, 0.71, 0.72],
        "n_per_side": [60] * 5,
    }).to_csv(rq1 / "isotropy_aggregated_balanced.csv", index=False)


def _write_rq3_csv(root: Path) -> None:
    dyn = root / "results/rq2_probing/dynamic"
    dyn.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in (0, 2500):
        for layer in range(3):
            rows.append({
                "step": step, "layer": layer, "property": "sign",
                "probing_acc": 0.5 + 0.01 * layer + 0.0001 * step,
                "geom_delta_math": 0.0 if step == 0 else 0.02 * (layer + 1),
                "geom_delta_ctrl": 0.0 if step == 0 else 0.01 * (layer + 1),
                "geom_delta_math_rel": 0.0 if step == 0 else 0.03 * (layer + 1),
                "geom_delta_ctrl_rel": 0.0 if step == 0 else 0.02 * (layer + 1),
            })
    pd.DataFrame(rows).to_csv(dyn / "trajectories_probing.csv", index=False)


def test_rq1_dashboard_renders(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_rq1_csvs(tmp_path)
    import importlib
    import src.viz.plot_rq1_emergence as m
    importlib.reload(m)
    m.plot_rq1_dashboard()
    assert (tmp_path / "results/figures/rq1_emergence/rq1_emergence.html").exists()


def test_rq3_dashboard_renders(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_rq3_csv(tmp_path)
    import importlib
    import src.viz.plot_rq3_trajectory as m
    importlib.reload(m)
    m.main()
    assert (tmp_path / "results/figures/rq3/rq3_dashboard.html").exists()
