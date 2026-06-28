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


def _write_rq5_csv(root: Path) -> None:
    rq5 = root / "results/rq5_determinization"
    rq5.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat in ("CAT-SIGN", "CAT-PARITY"):
        for step in (0, 2500, 12343):
            p = 0.1 + 0.00002 * step
            rows.append({
                "step": step, "category": cat, "n_rows": 1000,
                "n_single_token": 500 if cat == "CAT-SIGN" else 1000,
                "entropy_mean": 3.0 - 0.00005 * step, "margin_mean": 1.0 + 0.00003 * step,
                "p_first_token_mean": p, "p_correct_single": p,
                "p_correct_single_ci_lo": p - 0.02, "p_correct_single_ci_hi": p + 0.02,
            })
    pd.DataFrame(rows).to_csv(rq5 / "determinization.csv", index=False)


def test_rq5_dashboard_renders(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_rq5_csv(tmp_path)
    import importlib
    import src.viz.plot_rq5_determinization as m
    importlib.reload(m)
    m.main()
    assert (tmp_path / "results/figures/rq5/rq5_determinization.html").exists()


def _write_rq2_accuracy_csv(root: Path) -> None:
    rq2 = root / "results/rq2_probing"
    rq2.mkdir(parents=True, exist_ok=True)
    rows = []
    # sign: above 0.7 from L0. parity: sub-0.7 plateau then a clear jump at L3.
    sign_acc = [0.92, 0.98, 1.0, 1.0, 1.0]
    parity_acc = [0.50, 0.60, 0.65, 0.95, 0.97]
    for layer in range(5):
        for prop, acc in (("sign", sign_acc[layer]), ("parity", parity_acc[layer])):
            rows.append({
                "layer": layer, "property": prop, "accuracy": acc,
                "accuracy_lower_ci": acc - 0.03, "accuracy_upper_ci": acc + 0.03,
                "raw_p_value": 0.001, "is_significant": True,
            })
    pd.DataFrame(rows).to_csv(rq2 / "accuracy_metrics_corrected.csv", index=False)


def test_rq2_accuracy_figure_renders(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_rq2_accuracy_csv(tmp_path)
    import importlib
    import src.viz.plot_rq2_probing as m
    importlib.reload(m)

    df = pd.read_csv(tmp_path / "results/rq2_probing/accuracy_metrics_corrected.csv")
    # Emergence: sign at L0 (0.92 > 0.7), parity at L3 (first > 0.7).
    assert m.compute_emergence_layers(df) == {"sign": 0, "parity": 3}
    # Largest parity single-layer jump is L2→L3 (0.65 → 0.95).
    assert m.compute_jump_span(df, "parity") == (2, 3)

    m.main()
    assert (tmp_path / "results/figures/rq2/accuracy_curves.html").exists()
    assert (tmp_path / "results/figures/rq2/accuracy_curves.png").exists()


def test_pca_two_class_figure(tmp_path) -> None:
    import numpy as np
    from src.viz.pca_umap_viz import plot_layer_category_figures

    rng = np.random.default_rng(0)
    H = rng.standard_normal((40, 8)).astype(np.float32)
    categories = (["CAT-SIGN"] * 10 + ["CAT-PARITY"] * 10
                  + ["CTRL-NEU"] * 10 + ["CTRL-NUM"] * 10)

    fig = plot_layer_category_figures(H, categories, layer=23, out_dir=tmp_path, reducer="pca")

    assert (tmp_path / "pca_2class_layer_23.html").exists()
    assert (tmp_path / "pca_4way_layer_23.png").exists()
    # Exactly two color groups: math and ctrl.
    assert len(fig.data) == 2
    assert {tr.name for tr in fig.data} == {"math", "ctrl"}


def test_rq2_confound_effect_figure(tmp_path) -> None:
    from src.viz.probing_viz import plot_effect_vs_significance

    layers = list(range(4))
    # sign effect ~0.6 all-significant; parity effect ~0.15 all-significant; one NaN row.
    sign_df = pd.DataFrame({
        "layer": layers,
        "sign_logits_correlation_with_op1": [0.61, 0.59, 0.62, 0.60],
        "is_significant_op1_leak": [True, True, True, True],
    })
    parity_df = pd.DataFrame({
        "layer": layers,
        "parity_logits_correlation_with_op2parity": [0.15, 0.14, float("nan"), 0.16],
        "is_significant_op2_leak": [True, True, False, True],
    })
    out = tmp_path / "fig"
    out.mkdir()
    tidy = plot_effect_vs_significance(
        sign_df, parity_df, out / "c.png", out / "c.html")

    assert (out / "c.html").exists() and (out / "c.png").exists()
    # Both confound series present; sign effect dwarfs parity (E-M-03).
    assert set(tidy["confound"]) == {"sign↔op1", "parity↔op2-parity"}
    means = tidy.groupby("confound")["effect_abs"].mean()
    assert means["sign↔op1"] > 0.5 > means["parity↔op2-parity"]
    # NaN parity row dropped → parity has 3 rows; significance maps verbatim.
    parity_rows = tidy[tidy["confound"] == "parity↔op2-parity"]
    assert len(parity_rows) == 3
    assert parity_rows["significant"].all()  # the only False row was the dropped NaN
    assert tidy[tidy["confound"] == "sign↔op1"]["significant"].all()
