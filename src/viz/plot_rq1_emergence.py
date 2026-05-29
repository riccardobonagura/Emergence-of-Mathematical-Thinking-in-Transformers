"""
plot_rq1_emergence.py — RQ1 interactive dashboard (Plotly HTML).

Reads:  results/rq1_emergence/isotropy_pythia.csv
        results/rq1_emergence/cka_math_evol.npy
        results/rq1_emergence/cka_ctrl_evol.npy

Outputs: results/figures/rq1_emergence/rq1_emergence.html

Delta isotropy:
    ΔIso(l) = mean_iso(l, {CAT-SIGN, CAT-PARITY}) - mean_iso(l, {CTRL-NEU, CTRL-NUM})
    ISO high = anisotropic (vectors collinear); ISO low = isotropic (vectors spread).
    Negative ΔIso → math more isotropic than control at layer l.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.config.categories import MATH_CATS, CTRL_CATS


def plot_rq1_dashboard() -> None:
    RESULTS_DIR = Path("results/rq1_emergence")
    OUT_DIR     = Path("results/figures/rq1_emergence")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    iso_file      = RESULTS_DIR / "isotropy_pythia.csv"
    cka_math_file = RESULTS_DIR / "cka_math_evol.npy"
    cka_ctrl_file = RESULTS_DIR / "cka_ctrl_evol.npy"

    if not all(p.exists() for p in (iso_file, cka_math_file, cka_ctrl_file)):
        raise FileNotFoundError(
            "RQ1 data missing — run run_rq1.py first."
        )

    df_iso   = pd.read_csv(iso_file)
    cka_math = np.load(cka_math_file)
    cka_ctrl = np.load(cka_ctrl_file)

    n_layers = len(cka_math)
    layers   = np.arange(n_layers)

    # Aggregate isotropy: per-layer mean across the two math categories and two ctrl categories.
    # run_isotropy_analysis produces one row per (layer, category);
    # groupby+mean collapses CAT-SIGN and CAT-PARITY into one math curve.
    def mean_iso(cats: tuple) -> np.ndarray:
        return (
            df_iso[df_iso["category"].isin(cats)]
            .groupby("layer")["iso_mean"]
            .mean()
            .sort_index()
            .values
        )

    iso_math  = mean_iso(MATH_CATS)
    iso_ctrl  = mean_iso(CTRL_CATS)
    delta_iso = iso_math - iso_ctrl

    if len(delta_iso) != n_layers:
        raise ValueError(
            f"Layer count mismatch: CKA has {n_layers} layers "
            f"but isotropy CSV has {len(delta_iso)} rows after aggregation."
        )

    # ── Build dashboard ────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "ΔIsotropy: {CAT-SIGN,CAT-PARITY} − {CTRL-NEU,CTRL-NUM} "
            "(negative → math more isotropic)",
            "Evolutionary CKA: CKA(l, l−1) "
            "(low value → structural reorganisation at layer l)",
        ),
    )

    _line = dict(width=2.5)
    _mark = dict(size=7)

    fig.add_trace(go.Scatter(
        x=layers, y=delta_iso,
        mode="lines+markers", name="ΔIso",
        line=dict(color="#D85A30", **_line), marker=_mark,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=layers, y=cka_math,
        mode="lines+markers", name="CKA math",
        line=dict(color="#D85A30", **_line), marker=_mark,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=layers, y=cka_ctrl,
        mode="lines+markers", name="CKA ctrl",
        line=dict(color="#3B8BD4", **_line), marker=_mark,
    ), row=2, col=1)

    # ΔIso = 0 reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  row=1, col=1, opacity=0.6)

    fig.update_layout(
        title_text  = "RQ1 — Emergence of Mathematical Structure (Pythia-1.4B)",
        height=800, width=1000,
        template    = "plotly_white",
        showlegend  = True,
        hovermode   = "x unified",
    )
    fig.update_xaxes(title_text="Layer index", row=2, col=1,
                     tickmode="linear", dtick=2)
    fig.update_yaxes(title_text="ΔIso(l)",         row=1, col=1)
    fig.update_yaxes(title_text="CKA(l, l−1)",
                     range=[0, 1.05],               row=2, col=1)

    out = OUT_DIR / "rq1_emergence.html"
    fig.write_html(str(out))
    print(f"RQ1 dashboard saved: {out}")


if __name__ == "__main__":
    plot_rq1_dashboard()