"""
plot_rq1_emergence.py — RQ1 interactive dashboard (Plotly HTML).

Reads:  results/rq1_emergence/cka_results_annotated.csv
            (cka_inter_mean + cka_inter_ci_low/high, matched baselines, cka_evo_math/ctrl)
        results/rq1_emergence/isotropy_aggregated_balanced.csv (preferred: delta_iso + CIs)
        results/rq1_emergence/isotropy_pythia.csv (fallback for ΔIso, no ribbon)

Outputs: results/figures/rq1_emergence/rq1_emergence.html

Panels (top → bottom):
    1. PRIMARY — inter-category CKA(math, ctrl) per layer with bootstrap CI ribbon,
       overlaid with the matched-terminal baselines (CTRL-NEU↔CTRL-NUM, within-math
       across-template). The authority's primary RQ1 metric (Guida §3, RQ1).
    2. ΔIso(l) = ISO(math) − ISO(ctrl); negative → math more isotropic. CI ribbon +
       optional random-Gaussian floor reference line when present.
    3. Evolutionary CKA(l, l−1) (secondary): low → structural reorganisation at layer l.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.config.categories import MATH_CATS, CTRL_CATS


def _ci_ribbon(fig, x, lo, hi, row, col, fillcolor, name) -> None:
    """Add a shaded CI band (upper trace then lower trace with fill='tonexty')."""
    fig.add_trace(go.Scatter(
        x=x, y=hi, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor=fillcolor, name=name, hoverinfo="skip",
    ), row=row, col=col)


def plot_rq1_dashboard() -> None:
    RESULTS_DIR = Path("results/rq1_emergence")
    OUT_DIR     = Path("results/figures/rq1_emergence")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cka_file        = RESULTS_DIR / "cka_results_annotated.csv"
    iso_file        = RESULTS_DIR / "isotropy_pythia.csv"
    iso_balanced    = RESULTS_DIR / "isotropy_aggregated_balanced.csv"

    if not cka_file.exists():
        raise FileNotFoundError("RQ1 data missing — run run_rq1.py first.")

    df_cka = pd.read_csv(cka_file).sort_values("layer")
    layers = df_cka["layer"].to_numpy()

    # ── ΔIso source: prefer the balanced aggregated table (carries per-category CIs) ──
    delta_iso = iso_lo = iso_hi = None
    iso_floor = None
    if iso_balanced.exists():
        df_isob = pd.read_csv(iso_balanced).sort_values("layer")
        delta_iso = df_isob["delta_iso"].to_numpy()
        ci_cols = {"ci_low_math", "ci_high_math", "ci_low_ctrl", "ci_high_ctrl"}
        if ci_cols.issubset(df_isob.columns):
            # Conservative CI for the difference: low = lo_math − hi_ctrl, high = hi_math − lo_ctrl.
            iso_lo = (df_isob["ci_low_math"] - df_isob["ci_high_ctrl"]).to_numpy()
            iso_hi = (df_isob["ci_high_math"] - df_isob["ci_low_ctrl"]).to_numpy()
        if "iso_floor" in df_isob.columns:
            iso_floor = float(df_isob["iso_floor"].mean())
    elif iso_file.exists():
        df_iso = pd.read_csv(iso_file)

        def mean_iso(cats: tuple) -> np.ndarray:
            return (
                df_iso[df_iso["category"].isin(cats)]
                .groupby("layer")["iso_mean"].mean().sort_index().values
            )

        delta_iso = mean_iso(MATH_CATS) - mean_iso(CTRL_CATS)

    # ── Build 3-row dashboard ────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Inter-category CKA(math, ctrl) — PRIMARY "
            "(low → categories diverge; compare to matched baselines)",
            "ΔIsotropy: {CAT-SIGN,CAT-PARITY} − {CTRL-NEU,CTRL-NUM} "
            "(negative → math more isotropic)",
            "Evolutionary CKA: CKA(l, l−1) (low → structural reorganisation at layer l)",
        ),
    )

    _line = dict(width=2.5)
    _mark = dict(size=7)

    # ── Panel 1 — inter-category CKA (primary) ──
    if {"cka_inter_ci_low", "cka_inter_ci_high"}.issubset(df_cka.columns):
        _ci_ribbon(
            fig, layers,
            df_cka["cka_inter_ci_low"].to_numpy(), df_cka["cka_inter_ci_high"].to_numpy(),
            row=1, col=1, fillcolor="rgba(216,90,48,0.20)", name="inter-CKA 95% CI",
        )
    fig.add_trace(go.Scatter(
        x=layers, y=df_cka["cka_inter_mean"],
        mode="lines+markers", name="CKA(math, ctrl)",
        line=dict(color="#D85A30", **_line), marker=_mark,
    ), row=1, col=1)

    if "cka_ctrl_neu_vs_num" in df_cka.columns and df_cka["cka_ctrl_neu_vs_num"].notna().any():
        fig.add_trace(go.Scatter(
            x=layers, y=df_cka["cka_ctrl_neu_vs_num"],
            mode="lines+markers", name="baseline CTRL-NEU↔CTRL-NUM",
            line=dict(color="#3B8BD4", dash="dot", width=2), marker=dict(size=5),
        ), row=1, col=1)
    if "cka_math_template_baseline" in df_cka.columns and df_cka["cka_math_template_baseline"].notna().any():
        fig.add_trace(go.Scatter(
            x=layers, y=df_cka["cka_math_template_baseline"],
            mode="lines+markers", name="baseline within-math across-template",
            line=dict(color="#6B7280", dash="dash", width=2), marker=dict(size=5),
        ), row=1, col=1)

    # ── Panel 2 — ΔIso ──
    if delta_iso is not None:
        if iso_lo is not None and iso_hi is not None:
            _ci_ribbon(fig, layers, iso_lo, iso_hi, row=2, col=1,
                       fillcolor="rgba(216,90,48,0.18)", name="ΔIso 95% CI")
        fig.add_trace(go.Scatter(
            x=layers, y=delta_iso,
            mode="lines+markers", name="ΔIso",
            line=dict(color="#D85A30", **_line), marker=_mark,
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1, opacity=0.6)
        if iso_floor is not None:
            fig.add_hline(y=iso_floor, line_dash="dot", line_color="#10B981",
                          row=2, col=1, opacity=0.7,
                          annotation_text="random-Gaussian floor")

    # ── Panel 3 — evolutionary CKA (secondary) ──
    fig.add_trace(go.Scatter(
        x=layers, y=df_cka["cka_evo_math"],
        mode="lines+markers", name="CKA evo math",
        line=dict(color="#D85A30", **_line), marker=_mark,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=layers, y=df_cka["cka_evo_ctrl"],
        mode="lines+markers", name="CKA evo ctrl",
        line=dict(color="#3B8BD4", **_line), marker=_mark,
    ), row=3, col=1)

    fig.update_layout(
        title_text  = "RQ1 — Emergence of Mathematical Structure (Pythia-1.4B)",
        height=1050, width=1000,
        template    = "plotly_white",
        showlegend  = True,
        hovermode   = "x unified",
    )
    fig.update_xaxes(title_text="Layer index", row=3, col=1, tickmode="linear", dtick=2)
    fig.update_yaxes(title_text="Inter-category CKA", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="ΔIso(l)",            row=2, col=1)
    fig.update_yaxes(title_text="CKA(l, l−1)", range=[0, 1.05], row=3, col=1)

    out = OUT_DIR / "rq1_emergence.html"
    fig.write_html(str(out))
    print(f"RQ1 dashboard saved: {out}")


if __name__ == "__main__":
    plot_rq1_dashboard()
