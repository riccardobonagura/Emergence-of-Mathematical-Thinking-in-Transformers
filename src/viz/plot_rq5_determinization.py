"""
plot_rq5_determinization.py — RQ5 determinization dashboard.

Three panels vs Training Step, one trace per math category:
  (1) next-token entropy ↓   (2) P(answer | single-token) ↑   (3) top1-top2 margin ↑
all read at the "=" token (E-P-02: the model's expected-result distribution sharpening).

GSM8K is an OPTIONAL descriptive overlay (n=6 checkpoints, no statistical claim) drawn
only if results/rq4_drift/trajectories_probing.csv carries a gsm8k_acc column.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots

CATEGORY_COLORS = {"CAT-SIGN": "#1f77b4", "CAT-PARITY": "#d62728"}
GSM8K_CSV = Path("results/rq4_drift/trajectories_probing.csv")


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("rq5_viz")


def _load_gsm8k() -> pd.DataFrame | None:
    """Per-step GSM8K accuracy, or None if the RQ4 trajectory has no gsm8k column."""
    if not GSM8K_CSV.exists():
        return None
    traj = pd.read_csv(GSM8K_CSV)
    if "gsm8k_acc" not in traj.columns or traj["gsm8k_acc"].isna().all():
        return None
    return traj.groupby("step")["gsm8k_acc"].first().reset_index()


def main() -> None:
    logger = setup_logger()
    csv_path = Path("results/rq5_determinization/determinization.csv")
    out_dir = Path("results/figures/rq5")
    out_file = out_dir / "rq5_determinization.html"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run run_rq5.py to generate the determinization data first."
        )

    df = pd.read_csv(csv_path).sort_values("step")
    out_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_df = _load_gsm8k()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Next-token entropy ↓ (nats)",
            "P(answer | single-token) ↑",
            "Top1−Top2 logit margin ↑",
        ),
        horizontal_spacing=0.07,
    )

    panels = [
        (1, "entropy_mean", None),
        (2, "p_correct_single", ("p_correct_single_ci_lo", "p_correct_single_ci_hi")),
        (3, "margin_mean", None),
    ]

    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat].sort_values("step")
        color = CATEGORY_COLORS.get(cat, "#2ca02c")
        for col, ycol, ci_cols in panels:
            show_legend = (col == 1)  # one legend entry per category
            fig.add_trace(
                plotly_go.Scatter(
                    x=cat_df["step"], y=cat_df[ycol],
                    mode="lines+markers", name=cat,
                    legendgroup=cat, showlegend=show_legend,
                    line=dict(color=color),
                    hovertemplate=f"Step: %{{x}}<br>{ycol}: %{{y:.4f}}<br>{cat}<extra></extra>",
                ),
                row=1, col=col,
            )
            # Symmetric CI band where available (single-token P only).
            if ci_cols is not None:
                lo, hi = ci_cols
                fig.add_trace(
                    plotly_go.Scatter(
                        x=list(cat_df["step"]) + list(cat_df["step"])[::-1],
                        y=list(cat_df[hi]) + list(cat_df[lo])[::-1],
                        fill="toself", mode="lines",
                        line=dict(width=0), fillcolor=color, opacity=0.15,
                        legendgroup=cat, showlegend=False, hoverinfo="skip",
                    ),
                    row=1, col=col,
                )

    # Optional descriptive GSM8K overlay on the P(answer) panel (n=6, no stat claim).
    if gsm8k_df is not None:
        fig.add_trace(
            plotly_go.Scatter(
                x=gsm8k_df["step"], y=gsm8k_df["gsm8k_acc"],
                mode="lines+markers", name="GSM8K 0-shot (descriptive)",
                line=dict(color="black", dash="dash", width=2),
                hovertemplate="Step: %{x}<br>GSM8K: %{y:.3f}<extra></extra>",
            ),
            row=1, col=2,
        )

    for col in (1, 2, 3):
        fig.update_xaxes(title_text="Training Step", row=1, col=col)
    fig.update_yaxes(title_text="Entropy (nats)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=2)
    fig.update_yaxes(title_text="Logit margin", row=1, col=3)

    fig.update_layout(
        title="RQ5 — Behavioral determinization at the '=' token",
        height=520, width=1500, template="plotly_white", hovermode="x unified",
    )

    fig.write_html(str(out_file))
    logger.info(f"RQ5 dashboard generated: {out_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger("rq5_viz").error(f"Execution failed: {e}")
        sys.exit(1)
