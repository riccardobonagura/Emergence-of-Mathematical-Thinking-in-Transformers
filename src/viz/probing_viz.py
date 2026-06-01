"""probing_viz.py — static (Matplotlib) and interactive (Plotly) probe figures."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path


def plot_accuracy_curves(
    df: pd.DataFrame,
    model_name: str,
    output_png: Path,
    output_html: Path,
    emergence_layers: dict[str, int] | None = None,
    jump_span: tuple[int, int] | None = None,
    jump_label: str | None = None,
) -> None:
    """Accuracy curves with CI bands for all properties in df.

    Optional overlays (E-G-03 emergence reading): emergence_layers draws a per-property
    vertical line at the first layer crossing the threshold; jump_span shades the
    plateau→jump x-range (the parity L12→L13 step) with jump_label as its annotation.
    """
    properties = df["property"].unique()

    # Static figure (300 dpi for thesis)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for prop in properties:
        sub = df[df["property"] == prop].sort_values("layer")
        ax.plot(sub["layer"], sub["accuracy"], label=prop, linewidth=2)
        ax.fill_between(
            sub["layer"],
            sub["accuracy_lower_ci"],
            sub["accuracy_upper_ci"],
            alpha=0.2,
        )
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.7, label="Random baseline")

    # Shade the parity plateau→jump span first so curves/lines draw on top.
    if jump_span is not None:
        ax.axvspan(jump_span[0], jump_span[1], color="#FBBF24", alpha=0.18,
                   label=jump_label or "parity jump")
    if emergence_layers:
        for prop, lyr in emergence_layers.items():
            if lyr is None:
                continue
            ax.axvline(lyr, linestyle="-.", color="#6B7280", alpha=0.7)
            ax.annotate(f"{prop} emerges L{lyr}", xy=(lyr, 0.52),
                        rotation=90, va="bottom", ha="right", fontsize=8, color="#374151")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Linear Probing Accuracy — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    # Interactive figure (asymmetric error bars from CI)
    fig_html = go.Figure()
    for prop in properties:
        sub = df[df["property"] == prop].sort_values("layer")
        fig_html.add_trace(go.Scatter(
            x=sub["layer"], y=sub["accuracy"],
            mode="lines+markers", name=prop,
            error_y=dict(
                type="data", symmetric=False,
                array      = sub["accuracy_upper_ci"] - sub["accuracy"],
                arrayminus = sub["accuracy"]           - sub["accuracy_lower_ci"],
            ),
        ))
    fig_html.add_hline(y=0.5, line_dash="dash", line_color="gray")
    if jump_span is not None:
        fig_html.add_vrect(x0=jump_span[0], x1=jump_span[1], fillcolor="#FBBF24",
                           opacity=0.18, line_width=0,
                           annotation_text=jump_label or "parity jump",
                           annotation_position="top left")
    if emergence_layers:
        for prop, lyr in emergence_layers.items():
            if lyr is None:
                continue
            fig_html.add_vline(x=lyr, line_dash="dashdot", line_color="#6B7280",
                               annotation_text=f"{prop} emerges L{lyr}",
                               annotation_position="bottom")
    fig_html.update_layout(
        title=f"Linear Probing Accuracy — {model_name}",
        xaxis_title="Layer", yaxis_title="Accuracy",
        template="plotly_white",
    )
    fig_html.write_html(output_html)


def plot_angles_heatmap(
    df: pd.DataFrame,
    layer: int,
    output_png: Path,
    output_html: Path,
) -> None:
    """Cosine-similarity heatmap between probe weight vectors at a given layer."""
    props  = sorted(set(df["property_a"]) | set(df["property_b"]))
    n      = len(props)
    matrix = np.eye(n)

    for _, row in df.iterrows():
        i = props.index(row["property_a"])
        j = props.index(row["property_b"])
        matrix[i, j] = matrix[j, i] = row["cosine_similarity"]

    # Static heatmap
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(props, rotation=45, ha="right")
    ax.set_yticklabels(props)
    for i in range(n):
        for j in range(n):
            color = "white" if abs(matrix[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Semantic Orthogonality — Layer {layer}")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    # Interactive heatmap
    fig_html = px.imshow(
        matrix, x=props, y=props,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f",
        title=f"Semantic Orthogonality — Layer {layer}",
    )
    fig_html.write_html(output_html)