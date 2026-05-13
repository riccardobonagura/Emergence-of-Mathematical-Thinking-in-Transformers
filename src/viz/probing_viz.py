"""Modulo per la generazione delle figure statiche (Matplotlib) e interattive (Plotly)."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def plot_accuracy_curves(df: pd.DataFrame, model_name: str, output_png: Path, output_html: Path):
    """Genera le curve di accuratezza con bande di confidenza per tutte le proprietà."""
    properties = df["property"].unique()
    
    # 1. Matplotlib (Static)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for prop in properties:
        sub = df[df["property"] == prop].sort_values("layer")
        ax.plot(sub["layer"], sub["accuracy"], label=prop, linewidth=2)
        ax.fill_between(
            sub["layer"], 
            sub["accuracy_lower_ci"], 
            sub["accuracy_upper_ci"], 
            alpha=0.2
        )
        
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.7, label="Baseline Casuale")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Linear Probing Accuracy — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    # 2. Plotly (Interactive)
    fig_html = go.Figure()
    for prop in properties:
        sub = df[df["property"] == prop].sort_values("layer")
        fig_html.add_trace(go.Scatter(
            x=sub["layer"], y=sub["accuracy"],
            mode="lines+markers",
            name=prop,
            error_y=dict(
                type="data", symmetric=False,
                array=sub["accuracy_upper_ci"] - sub["accuracy"],
                arrayminus=sub["accuracy"] - sub["accuracy_lower_ci"]
            )
        ))
    fig_html.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_html.update_layout(
        title=f"Linear Probing Accuracy — {model_name}",
        xaxis_title="Layer Index", yaxis_title="Accuracy",
        template="plotly_white"
    )
    fig_html.write_html(output_html)

def plot_angles_heatmap(df: pd.DataFrame, layer: int, output_png: Path, output_html: Path):
    """Genera la heatmap delle similitudini coseno tra le direzioni semantiche."""
    props = sorted(list(set(df["property_a"]).union(set(df["property_b"]))))
    matrix = np.eye(len(props))
    
    for _, row in df.iterrows():
        i, j = props.index(row["property_a"]), props.index(row["property_b"])
        matrix[i, j] = matrix[j, i] = row["cosine_similarity"]

    # 1. Matplotlib
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(props)))
    ax.set_yticks(range(len(props)))
    ax.set_xticklabels(props, rotation=45, ha="right")
    ax.set_yticklabels(props)
    
    for i in range(len(props)):
        for j in range(len(props)):
            color = "white" if abs(matrix[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color)
            
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Ortogonalità Semantica — Layer {layer}")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    # 2. Plotly
    fig_html = px.imshow(
        matrix, x=props, y=props,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f", title=f"Ortogonalità Semantica — Layer {layer}"
    )
    fig_html.write_html(output_html)