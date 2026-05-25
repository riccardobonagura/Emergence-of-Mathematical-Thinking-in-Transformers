"""
plot_rq3_trajectory.py — Dynamic Phase Visualization (RQ3).
Generates an interactive Plotly dashboard to correlate internal geometric drift 
with linear probing accuracy and external benchmark (GSM8K) performance.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("rq3_viz")

def main() -> None:
    logger = setup_logger()
    csv_path = Path("results/rq2_probing/dynamic/trajectories.csv")
    out_dir = Path("results/figures/rq3")
    out_file = out_dir / "rq3_dashboard.html"

    # 1. Invariant: File Existence Check
    if not csv_path.exists():
        logger.error("FATAL: trajectories.csv not found.")
        raise FileNotFoundError(
            f"Missing {csv_path}. You must execute run_rq3.py (and optionally eval_gsm8k.py) "
            "to generate the trajectory data before running the visualization."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading trajectory data from {csv_path}...")
    
    df = pd.read_csv(csv_path)

    # Invariant: Layer count dynamically inferred
    layers = sorted(df['layer'].unique())
    properties = sorted(df['property'].unique())
    steps = sorted(df['step'].unique())
    
    if not steps:
        raise ValueError("The trajectory CSV is empty or malformed.")

    # 2. Data Pre-processing: Calculate Delta Accuracy relative to baseline
    baseline_step = steps[0]
    baseline_df = df[df['step'] == baseline_step][['layer', 'property', 'probing_acc']].copy()
    baseline_df.rename(columns={'probing_acc': 'base_acc'}, inplace=True)
    
    df = df.merge(baseline_df, on=['layer', 'property'], how='left')
    df['delta_acc'] = df['probing_acc'] - df['base_acc']

    # 3. Handle GSM8K overlay logic
    has_gsm8k = 'gsm8k_acc' in df.columns and not df['gsm8k_acc'].isna().all()
    if has_gsm8k:
        gsm8k_df = df.groupby('step')['gsm8k_acc'].first().reset_index()

    # 4. Setup Plotly Dashboard Layout
    # Row 1: Accuracy Curve (Col 1) | Drift Heatmap (Col 2)
    # Row 2: Scatter Drift vs Delta Acc (Col spans both)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Probing Accuracy Trajectory (Layer Selected)", 
            "Geometric Drift Heatmap (All Layers)", 
            "Correlation: Drift vs Δ Probing Accuracy"
        ),
        specs=[
            [{"secondary_y": True}, {"type": "heatmap"}],
            [{"colspan": 2, "type": "scatter"}, None]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    traces_visibility_map = []  # Keeps track of which layer each trace belongs to for the dropdown

    # --- SUBPLOT 1: Accuracy Curves (Parameterized by Layer) ---
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for layer in layers:
        layer_df = df[df['layer'] == layer]
        for idx, prop in enumerate(properties):
            prop_df = layer_df[layer_df['property'] == prop].sort_values('step')
            
            fig.add_trace(
                plotly_go.Scatter(
                    x=prop_df['step'],
                    y=prop_df['probing_acc'],
                    mode='lines+markers',
                    name=f"Acc: {prop}",
                    line=dict(color=colors[idx % len(colors)]),
                    visible=(layer == layers[0]), # Only first layer visible by default
                    hovertemplate="Step: %{x}<br>Acc: %{y:.3f}<br>Prop: " + prop + "<extra></extra>"
                ),
                row=1, col=1, secondary_y=False
            )
            traces_visibility_map.append(layer)

    # --- SUBPLOT 1 Overlay: GSM8K (Global) ---
    if has_gsm8k:
        fig.add_trace(
            plotly_go.Scatter(
                x=gsm8k_df['step'],
                y=gsm8k_df['gsm8k_acc'],
                mode='lines+markers',
                name="GSM8K 0-shot",
                line=dict(color='black', dash='dash', width=2),
                visible=True,
                hovertemplate="Step: %{x}<br>GSM8K: %{y:.3f}<extra></extra>"
            ),
            row=1, col=1, secondary_y=True
        )
        traces_visibility_map.append("GLOBAL")
        fig.update_yaxes(title_text="GSM8K Acc", secondary_y=True, row=1, col=1)

    fig.update_yaxes(title_text="Probing Acc", secondary_y=False, row=1, col=1)
    fig.update_xaxes(title_text="Training Step", row=1, col=1)

    # --- SUBPLOT 2: Drift Heatmap ---
    # Average geom_delta across properties for the heatmap topology
    heatmap_df = df.groupby(['layer', 'step'])['geom_delta'].mean().reset_index()
    heatmap_pivot = heatmap_df.pivot(index='layer', columns='step', values='geom_delta')
    
    fig.add_trace(
        plotly_go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Drift", x=1.00, y=0.75, len=0.45),
            hovertemplate="Step: %{x}<br>Layer: %{y}<br>Drift: %{z:.4f}<extra></extra>"
        ),
        row=1, col=2
    )
    traces_visibility_map.append("GLOBAL")
    fig.update_yaxes(title_text="Transformer Layer", row=1, col=2)
    fig.update_xaxes(title_text="Training Step", row=1, col=2)

    # --- SUBPLOT 3: Scatter Drift ↔ Δ Accuracy ---
    fig.add_trace(
        plotly_go.Scatter(
            x=df['geom_delta'],
            y=df['delta_acc'],
            mode='markers',
            marker=dict(
                color=df['layer'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Layer", x=1.00, y=0.20, len=0.45),
                size=8,
                opacity=0.7
            ),
            text=df['property'] + " (Step " + df['step'].astype(str) + ")",
            hovertemplate="Drift: %{x:.4f}<br>Δ Acc: %{y:.4f}<br>%{text}<br>Layer: %{marker.color}<extra></extra>"
        ),
        row=2, col=1
    )
    traces_visibility_map.append("GLOBAL")
    fig.update_xaxes(title_text="Geometric Drift (Frobenius Norm)", row=2, col=1)
    fig.update_yaxes(title_text="Δ Probing Accuracy (vs Baseline)", row=2, col=1)

    # 5. Build Interactive Dropdown Logic
    dropdown_buttons = []
    for layer in layers:
        visibility = [
            (trace_layer == layer or trace_layer == "GLOBAL") 
            for trace_layer in traces_visibility_map
        ]
        
        dropdown_buttons.append(dict(
            label=f"Layer {layer:02d}",
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"RQ3 Dashboard — Active Trace: Layer {layer:02d}"}
            ]
        ))

    fig.update_layout(
        title=f"RQ3 Dashboard — Active Trace: Layer {layers[0]:02d}",
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            x=0.01,
            xanchor="left",
            y=1.15,
            yanchor="top",
            pad={"r": 10, "t": 10},
            showactive=True,
        )],
        height=900,
        width=1400,
        template="plotly_white",
        hovermode="closest"
    )

    # 6. Save HTML Output
    fig.write_html(str(out_file))
    logger.info(f"Dashboard generated successfully: {out_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger("rq3_viz")
        logger.error(f"Execution failed: {e}")
        sys.exit(1)