"""
Fase 6 - Visualizzazione RQ1
Genera una dashboard HTML interattiva con Plotly per individuare il layer di emergenza l*.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def plot_rq1_dashboard():
    # Setup percorsi
    RESULTS_DIR = Path("results/rq1_emergence")
    OUT_DIR = Path("results/figures/rq1_emergence")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Caricamento Dati
    iso_file = RESULTS_DIR / "isotropy_pythia.csv"
    cka_math_file = RESULTS_DIR / "cka_math_evol.npy"
    cka_ctrl_file = RESULTS_DIR / "cka_ctrl_evol.npy"
    
    if not iso_file.exists() or not cka_math_file.exists() or not cka_ctrl_file.exists():
        raise FileNotFoundError("Dati RQ1 mancanti. Esegui prima src/run_rq1.py")
        
    df_iso = pd.read_csv(iso_file)
    cka_math = np.load(cka_math_file)
    cka_ctrl = np.load(cka_ctrl_file)
    
    # FIX: cka_inter non esiste più, calcoliamo i layer su cka_math
    n_layers = len(cka_math) 
    layers = np.arange(n_layers)
    
    # 2. Estrazione e calcolo Delta Isotropia
    # Formula: DeltaIso(l) = Iso(l, math) - Iso(l, generic)
    iso_math = df_iso[df_iso["category"] == "CAT-ARITH"].sort_values("layer")["iso_mean"].values
    iso_ctrl = df_iso[df_iso["category"] == "CAT-CTRL"].sort_values("layer")["iso_mean"].values
    
    delta_iso = iso_math - iso_ctrl
    
    # 3. Costruzione Dashboard Plotly
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Δ Isotropia: CAT-ARITH vs CAT-CTRL (Maggiore = matematica più isotropa)", 
            "CKA Evolutiva (Minore = riorganizzazione geometrica più marcata dal layer precedente)" # FIX: Titolo aggiornato
        )
    )
    
    # Traccia 1: Delta Isotropia
    fig.add_trace(
        go.Scatter(
            x=layers, y=delta_iso, 
            mode='lines+markers', 
            name='Δ Isotropia', 
            line=dict(color='#D85A30', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Traccia 2: CKA Evolutiva (Math)
    fig.add_trace(
        go.Scatter(
            x=layers, y=cka_math, 
            mode='lines+markers', 
            name='CKA Evolutiva (Math)', 
            line=dict(color='#D85A30', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Traccia 3: CKA Evolutiva (Ctrl)
    fig.add_trace(
        go.Scatter(
            x=layers, y=cka_ctrl, 
            mode='lines+markers', 
            name='CKA Evolutiva (Ctrl)', 
            line=dict(color='#3B8BD4', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Linea di baseline per il Delta Isotropia (y=0)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1, opacity=0.7)
    
    # 4. Formattazione e Layout
    fig.update_layout(
        title_text="RQ1: Emergenza del Ragionamento Matematico (Pythia-1.4B)",
        height=800,
        width=1000,
        template="plotly_white",
        showlegend=True, # FIX: Impostato a True per distinguere Math e Ctrl
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text="Indice del Layer (l)", row=2, col=1, tickmode='linear', dtick=2)
    fig.update_yaxes(title_text="Δ Iso(l)", row=1, col=1)
    fig.update_yaxes(title_text="CKA(l, l-1)", range=[0, 1.05], row=2, col=1) # FIX: y-label più preciso
    
    # 5. Salvataggio
    out_html = OUT_DIR / "rq1_emergence.html"
    fig.write_html(str(out_html))
    print(f"Dashboard RQ1 salvata in: {out_html}")

if __name__ == "__main__":
    plot_rq1_dashboard()