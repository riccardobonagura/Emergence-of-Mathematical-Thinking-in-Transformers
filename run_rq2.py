#!/usr/bin/env python
"""
Orchestratore SOLID per la Risoluzione della RQ2 (Static Probing).
Gestisce il flusso di lavoro: Config -> Data -> Engine -> Visualization.
"""

import argparse
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

# Import dei componenti dell'architettura SOLID
from src.probing.io_utils import MetadataHandler, setup_logging, load_hidden_states, save_weights, _atomic_write_csv, _atomic_write_json
from src.probing.probing_dataset import ProbingDataset
from src.probing.engine import ProbingEngine
from src.probing.directions import cosine_similarity, angle_degrees
import src.viz.probing_viz as viz

def process_task(layer_idx, prop_name, model_dir, engine, train_idx, test_idx, y_train, y_test, output_dir):
    """Worker atomico per la parallelizzazione."""
    # Caricamento tensore (Casting FP16 -> FP32 gestito in IO)
    layer_path = model_dir / f"layer_{layer_idx:02d}.pt"
    H = load_hidden_states(layer_path)
    
    # Esecuzione Probing via Engine
    X_train, X_test = H[train_idx], H[test_idx]
    result = engine.run_layer(X_train, y_train, X_test, y_test, layer_idx, prop_name)
    
    # Estrazione e salvataggio pesi (denormalizzati via Engine)
    w_orig = result.pop("weights")
    b_orig = result.pop("bias")
    save_weights(output_dir, layer_idx, prop_name, w_orig, b_orig)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="SOLID Probing Orchestrator")
    parser.add_argument("--config", type=str, required=True, default="configs/config.yaml", help="Path al file config.yaml")
    args = parser.parse_args()

    # 1. SETUP AMBIENTE
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config["output_dir"])
    figures_dir = Path(config["figures_dir"])
    logger = setup_logging(output_dir)
    logger.info(f"--- Avvio Produzione RQ2: {config['model_name']} ---")

    model_dir = Path("data/processed") / config["model_name"]
    
    # 2. DATA LAYER (SOLID)
    # MetadataHandler risolve le incongruenze dei nomi (n_layers vs layers)
    meta = MetadataHandler(model_dir / "metadata.json")
    n_layers = meta.get_n_layers()
    stimuli_ids = meta.get_stimuli_ids()
    
    # ProbingDataset gestisce allineamento ID, undersampling e split
    dataset = ProbingDataset(Path("data/raw/stimuli_arithmetic_v2.jsonl"), stimuli_ids)
    
    # 3. ENGINE LAYER (SOLID)
    # ProbingEngine incapsula la logica di training e validazione statistica
    engine = ProbingEngine(config)

    # 4. ESECUZIONE PARALLELA
    tasks = []
    logger.info(f"Preparazione task per {n_layers} layer...")
    
    for prop_name, prop_cfg in config["properties"].items():
        tr_idx, te_idx, y_tr, y_te = dataset.get_property_split(
            prop_name, prop_cfg, config["train_split"], config["seed"]
        )
        
        for l_idx in range(n_layers):
            tasks.append((l_idx, prop_name, model_dir, engine, tr_idx, te_idx, y_tr, y_te, output_dir))

    logger.info(f"Lancio di {len(tasks)} job su {config['n_jobs']} core...")
    results = Parallel(n_jobs=config["n_jobs"])(delayed(process_task)(*t) for t in tasks)

    # 5. POST-PROCESSING & ANALISI RISULTATI
    acc_df = pd.DataFrame(results)
    _atomic_write_csv(output_dir / "accuracy_matrix.csv", acc_df.to_dict("records"), acc_df.columns.tolist())

    # Calcolo Emergenza Semantica
    emergence_data = {"model": config["model_name"], "layers": {}}
    for prop in config["properties"]:
        prop_df = acc_df[acc_df["property"] == prop].sort_values("layer")
        peak = prop_df.loc[prop_df["accuracy"].idxmax()]
        # Un layer "emerge" se supera 0.7 e la baseline statistica
        em_layer = prop_df[prop_df["accuracy"] > 0.7].head(1)
        
        emergence_data["layers"][prop] = {
            "peak_layer": int(peak["layer"]),
            "peak_acc": float(peak["accuracy"]),
            "emergence_layer": int(em_layer["layer"].values[0]) if not em_layer.empty else None
        }
    _atomic_write_json(output_dir / "emergence_summary.json", emergence_data)

  # Analisi Ortogonalità (Similarità Coseno tra pesi)
    logger.info("Analisi delle direzioni semantiche...")
    angle_results = []
    binary_props = [p for p, c in config["properties"].items() if c["type"] == "binary"]
    
    # Estrarre d_model direttamente da un peso qualsiasi (assumiamo che il primo layer/prop sia valido)
    # per avere un riferimento sulla dimensionalità corretta dello spazio latente.
    d_model = meta.get_d_model(default=2048) # Usiamo il d_model dai metadati

    for l in range(n_layers):
        for i, p_a in enumerate(binary_props):
            for p_b in binary_props[i+1:]:
                # Caricamento pesi salvati dai worker
                wa_path = output_dir / "weights" / f"layer_{l:02d}_{p_a}.npy"
                wb_path = output_dir / "weights" / f"layer_{l:02d}_{p_b}.npy"
                
                if wa_path.exists() and wb_path.exists():
                    w_a = np.load(wa_path)
                    w_b = np.load(wb_path)
                    
                    # --- FIX DIMENSIONALE ---
                    # Scartiamo il calcolo se uno dei classificatori è inaspettatamente diventato multiclasse (size > d_model)
                    if w_a.size != d_model or w_b.size != d_model:
                         logger.debug(f"Skip ortogonalità {p_a} vs {p_b} al layer {l}: Shape mismatch (w_a:{w_a.size}, w_b:{w_b.size})")
                         continue
                         
                    cos_sim = cosine_similarity(w_a, w_b)
                    angle_results.append({
                        "layer": l, "property_a": p_a, "property_b": p_b,
                        "cosine_similarity": cos_sim, "angle_degrees": angle_degrees(cos_sim)
                    })
    
    if angle_results:
        angles_df = pd.DataFrame(angle_results)
        _atomic_write_csv(output_dir / "direction_angles.csv", angles_df.to_dict("records"), angles_df.columns.tolist())

    # 6. VISUALIZZAZIONE
    logger.info("Generazione figure finali...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    viz.plot_accuracy_curves(acc_df, config["model_name"], figures_dir / "accuracy_curves.png", figures_dir / "accuracy_curves.html")
    
    if angle_results:
        # Heatmap del layer di picco della prima proprietà
        peak_l = emergence_data["layers"][binary_props[0]]["peak_layer"]
        viz.plot_angles_heatmap(angles_df[angles_df["layer"] == peak_l], peak_l, 
                                figures_dir / f"orthogonality_l{peak_l}.png", figures_dir / f"orthogonality_l{peak_l}.html")

    logger.info("Fase 3 Completata con Successo.")

if __name__ == "__main__":
    main()