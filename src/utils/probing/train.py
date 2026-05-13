#!/usr/bin/env python
"""
Entry point per l'Analisi Statica (Contesto A) del Linear Probing.
Orchestra l'addestramento, l'estrazione geometrica e i test statistici layer-wise.
"""

import os
import argparse
import yaml
import tempfile
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Import dei moduli interni della pipeline
from probing_dataset import prepare_property_indices
from pipeline import build_pipeline, denormalize_classifier
from metrics import bootstrap_ci, permutation_test_parallel
from directions import cosine_similarity, angle_degrees
from seeds import get_seed
import io as probing_io

def process_layer_property(
    layer_idx: int, 
    prop_name: str, 
    model_dir: Path, 
    config: dict, 
    train_indices: np.ndarray, 
    test_indices: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> dict:
    """
    Worker joblib per elaborare la classificazione di una singola coppia (layer, property).
    """
    logger = logging.getLogger("probing")
    
    # 1. Caricamento tensori (Casting FP16 -> FP32 garantito da probing_io)
    layer_path = model_dir / f"layer_{layer_idx:02d}.pt"
    H = probing_io.load_hidden_states(layer_path)
    
    X_train = H[train_indices]
    X_test = H[test_indices]
    
    # 2. Costruzione e addestramento pipeline
    pipe = build_pipeline(
        max_iter=config["max_iter"],
        C=config["C"],
        solver=config["solver"],
        multiclass_strategy=config["multiclass_strategy"]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = pipe.score(X_test, y_test)
    
    # 3. Denormalizzazione e salvataggio pesi nello spazio geometrico originale
    w_orig, b_orig = denormalize_classifier(pipe)
    output_dir = Path(config["output_dir"])
    probing_io.save_weights(output_dir, layer_idx, prop_name, w_orig, b_orig)
    
    # 4. Calcolo Bootstrap Confidence Interval
    ci_seed = get_seed(config["seed"], "bootstrap", layer_idx * 1000 + hash(prop_name) % 1000)
    lower_ci, upper_ci = bootstrap_ci(
        y_test, y_pred, 
        n_samples=config["bootstrap_n_samples"], 
        ci=config["bootstrap_ci"], 
        base_seed=ci_seed
    )
    
    # 5. Permutation Test Parallelizzato
    perm_seed = get_seed(config["seed"], "permutation", layer_idx * 1000 + hash(prop_name) % 1000)
    baseline_mean, p_value = permutation_test_parallel(
        pipe.named_steps["clf"], 
        X_train, y_train, 
        X_test, y_test, 
        actual_accuracy=accuracy, 
        n_permutations=config["n_permutation_tests"], 
        base_seed=perm_seed,
        n_jobs=1  # Lasciamo a 1 qui se parallelizziamo già sul loop esterno, per evitare oversubscription
    )
    
    logger.debug(f"Layer {layer_idx:02d} | Prop: {prop_name} | Acc: {accuracy:.4f} | p-val: {p_value:.4f}")
    
    return {
        "layer": layer_idx,
        "property": prop_name,
        "accuracy": float(np.round(accuracy, 4)),
        "accuracy_lower_ci": float(np.round(lower_ci, 4)),
        "accuracy_upper_ci": float(np.round(upper_ci, 4)),
        "permutation_baseline": float(np.round(baseline_mean, 4)),
        "permutation_p_value": float(np.round(p_value, 4)),
    }

def main():
    parser = argparse.ArgumentParser(description="Linear Probing - Context A")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # Setup iniziale
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    output_dir = Path(config["output_dir"])
    logger = probing_io.setup_logging(output_dir)
    logger.info(f"Avvio Linear Probing (Contesto A) | Modello: {config['model_name']}")

    # Caricamento dinamico dei layer per supporto a modelli arbitrari (es. GPT-2 = 24, Phi-3 = 32)
    model_dir = Path(f"data/processed/{config['model_name']}")
    metadata = probing_io.load_metadata(model_dir / "metadata.json")
    n_layers = metadata["n_layers"]
    logger.info(f"Rilevati {n_layers} layer dai metadati.")

    # Partizionamento del dataset: avviene UNA SOLA VOLTA per proprietà
    split_cache = {}
    for prop_name, prop_cfg in config["properties"].items():
        tr_idx, te_idx, y_tr, y_te = prepare_property_indices(
            Path("data/stimuli/stimuli.jsonl"), 
            model_dir / "metadata.json",
            prop_name, 
            prop_cfg, 
            config["train_split"], 
            config["seed"]
        )
        split_cache[prop_name] = (tr_idx, te_idx, y_tr, y_te)
        
        # Fissiamo il test_idx su disco per il Contesto B (Zero Data Leakage)
        probing_io.save_test_indices(output_dir, prop_name, te_idx)
        logger.info(f"Split preparato e salvato per la proprietà '{prop_name}'.")

    # Creazione dei task per l'elaborazione parallela
    tasks = []
    for layer_idx in range(n_layers):
        for prop_name in config["properties"].keys():
            tr_idx, te_idx, y_tr, y_te = split_cache[prop_name]
            tasks.append((layer_idx, prop_name, model_dir, config, tr_idx, te_idx, y_tr, y_te))

    # Esecuzione parallela
    logger.info(f"Avvio processing parallelo con {config['n_jobs']} core...")
    results = Parallel(n_jobs=config["n_jobs"], verbose=10)(
        delayed(process_layer_property)(*task) for task in tasks
    )

    # Scrittura atomica della matrice di accuratezza
    acc_df = pd.DataFrame(results)
    out_csv = output_dir / "accuracy_matrix.csv"
    fd, temp_path = tempfile.mkstemp(dir=output_dir, suffix=".csv")
    with os.fdopen(fd, 'w') as tf:
        acc_df.to_csv(tf, index=False)
    os.replace(temp_path, out_csv)
    logger.info(f"Matrice delle accuratezze salvata atomicamente in {out_csv}")

    # Estrazione dei Layer di Emergenza
    emergence_threshold = 0.75
    emergence_data = {"model_name": config["model_name"], "seed": config["seed"], "emergence": {}}
    
    for prop_name in config["properties"].keys():
        sub_df = acc_df[acc_df["property"] == prop_name].sort_values("layer")
        if sub_df.empty:
            continue
            
        peak_row = sub_df.loc[sub_df["accuracy"].idxmax()]
        
        # Primo layer che supera la soglia con confidenza statistica
        valid_emergence = sub_df[
            (sub_df["accuracy"] > emergence_threshold) & 
            (sub_df["permutation_p_value"] < 0.05)
        ]
        
        emergence_layer = int(valid_emergence.iloc[0]["layer"]) if not valid_emergence.empty else None
        
        emergence_data["emergence"][prop_name] = {
            "peak_layer": int(peak_row["layer"]),
            "peak_accuracy": float(peak_row["accuracy"]),
            "emergence_layer": emergence_layer,
            "emergence_threshold": emergence_threshold
        }
        
    probing_io._atomic_write_json(output_dir / "emergence_layers.json", emergence_data)
    logger.info("Layer di emergenza estratti e salvati.")

    # Calcolo Angoli tra Direzioni Semantiche (Solo per feature binarie)
    binary_props = [p for p, cfg in config["properties"].items() if cfg["type"] == "binary"]
    angle_rows = []
    
    for layer_idx in range(n_layers):
        for i, pa in enumerate(binary_props):
            for pb in binary_props[i+1:]:
                wa_path = output_dir / "weights" / f"layer_{layer_idx:02d}_{pa}.npy"
                wb_path = output_dir / "weights" / f"layer_{layer_idx:02d}_{pb}.npy"
                
                if wa_path.exists() and wb_path.exists():
                    wa = np.load(wa_path)
                    wb = np.load(wb_path)
                    cos_sim = cosine_similarity(wa, wb)
                    angle = angle_degrees(cos_sim)
                    angle_rows.append({
                        "layer": layer_idx,
                        "property_a": pa,
                        "property_b": pb,
                        "cosine_similarity": float(np.round(cos_sim, 4)),
                        "angle_degrees": float(np.round(angle, 2))
                    })
                    
    if angle_rows:
        angles_df = pd.DataFrame(angle_rows)
        angles_csv = output_dir / "direction_angles.csv"
        fd, temp_path = tempfile.mkstemp(dir=output_dir, suffix=".csv")
        with os.fdopen(fd, 'w') as tf:
            angles_df.to_csv(tf, index=False)
        os.replace(temp_path, angles_csv)
        logger.info(f"Relazioni angolari calcolate e salvate in {angles_csv}")

    logger.info("Pipeline Fase 3 (Contesto A) conclusa con successo.")

if __name__ == "__main__":
    main()