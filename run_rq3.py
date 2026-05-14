#!/usr/bin/env python

"""

Entry point per l'Analisi Dinamica (Contesto B).

Valuta classificatori congelati su checkpoint QLoRA e calcola il drift geometrico globale.

"""



import argparse

import yaml

import logging

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import spearmanr

from sklearn.metrics import accuracy_score



from src.probing.seeds import get_seed
import src.probing.io_utils as probing_io



def compute_geometric_drift(H_current: np.ndarray, H_base: np.ndarray) -> float:

    """

    Calcola il drift geometrico normalizzato (Frobenius norm).

    Formula: ||H_ckpt - H_base||_F / (N_eval * d)

    """

    N, d = H_current.shape

    diff = H_current - H_base

    fro_norm = np.linalg.norm(diff, ord="fro")

    return float(fro_norm / (N * d))



def main():

    parser = argparse.ArgumentParser(description="Linear Probing - Context B")

    parser.add_argument("--config", required=True, help="Path to config.yaml")

    parser.add_argument("--checkpoint_dir", required=True, help="Path ai tensori del checkpoint")

    args = parser.parse_args()



    # Setup

    with open(args.config, "r") as f:

        config = yaml.safe_load(f)

       

    output_dir = Path(config["output_dir"])

    logger = probing_io.setup_logging(output_dir)

    ckpt_path = Path(args.checkpoint_dir)

    step_num = int(ckpt_path.stem.split("_")[-1]) if "_" in ckpt_path.stem else 0

   

    logger.info(f"Avvio Valutazione Dinamica | Checkpoint Step: {step_num}")



    model_dir = Path(f"data/processed/{config['model_name']}")

    metadata = probing_io.load_metadata(model_dir / "metadata.json")

    n_layers = metadata["n_layers"]

    d_model = metadata["d_model"]

    n_stimuli = metadata["n_stimuli"]



    # 1. Campionamento GLOBALE per il Drift Geometrico (incluso CTRL)

    eval_size = config.get("eval_subset_size", 200)

    rng_drift = np.random.default_rng(get_seed(config["seed"], "global_drift_sampling"))

    global_eval_indices = rng_drift.choice(n_stimuli, size=min(eval_size, n_stimuli), replace=False)



    # Pre-caricamento delle label dal dataset

    stimuli_data = probing_io.load_stimuli_jsonl(Path("data/stimuli/stimuli.jsonl"))

    id_to_labels = {item["id"]: item["labels"] for item in stimuli_data}

    stimuli_ids = metadata["stimuli_ids"]



    results = []



    # 2. Elaborazione Layer-wise

    for l in range(n_layers):

        # Caricamento tensori (FP16 -> FP32)

        H_base = probing_io.load_hidden_states(model_dir / f"layer_{l:02d}.pt")

        H_ckpt = probing_io.load_hidden_states(ckpt_path / f"layer_{l:02d}.pt")



        # Calcolo Delta Geometrico Globale

        geom_delta = compute_geometric_drift(

            H_ckpt[global_eval_indices],

            H_base[global_eval_indices]

        )



        # 3. Valutazione Proprietà sui pesi denormalizzati

        for prop_name, prop_cfg in config["properties"].items():

            test_idx = probing_io.load_test_indices(output_dir, prop_name)

            X_test_ckpt = H_ckpt[test_idx]



            # Caricamento Pesi Geometrici (Niente Scaler)

            w_path = output_dir / "weights" / f"layer_{l:02d}_{prop_name}.npy"

            b_path = output_dir / "weights" / f"layer_{l:02d}_{prop_name}_bias.npy"

           

            if not w_path.exists():

                continue

               

            w_orig = np.load(w_path)

            b_orig = np.load(b_path)



            # Inferenza Geometrica Diretta

            if w_orig.ndim == 1:

                # Proprietà Binaria: Iperpiano (X * w + b > 0)

                scores = np.dot(X_test_ckpt, w_orig) + b_orig

                y_pred = (scores > 0).astype(int)

            else:

                # Proprietà Multiclasse: Argmax (X * W.T + b)

                scores = np.dot(X_test_ckpt, w_orig.T) + b_orig

                y_pred = np.argmax(scores, axis=1)



            # Ricostruzione True Labels

            y_true = []

            for idx in test_idx:

                sid = stimuli_ids[idx]

                lbl = id_to_labels.get(sid, {}).get(prop_cfg["label_field"], -1)

                if prop_cfg["type"] == "multiclass" and lbl != -1:

                    lbl = prop_cfg["class_names"].index(lbl)

                y_true.append(lbl)

           

            y_true = np.array(y_true)

            acc = float(np.round(accuracy_score(y_true, y_pred), 4))



            results.append({

                "step": step_num,

                "layer": l,

                "property": prop_name,

                "probing_acc": acc,

                "geom_delta": float(np.round(geom_delta, 6))

            })



    # 4. Salvataggio Traiettorie

    df = pd.DataFrame(results)

    dyn_dir = output_dir / "dynamic"

    dyn_dir.mkdir(parents=True, exist_ok=True)

   

    # Se il file esiste, facciamo append (utile per esecuzioni iterative sui checkpoint)

    traj_path = dyn_dir / "trajectories.csv"

    if traj_path.exists():

        old_df = pd.read_csv(traj_path)

        # Rimuoviamo vecchi log per questo step per evitare duplicati in caso di re-run

        old_df = old_df[old_df["step"] != step_num]

        df = pd.concat([old_df, df], ignore_index=True)

       

    probing_io._atomic_write_csv(traj_path, df.to_dict("records"), df.columns.tolist())

    logger.info(f"Traiettorie salvate. Drift geometrico max: {df['geom_delta'].max():.4f}")



if __name__ == "__main__":

    main()

