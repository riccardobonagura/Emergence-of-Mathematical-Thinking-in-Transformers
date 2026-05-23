#!/usr/bin/env python
"""
run_rq2.py — RQ2 orchestrator: static linear probing for sign and parity.
Flow: config → data → parallel engine → weights → direction analysis → figures.

Expected config["properties"] schema (v5):
  properties:
    sign:   {label_field: sign,   category: CAT-SIGN}
    parity: {label_field: parity, category: CAT-PARITY}
"""

import argparse
import logging
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from pathlib import Path


from src.probing.io_utils      import (MetadataHandler, setup_logging,
                                        load_hidden_states, save_weights,
                                        _atomic_write_csv, _atomic_write_json,
                                        load_test_indices, save_test_indices)
from src.probing.probing_dataset import ProbingDataset
from src.probing.engine          import ProbingEngine
from src.probing.directions      import cosine_similarity, angle_degrees
import src.viz.probing_viz       as viz


def process_task(
    layer_idx, prop_name, model_dir,
    engine, train_idx, test_idx, y_train, y_test, output_dir,
):
    """Atomic worker: load tensor slice → fit probe → persist weights → return metrics."""
    H       = load_hidden_states(model_dir / f"layer_{layer_idx:02d}.pt")
    result  = engine.run_layer(
        H[train_idx], y_train, H[test_idx], y_test, layer_idx, prop_name
    )
    save_weights(output_dir, layer_idx, prop_name,
                 result.pop("weights"), result.pop("bias"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 static probing")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir  = Path(config["output_dir"])
    figures_dir = Path(config["figures_dir"])
    logger      = setup_logging(output_dir)
    logger.info("--- RQ2 static probing: %s ---", config["model_name"])

    model_dir = Path("data/processed") / config["model_name"]

    # MetadataHandler resolves n_layers robustly (key or .pt file count)
    meta       = MetadataHandler(model_dir / "metadata.json")
    n_layers   = meta.get_n_layers()
    stimuli_ids = meta.get_stimuli_ids()

    dataset = ProbingDataset(
        Path("data/processed/dataset_master_v5.jsonl"), stimuli_ids
    )
    engine = ProbingEngine(config)

    # Build (layer, property) task list; splits are computed once per property
    tasks = []
    for prop_name, prop_cfg in config["properties"].items():
        tr_idx, te_idx, y_tr, y_te = dataset.get_property_split(
            prop_name, prop_cfg, config["train_split"], config["seed"]
        )
        # Persist test indices now so RQ3 can reload them without re-splitting
        save_test_indices(output_dir, prop_name, te_idx)

        for l in range(n_layers):
            tasks.append((l, prop_name, model_dir, engine,
                          tr_idx, te_idx, y_tr, y_te, output_dir))

    logger.info("Dispatching %d tasks on %d workers...", len(tasks), config["n_jobs"])
    results = Parallel(n_jobs=config["n_jobs"])(
        delayed(process_task)(*t) for t in tasks
    )

    # Accuracy matrix
    acc_df = pd.DataFrame(results)
    _atomic_write_csv(
        output_dir / "accuracy_matrix.csv",
        acc_df.to_dict("records"), acc_df.columns.tolist(),
    )

    # Emergence summary: first layer exceeding 0.7, and peak layer
    emergence = {"model": config["model_name"], "layers": {}}
    for prop in config["properties"]:
        sub = acc_df[acc_df["property"] == prop].sort_values("layer")
        peak     = sub.loc[sub["accuracy"].idxmax()]
        em_rows  = sub[sub["accuracy"] > 0.7]
        emergence["layers"][prop] = {
            "peak_layer":      int(peak["layer"]),
            "peak_acc":        float(peak["accuracy"]),
            "emergence_layer": int(em_rows["layer"].iloc[0]) if not em_rows.empty else None,
        }
    _atomic_write_json(output_dir / "emergence_summary.json", emergence)

    # Direction analysis: cosine similarity between probe weight vectors
    # All v5 probes are binary; fall back to "binary" when "type" key is absent
    logger.info("Computing direction analysis...")
    d_model     = meta.get_d_model(default=2048)
    binary_props = [
        p for p, c in config["properties"].items()
        if c.get("type", "binary") == "binary"
    ]

    angle_results = []
    for l in range(n_layers):
        for i, p_a in enumerate(binary_props):
            for p_b in binary_props[i + 1:]:
                wa_path = output_dir / "weights" / f"layer_{l:02d}_{p_a}.npy"
                wb_path = output_dir / "weights" / f"layer_{l:02d}_{p_b}.npy"
                if not (wa_path.exists() and wb_path.exists()):
                    continue
                w_a, w_b = np.load(wa_path), np.load(wb_path)
                # Guard against unexpected multiclass shapes
                if w_a.size != d_model or w_b.size != d_model:
                    logger.debug(
                        "Skip %s vs %s layer %d: size mismatch (%d, %d)",
                        p_a, p_b, l, w_a.size, w_b.size,
                    )
                    continue
                cos = cosine_similarity(w_a, w_b)
                angle_results.append({
                    "layer": l, "property_a": p_a, "property_b": p_b,
                    "cosine_similarity": cos, "angle_degrees": angle_degrees(cos),
                })

    if angle_results:
        angles_df = pd.DataFrame(angle_results)
        _atomic_write_csv(
            output_dir / "direction_angles.csv",
            angles_df.to_dict("records"), angles_df.columns.tolist(),
        )

    # Figures
    logger.info("Generating figures...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    viz.plot_accuracy_curves(
        acc_df, config["model_name"],
        figures_dir / "accuracy_curves.png",
        figures_dir / "accuracy_curves.html",
    )
    if angle_results and binary_props:
        peak_l = emergence["layers"][binary_props[0]]["peak_layer"]
        layer_angles = angles_df[angles_df["layer"] == peak_l]
        viz.plot_angles_heatmap(
            layer_angles, peak_l,
            figures_dir / f"orthogonality_l{peak_l}.png",
            figures_dir / f"orthogonality_l{peak_l}.html",
        )

    logger.info("RQ2 complete.")


if __name__ == "__main__":
    main()