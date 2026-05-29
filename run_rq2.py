#!/usr/bin/env python
"""
run_rq2.py — RQ2 orchestrator: static linear probing with strict statistical gating.
Flow: config → data → parallel engine → permutation stats → FDR correction → selectivity & difficulty gradient.

Expected config["properties"] schema (v5):
  properties:
    sign:   {label_field: sign,   category: CAT-SIGN,   type: binary}
    parity: {label_field: parity, category: CAT-PARITY, type: binary}
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from pathlib import Path

from src.config.categories       import CTRL_CATS
from src.probing.io_utils        import (MetadataHandler, setup_logging,
                                         load_hidden_states, save_weights,
                                         _atomic_write_csv,
                                         load_test_indices, save_test_indices)
from src.probing.probing_dataset import ProbingDataset
from src.probing.engine          import ProbingEngine
from src.probing.directions      import compute_selectivity
from src.probing.stats           import benjamini_hochberg_correction
from src.probing.seeds           import get_seed


def process_task(
    layer_idx: int, prop_name: str, model_dir: Path,
    engine: ProbingEngine, train_idx: np.ndarray, test_idx: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray, ctrl_idx: np.ndarray, y_ctrl: np.ndarray,
    magnitudes_test: np.ndarray, gaps_test: np.ndarray, output_dir: Path
) -> dict:
    """Atomic worker: load tensor slice → fit probe → evaluate selectivity, confounds & difficulty gradient."""
    H = load_hidden_states(model_dir / f"layer_{layer_idx:02d}.pt")

    result = engine.run_layer(
        X_train=H[train_idx], y_train=y_train,
        X_test=H[test_idx], y_test=y_test,
        layer_idx=layer_idx, prop_name=prop_name,
        magnitudes_test=magnitudes_test
    )

    w_orig = result["weights"]
    b_orig = result["bias"]

    # ── P0-1: Hewitt & Liang Selectivity Test ──
    if w_orig.ndim == 1:
        y_pred_ctrl = (np.dot(H[ctrl_idx], w_orig) + b_orig > 0).astype(int)
        y_pred_test = (np.dot(H[test_idx], w_orig) + b_orig > 0).astype(int)
    else:
        y_pred_ctrl = np.argmax(np.dot(H[ctrl_idx], w_orig.T) + b_orig, axis=1)
        y_pred_test = np.argmax(np.dot(H[test_idx], w_orig.T) + b_orig, axis=1)

    acc_ctrl = float(np.mean(y_pred_ctrl == y_ctrl))
    result["accuracy_control"] = round(acc_ctrl, 4)
    result["selectivity"] = round(compute_selectivity(result["accuracy"], acc_ctrl), 4)

    # ── D-09: Difficulty Gradient (Gap Stratification) ──
    if len(gaps_test) > 0 and np.var(gaps_test) > 0:
        q25, q75 = np.percentile(gaps_test, [25, 75])
        hard_mask = gaps_test <= q25
        easy_mask = gaps_test >= q75

        if np.sum(hard_mask) > 0 and np.sum(easy_mask) > 0:
            acc_hard = float(np.mean(y_pred_test[hard_mask] == y_test[hard_mask]))
            acc_easy = float(np.mean(y_pred_test[easy_mask] == y_test[easy_mask]))
            result["gap_robustness_delta"] = round(acc_easy - acc_hard, 4)
        else:
            result["gap_robustness_delta"] = 0.0
    else:
        result["gap_robustness_delta"] = 0.0

    save_weights(output_dir, layer_idx, prop_name, w_orig, b_orig)

    del result["weights"]
    del result["bias"]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict RQ2 Probing")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir)

    model_dir = Path("data/processed") / config["model_name"]
    meta = MetadataHandler(model_dir / "metadata.json")
    n_layers = meta.get_n_layers()

    dataset = ProbingDataset(
        stimuli_path=Path("data/processed/dataset_master_v5.jsonl"),
        stimuli_ids=meta.get_stimuli_ids(),
        cfg=config
    )

    # ── Extract Metadata for Confounds and Gradients ──
    all_magnitudes = np.array([
        r.get("labels", {}).get("operand1", 0) for r in dataset._df
    ])
    all_gaps = np.array([
        abs(r.get("labels", {}).get("operand1", 0) - r.get("labels", {}).get("operand2", 0))
        for r in dataset._df
    ])

    all_categories = np.array([r.get("category") for r in dataset._df])
    ctrl_idx_global = np.where(np.isin(all_categories, list(CTRL_CATS)))[0]

    engine = ProbingEngine(config)
    all_results = []

    for prop_name, prop_cfg in config["properties"].items():
        logger.info(f"Extracting splits for {prop_name}...")

        train_idx, test_idx, y_train, y_test = dataset.get_property_split(
            prop_name, prop_cfg, config["train_split"], config["seed"]
        )
        save_test_indices(out_dir, prop_name, test_idx)

        magnitudes_test = all_magnitudes[test_idx]
        gaps_test = all_gaps[test_idx]

        # ── Deterministic property offset to diversify control subsets ──
        prop_offset = hash(prop_name) % 1000
        ctrl_seed = get_seed(config["seed"], "ctrl_sampling", prop_offset)
        rng = np.random.default_rng(ctrl_seed)

        ctrl_idx = rng.choice(ctrl_idx_global, size=min(len(ctrl_idx_global), len(test_idx)), replace=False)
        y_ctrl = rng.permutation(y_test[:len(ctrl_idx)])

        logger.info(f"Parallel probing {prop_name} across {n_layers} layers...")
        tasks = [
            delayed(process_task)(
                l, prop_name, model_dir, engine,
                train_idx, test_idx, y_train, y_test,
                ctrl_idx, y_ctrl, magnitudes_test, gaps_test, out_dir
            ) for l in range(n_layers)
        ]

        prop_results = Parallel(n_jobs=config.get("n_jobs", -1))(tasks)
        all_results.extend(prop_results)

    logger.info("Applying Benjamini-Hochberg FDR correction...")
    p_values = [res["raw_p_value"] for res in all_results]
    fdr_results = benjamini_hochberg_correction(p_values, fdr_level=0.05)

    for res, is_sig in zip(all_results, fdr_results):
        res["is_significant"] = is_sig

        if res.get("gap_robustness_delta", 0.0) > 0.2:
            logger.warning(
                f"[!] D-09 Confound: {res['property']} at layer {res['layer']} relies heavily on operand gap magnitude. "
                f"Accuracy drops by {res['gap_robustness_delta']*100:.1f}% on hard stimuli."
            )

    acc_df = pd.DataFrame(all_results)
    _atomic_write_csv(
        out_dir / "accuracy_metrics_corrected.csv",
        acc_df.to_dict("records"),
        acc_df.columns.tolist()
    )

    logger.info("RQ2 Strict Evaluation Complete.")


if __name__ == "__main__":
    main()
