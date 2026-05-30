"""
run_confound_checks.py — Hardened Confound Mitigation & Epistemological Validation Module.
Executes deep statistical checks on the N-01 Confound (First Operand Leakage in CAT-SIGN).

Verifies if the "sign" probe trained in RQ2 genuinely decodes the abstract mathematical
property or collapses into a surface token length/magnitude proxy of operand1.
Enforces Benjamini-Hochberg FDR correction over control tests and dumps atomic metrics.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # pin BLAS to 1 thread/worker

import argparse
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LinearRegression

from src.probing.directions import cosine_similarity
from src.probing.seeds import get_seed
from src.probing.io_utils import _atomic_write_csv
from src.probing.stats import benjamini_hochberg_correction


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("confound_check")


def extract_operand1(stimulus: dict) -> float:
    """Extracts the exact scalar value of the first operand to check proxy memorization."""
    return float(stimulus["labels"]["operand1"])


def extract_magnitude_delta(stimulus: dict) -> float:
    """Extracts the mathematical magnitude delta |a - b| of the expression."""
    return float(abs(int(stimulus["labels"]["operand1"]) - int(stimulus["labels"]["operand2"])))


def main() -> None:
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Epistemological Confound Verification Suite")
    parser.add_argument("--config", type=str, required=True, help="Path to the canonical config.yaml file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_seed = config["seed"]
    n_permutations = config.get("n_permutation_tests", 1000)
    model_name = config.get("model_name", "pythia-1.4b")
    output_dir = Path(config.get("output_dir", "results/rq2_probing"))

    dataset_path = Path(config.get("dataset_path", "data/processed/dataset_master_v5.jsonl"))
    tensors_dir = Path("data/processed") / model_name
    weights_dir = output_dir / "weights"
    test_idx_path = output_dir / "test_indices" / "sign_test_idx.npy"
    out_csv = output_dir / "confound_checks_hardened.csv"

    for p in [dataset_path, tensors_dir, weights_dir, test_idx_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing dependency: {p}. Ensure RQ2 has run successfully.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(tensors_dir / "metadata.json", encoding="utf-8") as f:
        _meta = json.load(f)
    with open(dataset_path, encoding="utf-8") as f:
        _jsonl_ids = [json.loads(line)["id"] for line in f if line.strip()]
    if _meta.get("stimuli_ids") != _jsonl_ids:
        raise ValueError(
            "Index-space mismatch: metadata.json stimuli_ids order != JSONL line order. "
            "Tensor rows and operand labels would be misaligned."
        )

    logger.info("Parsing dataset dictionaries for CAT-SIGN tokens...")
    sign_global_indices = []
    y_op1_list = []
    y_mag_list = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for global_idx, line in enumerate(f):
            stimulus = json.loads(line)
            if stimulus.get("category") == "CAT-SIGN":
                sign_global_indices.append(global_idx)
                y_op1_list.append(extract_operand1(stimulus))
                y_mag_list.append(extract_magnitude_delta(stimulus))

    y_op1 = np.array(y_op1_list)
    y_mag = np.array(y_mag_list)
    logger.info(f"Loaded {len(sign_global_indices)} CAT-SIGN elements for confound verification.")

    test_idx_global = np.load(test_idx_path)
    test_set = set(test_idx_global.tolist())
    train_idx_global = np.array([i for i in sign_global_indices if i not in test_set])

    # Map global indices to subset targets
    g2op1 = {g: o for g, o in zip(sign_global_indices, y_op1)}
    g2mag = {g: m for g, m in zip(sign_global_indices, y_mag)}

    y_train_op1 = np.array([g2op1[i] for i in train_idx_global])
    y_test_op1 = np.array([g2op1[i] for i in test_idx_global])

    y_train_mag = np.array([g2mag[i] for i in train_idx_global])
    y_test_mag = np.array([g2mag[i] for i in test_idx_global])

    layer_files = sorted(tensors_dir.glob("layer_*.pt"))
    n_layers = len(layer_files)
    if n_layers == 0:
        raise ValueError(f"No extracted tensors found in {tensors_dir}. Run extraction first.")

    raw_results = []
    logger.info(f"Initiating multi-dimensional check across {n_layers} layers with {n_permutations} permutations...")

    for l in range(n_layers):
        tensor_path = tensors_dir / f"layer_{l:02d}.pt"
        weight_path = weights_dir / f"layer_{l:02d}_sign.npy"
        bias_path = weights_dir / f"layer_{l:02d}_sign_bias.npy"

        if not weight_path.exists() or not bias_path.exists():
            logger.warning(f"Probe weights/bias missing for layer {l:02d}. Skipping.")
            continue

        # Core seed allocation isolated per layer
        rng = np.random.default_rng(get_seed(base_seed, "confound_permutation", l))

        H_full = torch.load(tensor_path, map_location="cpu", weights_only=True).float().numpy()
        X_train = H_full[train_idx_global]
        X_test = H_full[test_idx_global]

        w_sign = np.load(weight_path).flatten()
        b_sign = np.load(bias_path).flatten()[0]

        # ── VERIFICATION 1: Train an explicit Operand-1 Control Probe ──
        # Directly measures if operand1 is linearizable from the activation space
        op1_probe = LinearRegression()
        op1_probe.fit(X_train, y_train_op1)
        op1_r2 = float(op1_probe.score(X_test, y_test_op1))

        # Permutation gating for Operand-1 control R²
        null_r2s_op1 = np.array([
            LinearRegression().fit(X_train, rng.permutation(y_train_op1)).score(X_test, y_test_op1)
            for _ in range(n_permutations)
        ])
        op1_pvalue = float((np.sum(null_r2s_op1 >= op1_r2) + 1) / (n_permutations + 1))

        # ── VERIFICATION 2: Vector Alignment (Cosine Similarity) ──
        w_op1 = op1_probe.coef_.flatten()
        cos_sign_vs_op1 = float(cosine_similarity(w_sign, w_op1))

        # ── VERIFICATION 3: Train a Delta Magnitude Control Probe ──
        # Does the layer encode the absolute size of the difference |a - b|?
        mag_probe = LinearRegression()
        mag_probe.fit(X_train, y_train_mag)
        mag_r2 = float(mag_probe.score(X_test, y_test_mag))

        w_mag = mag_probe.coef_.flatten()
        cos_sign_vs_mag = float(cosine_similarity(w_sign, w_mag))

        # ── VERIFICATION 4: Direct Behavior Leakage Check (Critical Triangulation) ──
        # Evaluates if the frozen sign probe's logits correlate directly with operand1's scale.
        # If it acts as a shortcut proxy, its predictions will align with operand1 bounds.
        raw_preds = np.dot(X_test, w_sign) + b_sign

        # Pearson correlation between the frozen sign probe's logits and operand1 values
        if np.var(raw_preds) > 1e-9 and np.var(y_test_op1) > 1e-9:
            corr_matrix = np.corrcoef(raw_preds, y_test_op1)
            logit_corr_with_op1 = float(corr_matrix[0, 1])
        else:
            logit_corr_with_op1 = 0.0

        raw_results.append({
            "layer": l,
            "op1_decodability_r2": op1_r2,
            "op1_r2_raw_pvalue": op1_pvalue,
            "cosine_sign_vs_op1": cos_sign_vs_op1,
            "mag_delta_r2": mag_r2,
            "cosine_sign_vs_mag": cos_sign_vs_mag,
            "sign_logits_correlation_with_op1": logit_corr_with_op1,
            "is_significant_op1_leak": False  # Handled by Benjamini-Hochberg downstream
        })

    # ── SECTION 5 — MULTIPLE COMPARISON CORRECTION (E-M-05) ───────────────────
    # Apply Benjamini-Hochberg FDR correction across all layers to secure the p-values
    p_values_op1 = [res["op1_r2_raw_pvalue"] for res in raw_results]
    fdr_gating = benjamini_hochberg_correction(p_values_op1, fdr_level=0.05)

    for idx, is_significant in enumerate(fdr_gating):
        raw_results[idx]["is_significant_op1_leak"] = is_significant

        # Hard Sentinel Alarm flags
        c_op1 = raw_results[idx]["cosine_sign_vs_op1"]
        c_corr = raw_results[idx]["sign_logits_correlation_with_op1"]

        if abs(c_op1) > 0.50 or abs(c_corr) > 0.50:
            layer_idx = raw_results[idx]["layer"]
            logger.warning(
                f"[CRITICAL CONFOUND] Layer {layer_idx:02d}: Sign probe is heavily contaminated by Confound N-01! "
                f"Cosine alignment: {c_op1:.3f} | Logit Correlation: {c_corr:.3f}. "
                f"The probe is partially decoding operand magnitude shortcuts instead of abstract mathematical sign."
            )

    # Atomic serialization output dump to preserve disk tracking invariants
    df_out = pd.DataFrame(raw_results)
    _atomic_write_csv(out_csv, df_out.to_dict("records"), df_out.columns.tolist())
    logger.info(f"[✔] Confound analysis successfully committed to disk -> {out_csv}")


if __name__ == "__main__":
    main()
