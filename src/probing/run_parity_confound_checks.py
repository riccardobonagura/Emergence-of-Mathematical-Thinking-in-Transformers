"""
run_parity_confound_checks.py — Parity Confound Mitigation (N-02) for CAT-PARITY.

Sibling of run_confound_checks.py (which covers the sign confound N-01).
CAT-PARITY pairs vary only the second operand (b vs b+1) at fixed first operand
and fixed operator. The natural shortcut is therefore "read the parity of the
changing operand token" instead of decoding the parity of the *result*.

This module asks, per layer, whether the frozen RQ2 parity probe genuinely
decodes result parity or collapses into an operand2-parity proxy:

  V1  operand2 *value* decodability (LinearRegression R², permutation-gated)
      + cosine(w_parity, w_op2value)              — magnitude leakage of operand2
  V2  operand2 *parity* decodability (LogisticRegression accuracy)
      + cosine(w_parity, w_op2parity)             — direction alignment with the shortcut
  V3  direct triangulation: Pearson(frozen parity-probe logits, operand2 parity)

The shortcut is only *viable* when result parity and operand2 parity coincide,
which happens iff the first operand parity is fixed. The design breaks this by
drawing the first operand across [10,50] (both parities). Two dataset-level
diagnostics make that protection auditable instead of assumed:

  - first-operand parity balance among CAT-PARITY (should be ≈ 0.5)
  - ground-truth corr(result parity, operand2 parity) on the test split
    (should be ≈ 0; if high, V3 high reflects a dataset confound, not a probe
    shortcut — the triangulation that separates the two cases)

Benjamini-Hochberg FDR correction is applied across layers on the operand2
value R² p-values (E-M-05), mirroring N-01.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.probing.directions import cosine_similarity
from src.probing.seeds import get_seed
from src.probing.io_utils import _atomic_write_csv
from src.probing.stats import benjamini_hochberg_correction


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("parity_confound_check")


def main() -> None:
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Parity Confound (N-02) Verification Suite")
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
    test_idx_path = output_dir / "test_indices" / "parity_test_idx.npy"
    out_csv = output_dir / "parity_confound_checks.csv"

    for p in [dataset_path, tensors_dir, weights_dir, test_idx_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing dependency: {p}. Ensure RQ2 (parity) has run successfully.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Parse CAT-PARITY stimuli: operand2 value, operand2 parity, result parity, op1 parity ──
    logger.info("Parsing dataset dictionaries for CAT-PARITY tokens...")
    par_global_indices = []
    y_op2_list = []        # operand2 value (continuous)
    y_op2par_list = []     # operand2 parity (0/1)
    y_resultpar_list = []  # ground-truth result parity (0/1)
    op1_parities = []      # first-operand parity (0/1) — protection diagnostic

    with open(dataset_path, "r", encoding="utf-8") as f:
        for global_idx, line in enumerate(f):
            stimulus = json.loads(line)
            if stimulus.get("category") != "CAT-PARITY":
                continue
            labels = stimulus["labels"]
            par_global_indices.append(global_idx)
            y_op2_list.append(float(labels["operand2"]))
            y_op2par_list.append(int(labels["operand2"]) % 2)
            y_resultpar_list.append(int(labels["parity"]))
            op1_parities.append(int(labels["operand1"]) % 2)

    y_op2 = np.array(y_op2_list, dtype=np.float64)
    y_op2par = np.array(y_op2par_list, dtype=np.int64)
    logger.info(f"Loaded {len(par_global_indices)} CAT-PARITY elements for confound verification.")

    # ── Dataset-level protection diagnostic: first-operand parity balance ──
    frac_op1_even = float(np.mean(np.array(op1_parities) == 0))
    logger.info(
        f"First-operand parity balance (protective factor): "
        f"P(operand1 even) = {frac_op1_even:.3f} (target ≈ 0.5)."
    )
    if abs(frac_op1_even - 0.5) > 0.1:
        logger.warning(
            "[N-02 RISK] First-operand parity is imbalanced. result parity and operand2 "
            "parity may coincide, making the operand2-parity shortcut viable. Interpret "
            "high V3 correlation as a dataset confound rather than a probe shortcut."
        )

    test_idx_global = np.load(test_idx_path)
    test_set = set(test_idx_global.tolist())
    # Mirrors N-01: CAT-PARITY is class-balanced, so the RQ2 train partition is
    # exactly the CAT-PARITY pool minus the frozen test split.
    train_idx_global = np.array([i for i in par_global_indices if i not in test_set])

    g2op2 = {g: v for g, v in zip(par_global_indices, y_op2)}
    g2op2par = {g: v for g, v in zip(par_global_indices, y_op2par)}
    g2resultpar = {g: v for g, v in zip(par_global_indices, y_resultpar_list)}

    y_train_op2 = np.array([g2op2[i] for i in train_idx_global])
    y_test_op2 = np.array([g2op2[i] for i in test_idx_global])
    y_train_op2par = np.array([g2op2par[i] for i in train_idx_global])
    y_test_op2par = np.array([g2op2par[i] for i in test_idx_global])
    y_test_resultpar = np.array([g2resultpar[i] for i in test_idx_global])

    # Ground-truth confound strength on the test split: if ≈ 0, result parity and
    # operand2 parity are decorrelated (design working as intended).
    if np.var(y_test_resultpar) > 1e-9 and np.var(y_test_op2par) > 1e-9:
        gt_corr = float(np.corrcoef(y_test_resultpar, y_test_op2par)[0, 1])
    else:
        gt_corr = 0.0
    logger.info(
        f"Ground-truth corr(result parity, operand2 parity) on test split = {gt_corr:.3f} "
        f"(target ≈ 0; high values indicate a dataset-level confound)."
    )

    layer_files = sorted(tensors_dir.glob("layer_*.pt"))
    n_layers = len(layer_files)
    if n_layers == 0:
        raise ValueError(f"No extracted tensors found in {tensors_dir}. Run extraction first.")

    raw_results = []
    logger.info(f"Initiating parity confound check across {n_layers} layers with {n_permutations} permutations...")

    for l in range(n_layers):
        tensor_path = tensors_dir / f"layer_{l:02d}.pt"
        weight_path = weights_dir / f"layer_{l:02d}_parity.npy"
        bias_path = weights_dir / f"layer_{l:02d}_parity_bias.npy"

        if not weight_path.exists() or not bias_path.exists():
            logger.warning(f"Parity probe weights/bias missing for layer {l:02d}. Skipping.")
            continue

        rng = np.random.default_rng(get_seed(base_seed, "parity_confound_permutation", l))

        H_full = torch.load(tensor_path, map_location="cpu", weights_only=True).float().numpy()
        X_train = H_full[train_idx_global]
        X_test = H_full[test_idx_global]

        w_parity = np.load(weight_path).flatten()
        b_parity = np.load(bias_path).flatten()[0]

        # ── V1: operand2 value decodability (magnitude of the changing operand) ──
        op2_probe = LinearRegression()
        op2_probe.fit(X_train, y_train_op2)
        op2_r2 = float(op2_probe.score(X_test, y_test_op2))

        null_r2s_op2 = np.array([
            LinearRegression().fit(X_train, rng.permutation(y_train_op2)).score(X_test, y_test_op2)
            for _ in range(n_permutations)
        ])
        op2_pvalue = float((null_r2s_op2 >= op2_r2).mean())
        cos_parity_vs_op2 = float(cosine_similarity(w_parity, op2_probe.coef_.flatten()))

        # ── V2: operand2 parity decodability (the direct shortcut direction) ──
        if len(np.unique(y_train_op2par)) < 2:
            op2par_acc = float("nan")
            cos_parity_vs_op2par = float("nan")
        else:
            op2par_probe = LogisticRegression(max_iter=1000)
            op2par_probe.fit(X_train, y_train_op2par)
            op2par_acc = float(op2par_probe.score(X_test, y_test_op2par))
            cos_parity_vs_op2par = float(cosine_similarity(w_parity, op2par_probe.coef_.flatten()))

        # ── V3: direct triangulation — frozen parity-probe logits vs operand2 parity ──
        raw_preds = np.dot(X_test, w_parity) + b_parity
        if np.var(raw_preds) > 1e-9 and np.var(y_test_op2par) > 1e-9:
            logit_corr_with_op2par = float(np.corrcoef(raw_preds, y_test_op2par)[0, 1])
        else:
            logit_corr_with_op2par = 0.0

        raw_results.append({
            "layer": l,
            "op2_value_r2": op2_r2,
            "op2_value_r2_raw_pvalue": op2_pvalue,
            "cosine_parity_vs_op2value": cos_parity_vs_op2,
            "op2_parity_decode_acc": op2par_acc,
            "cosine_parity_vs_op2parity": cos_parity_vs_op2par,
            "parity_logits_correlation_with_op2parity": logit_corr_with_op2par,
            "gt_resultparity_vs_op2parity_corr": gt_corr,
            "frac_operand1_even": frac_op1_even,
            "is_significant_op2_leak": False,  # set by Benjamini-Hochberg below
        })

    # ── MULTIPLE COMPARISON CORRECTION (E-M-05) ──
    p_values_op2 = [res["op2_value_r2_raw_pvalue"] for res in raw_results]
    fdr_gating = benjamini_hochberg_correction(p_values_op2, fdr_level=0.05)

    for idx, is_significant in enumerate(fdr_gating):
        raw_results[idx]["is_significant_op2_leak"] = is_significant

        c_op2par = raw_results[idx]["cosine_parity_vs_op2parity"]
        c_corr = raw_results[idx]["parity_logits_correlation_with_op2parity"]

        # Sentinel alarm: shortcut suspicion only when ground-truth confound is weak.
        suspicious_alignment = (
            (not np.isnan(c_op2par) and abs(c_op2par) > 0.50) or abs(c_corr) > 0.50
        )
        if suspicious_alignment and abs(gt_corr) < 0.30:
            layer_idx = raw_results[idx]["layer"]
            logger.warning(
                f"[CRITICAL CONFOUND] Layer {layer_idx:02d}: Parity probe aligns with operand2 "
                f"parity (cosine={c_op2par:.3f}, logit-corr={c_corr:.3f}) while the dataset-level "
                f"confound is weak (gt_corr={gt_corr:.3f}). The probe may be reading the changing "
                f"operand's parity rather than the abstract result parity."
            )

    df_out = pd.DataFrame(raw_results)
    _atomic_write_csv(out_csv, df_out.to_dict("records"), df_out.columns.tolist())
    logger.info(f"[✔] Parity confound analysis committed to disk -> {out_csv}")


if __name__ == "__main__":
    main()
