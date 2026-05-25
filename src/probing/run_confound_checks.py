"""
run_confound_checks.py — Confound Mitigation Module (T02).
Tests the N-01 hypothesis: does the "sign" probe actually encode magnitude |a-b|?
Trains a linear regression on magnitude and compares its direction to the sign probe.
"""

import sys
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

try:
    from src.probing.directions import cosine_similarity
except ImportError:
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        v1, v2 = v1.flatten(), v2.flatten()
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

try:
    from src.probing.seeds import get_seed
except ImportError:
    def get_seed(base_seed: int, operation: str, index: int = 0) -> int:
        return base_seed + hash(operation) % 10000

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("confound_check")

def extract_magnitude(text: str) -> float:
    """Extracts the magnitude |a - b| from the stimulus text."""
    nums = re.findall(r'\b\d+\b', text)
    if len(nums) >= 2:
        return float(abs(int(nums[0]) - int(nums[1])))
    return 0.0

def main() -> None:
    logger = setup_logger()
    
    dataset_path = Path("data/processed/dataset_master_v5.jsonl")
    tensors_dir = Path("data/processed/pythia-1.4b")
    weights_dir = Path("results/rq2_probing/weights")
    test_idx_path = Path("results/rq2_probing/test_indices/sign_test_idx.npy")
    out_csv = Path("results/rq2_probing/confound_checks.csv")

    for p in [dataset_path, tensors_dir, weights_dir, test_idx_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing dependency: {p}. Ensure RQ2 has run successfully.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing dataset for CAT-SIGN magnitudes...")
    sign_global_indices = []
    y_mag_list = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for global_idx, line in enumerate(f):
            stimulus = json.loads(line)
            if stimulus.get("category") == "CAT-SIGN":
                sign_global_indices.append(global_idx)
                y_mag_list.append(extract_magnitude(stimulus["text"]))
                
    y_mag = np.array(y_mag_list)
    logger.info(f"Found {len(sign_global_indices)} CAT-SIGN stimuli.")

    # Global indexing schema to prevent merge-order dependency
    test_idx_global = np.load(test_idx_path)
    sign_set = set(sign_global_indices)
    test_set = set(test_idx_global.tolist())
    train_idx_global = np.array([i for i in sign_global_indices if i not in test_set])

    g2mag = {g: m for g, m in zip(sign_global_indices, y_mag)}
    y_train = np.array([g2mag[i] for i in train_idx_global])
    y_test = np.array([g2mag[i] for i in test_idx_global])

    layer_files = sorted(tensors_dir.glob("layer_*.pt"))
    n_layers = len(layer_files)
    if n_layers == 0:
        raise ValueError(f"No extracted tensors found in {tensors_dir}")

    results = []
    logger.info(f"Initiating Confound Checks across {n_layers} layers...")

    for l in range(n_layers):
        tensor_path = tensors_dir / f"layer_{l:02d}.pt"
        weight_path = weights_dir / f"layer_{l:02d}_sign.npy"
        
        if not weight_path.exists():
            logger.warning(f"Probe weights missing for layer {l:02d}. Skipping.")
            continue

        # CRITICAL FIX: Layer-bound deterministic RNG instantiation
        rng = np.random.default_rng(get_seed(42, "permutation", l))

        # Load representations directly into the global index schema
        H_full = torch.load(tensor_path, map_location="cpu", weights_only=True).float().numpy()
        X_train = H_full[train_idx_global]
        X_test = H_full[test_idx_global]

        # A) Train Magnitude Probe (Linear Regression)
        mag_probe = LinearRegression()
        mag_probe.fit(X_train, y_train)
        mag_r2 = mag_probe.score(X_test, y_test)
        
        # B) Custom R2 Permutation Test
        null_r2s = np.array([
            LinearRegression().fit(X_train, rng.permutation(y_train)).score(X_test, y_test)
            for _ in range(100)
        ])
        mag_r2_pvalue = float((null_r2s >= mag_r2).mean())

        # C) Compare Directions
        w_mag = mag_probe.coef_.flatten()
        w_sign = np.load(weight_path).flatten()
        cos_sim = cosine_similarity(w_sign, w_mag)
        
        results.append({
            "layer": l,
            "mag_r2": mag_r2,
            "mag_r2_pvalue": mag_r2_pvalue,
            "cosine_sign_vs_mag": cos_sim
        })

        # D) N-01 Sentinel Flag
        if abs(cos_sim) > 0.8:
            logger.warning(f"[WARNING] Layer {l:02d}: sign probe direction collinear with magnitude (cos={cos_sim:.3f}).")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    logger.info(f"Confound checks complete. Results saved to {out_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger("confound_check")
        logger.error(f"Execution failed: {e}")
        sys.exit(1)