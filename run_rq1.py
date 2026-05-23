"""
run_rq1.py — RQ1 orchestrator: isotropy + evolutionary CKA to locate l*.
Pipeline: load metadata → run_isotropy_analysis → compute layer-wise CKA evolution.
"""

import numpy as np
from pathlib import Path

from src.config.categories import MATH_CATS, CTRL_CATS
from src.metrics.isotropy import run_isotropy_analysis
from src.metrics.cka      import linear_cka
from src.probing.io_utils import MetadataHandler
import src.probing.io_utils as io

N_LAYERS   = 24     # Pythia-1.4B
SEED       = 42


def main() -> None:
    PROC_DIR     = Path("data/processed/pythia-1.4b")
    STIMULI_PATH = Path("data/processed/dataset_master_v5.jsonl")
    OUT_DIR      = Path("results/rq1_emergence")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not (PROC_DIR / "metadata.json").exists():
        raise FileNotFoundError(f"Extraction incomplete — missing {PROC_DIR}/metadata.json")

    print("\n--- RQ1 EMERGENCE ANALYSIS ---")

    # 1. Layer-wise isotropy per category
    print("\n1. Isotropy (all categories, all layers)...")
    df_iso = run_isotropy_analysis(
        processed_dir = str(PROC_DIR),
        stimuli_path  = str(STIMULI_PATH),
        output_path   = str(OUT_DIR / "isotropy_pythia.csv"),
        n_layers      = N_LAYERS,
        seed          = SEED,
    )
    print(f"   {len(df_iso)} layer/category records saved.")

    # 2. Evolutionary CKA: CKA(H_l, H_{l-1}) for math and ctrl separately
    print("\n2. Evolutionary CKA (layer-to-layer similarity)...")

    meta       = MetadataHandler(PROC_DIR / "metadata.json")
    categories = np.array(meta.data["categories"])

    math_idx = np.where(np.isin(categories, list(MATH_CATS)))[0]
    ctrl_idx = np.where(np.isin(categories, list(CTRL_CATS)))[0]

    if math_idx.size == 0 or ctrl_idx.size == 0:
        print("  WARNING: expected categories not found in metadata.")
        print(f"  Detected: {np.unique(categories).tolist()}")
        return

    # Seed layer 0 — load via io helper (FP16→FP32 cast applied there)
    H_prev      = io.load_hidden_states(PROC_DIR / "layer_00.pt").astype(np.float64)
    H_prev_math = H_prev[math_idx]
    H_prev_ctrl = H_prev[ctrl_idx]

    cka_math = [1.0]   # CKA(l0, l0) = 1 by definition
    cka_ctrl = [1.0]

    for l in range(1, N_LAYERS):
        H_curr      = io.load_hidden_states(PROC_DIR / f"layer_{l:02d}.pt").astype(np.float64)
        H_curr_math = H_curr[math_idx]
        H_curr_ctrl = H_curr[ctrl_idx]

        cka_m = linear_cka(H_prev_math, H_curr_math)
        cka_c = linear_cka(H_prev_ctrl, H_curr_ctrl)

        cka_math.append(cka_m)
        cka_ctrl.append(cka_c)

        # Update rolling previous layer
        H_prev_math = H_curr_math
        H_prev_ctrl = H_curr_ctrl

        print(f"   layer {l:02d} | CKA math={cka_m:.4f}  ctrl={cka_c:.4f}")

    np.save(OUT_DIR / "cka_math_evol.npy", np.array(cka_math))
    np.save(OUT_DIR / "cka_ctrl_evol.npy", np.array(cka_ctrl))
    print("   Evolutionary CKA saved.")

    print("\n--- RQ1 COMPLETE ---")


if __name__ == "__main__":
    main()