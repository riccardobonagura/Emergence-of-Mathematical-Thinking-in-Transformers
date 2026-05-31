"""
Linear Centered Kernel Alignment (CKA) for comparing representation spaces.

Linear CKA uses linear Gram matrices K = X X^T, L = Y Y^T, centered with
H = I_n - (1/n) 1 1^T (so K_c = H K H), then
    CKA(X, Y) = <K_c, L_c>_F / sqrt(<K_c, K_c>_F <L_c, L_c>_F).

Three usage modes back the project's research questions:
  1. Intra-model CKA  — L×L matrix comparing every layer pair of one model.
  2. Inter-category   — per layer, math vs generic-text representations (RQ1).
  3. Cross-temporal   — base model vs QLoRA checkpoints, per layer (RQ3).

The live pipeline (run_rq1.py) uses only `linear_cka` and
`compute_cka_intercategory`; the other helpers are kept as a reusable API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from src.probing.io_utils import _atomic_save_npy

logger = logging.getLogger("cka")


# ── Math primitive: center_gram ──────────────────────────────────────────────

def center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix: K' = H K H, making CKA invariant to translations.

    Uses the algebraic identity H K H = K - row_mean - col_mean + global_mean
    to avoid materializing the n×n centering matrix H.

    Args:
        K: Gram matrix (n, n), typically X @ X.T.
    Returns:
        Centered Gram matrix, same shape as K.
    """
    row_mean = K.mean(axis=1, keepdims=True)   # (n, 1)
    col_mean = K.mean(axis=0, keepdims=True)   # (1, n)
    global_mean = K.mean()
    return K - row_mean - col_mean + global_mean


# ── Math primitive: linear_cka ───────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation sets X and Y.

    Measures how similar the relational structure of the samples is across the
    two spaces; invariant to orthogonal transforms and isotropic scaling, and
    independent of dimensionality (d_X may differ from d_Y). Returns a value in
    [0, 1]: 1.0 = identical geometry (up to rotation/scaling), 0.0 = unrelated.

    Args:
        X: Layer activations (n_samples, d_X).
        Y: Layer activations (n_samples, d_Y). Same n_samples as X.
    Returns:
        CKA value in [0, 1].
    Raises:
        ValueError: if X and Y have a different number of samples.
        RuntimeError: if a representation is (near-)constant (zero norm).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples. "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}"
        )

    # Linear-kernel Gram matrices: K[i,j] = X[i] . X[j].
    K = X @ X.T   # (n, n)
    L = Y @ Y.T   # (n, n)

    K_c = center_gram(K)
    L_c = center_gram(L)

    # Frobenius inner products: <A, B>_F = sum(A * B).
    hsic_kl = np.sum(K_c * L_c)   # HSIC(X, Y)
    hsic_kk = np.sum(K_c * K_c)   # HSIC(X, X)
    hsic_ll = np.sum(L_c * L_c)   # HSIC(Y, Y)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        # Degenerate case: a representation is constant (centered Gram = 0).
        raise RuntimeError(
            "Near-zero norm: representations are constant or nearly so. "
            "Check that hidden states were extracted correctly."
        )

    return float(hsic_kl / denom)


def cka_matrix_across_layers(
    activations_per_layer: Iterable[np.ndarray | torch.Tensor],
) -> np.ndarray:
    """Build a layer×layer CKA matrix from a sequence of per-layer activations.

    Args:
        activations_per_layer: one array/tensor per layer, each [n_samples, d_l]
            with the same n_samples across layers.
    Returns:
        Symmetric CKA matrix [L, L].
    """
    layers = [_to_numpy_2d(x) for x in activations_per_layer]
    n_layers = len(layers)
    if n_layers == 0:
        raise ValueError("activations_per_layer is empty.")

    n_samples = layers[0].shape[0]
    for idx, arr in enumerate(layers):
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent sample count at layer {idx}: "
                f"{arr.shape[0]} vs expected {n_samples}"
            )

    cka_mat = np.zeros((n_layers, n_layers), dtype=np.float64)
    for i in range(n_layers):
        for j in range(i, n_layers):
            val = linear_cka(layers[i], layers[j])
            cka_mat[i, j] = val
            cka_mat[j, i] = val
    return cka_mat


def _to_numpy_2d(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array [n, d], got shape={arr.shape}")
    return arr


# ── Reproducible subsampling ─────────────────────────────────────────────────

def subsample_indices(n_total: int, n_sub: int, seed: int = 42) -> np.ndarray:
    """Pick n_sub reproducible random indices from [0, n_total).

    Subsampling keeps the n×n Gram matrices tractable; CKA estimates are stable
    for n_sub >= 256. Callers should pass a project seed (get_seed); the 42
    default preserves historical behavior.

    Args:
        n_total: total available samples.
        n_sub: number to select (n_sub <= n_total).
        seed: RNG seed.
    Returns:
        Sorted array of n_sub unique integer indices.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_sub, n_total), replace=False)
    return np.sort(indices)


# ── Mode 1: intra-model CKA (L×L matrix) ─────────────────────────────────────

def compute_cka_matrix_intramodel(
    hidden_states_dir: Path,
    n_layers: int,
    n_sub: int = 512,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """Compute the L×L CKA matrix comparing every layer pair of one model.

    High-CKA contiguous blocks indicate processing "stages"; sharp drops mark
    representational transitions. Expects files hidden_states_dir/layer_XX.pt.

    Args:
        hidden_states_dir: directory with layer_XX.pt files.
        n_layers: number of model layers.
        n_sub: subsample size.
        seed: subsampling seed.
        device: "cpu" or "cuda".
    Returns:
        Symmetric (n_layers, n_layers) CKA matrix, values in [0, 1].
    """
    first_layer_path = hidden_states_dir / "layer_00.pt"
    H_first = torch.load(first_layer_path, map_location=device).cpu().numpy()  # (N, d)
    N = H_first.shape[0]
    # Work in float64: fp16 Gram matrices (X @ X.T over d=2048) lose precision
    # and risk overflow; this matches the other loaders.

    # Fixed subsample indices reused for every layer -> consistent comparison.
    sub_idx = subsample_indices(n_total=N, n_sub=n_sub, seed=seed)

    logger.info(
        "Loading hidden states for %d layers (subsample N_sub=%d of N=%d)...",
        n_layers, n_sub, N,
    )
    H_sub_all = []  # one (n_sub, d) array per layer
    for l in range(n_layers):
        layer_path = hidden_states_dir / f"layer_{l:02d}.pt"
        H_l = torch.load(layer_path, map_location=device).cpu().numpy().astype(np.float64)  # (N, d)
        H_sub_all.append(H_l[sub_idx])  # (n_sub, d)

    cka_matrix = np.zeros((n_layers, n_layers))

    # Symmetric: compute the upper triangle (incl. diagonal) and mirror.
    total_pairs = n_layers * (n_layers + 1) // 2
    pbar = tqdm(total=total_pairs, desc="Intra-model CKA")
    for l1 in range(n_layers):
        for l2 in range(l1, n_layers):
            cka_val = linear_cka(H_sub_all[l1], H_sub_all[l2])
            cka_matrix[l1, l2] = cka_val
            cka_matrix[l2, l1] = cka_val
            pbar.update(1)
    pbar.close()
    return cka_matrix


# ── Mode 2: inter-category CKA (RQ1 — geometric bifurcation) ─────────────────

def compute_cka_intercategory(
    H_math: np.ndarray,
    H_generic: np.ndarray,
    seed: int | None = None,
) -> float:
    """CKA between math and generic-text representations at a single layer.

    A low value means the two categories occupy structurally different
    geometries at that layer. If the two sets differ in size, a balanced
    subsample of min(n1, n2) is drawn from each.

    Args:
        H_math: math-stimulus hidden states (n1, d).
        H_generic: generic-text hidden states (n2, d).
        seed: subsampling seed used when n1 != n2; None falls back to 42.
            Callers should pass get_seed(...) for the project seed discipline.
    Returns:
        Inter-category CKA value in [0, 1].
    """
    n1, n2 = H_math.shape[0], H_generic.shape[0]

    if n1 != n2:
        n_common = min(n1, n2)
        rng = np.random.default_rng(seed if seed is not None else 42)
        idx_math    = rng.choice(n1, size=n_common, replace=False)
        idx_generic = rng.choice(n2, size=n_common, replace=False)
        H_math    = H_math[idx_math]
        H_generic = H_generic[idx_generic]

    return linear_cka(H_math, H_generic)


def compute_cka_intercategory_all_layers(
    hidden_states_dir: Path,
    n_layers: int,
    math_indices: np.ndarray,
    generic_indices: np.ndarray,
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """Compute inter-category CKA for every layer.

    Produces CKA_inter(l); a sharp drop from layer l* marks the math/text
    geometric bifurcation (RQ1).

    Args:
        hidden_states_dir: directory with layer_XX.pt.
        n_layers: number of layers.
        math_indices: indices of math stimuli in the hidden-state tensor.
        generic_indices: indices of generic stimuli.
        device: "cpu" or "cuda".
        seed: subsampling seed.
    Returns:
        Array (n_layers,) of inter-category CKA values.
    """
    cka_intercategory = np.zeros(n_layers)

    for l in tqdm(range(n_layers), desc="Inter-category CKA per layer"):
        layer_path = hidden_states_dir / f"layer_{l:02d}.pt"
        H_l = torch.load(layer_path, map_location=device).cpu().numpy().astype(np.float64)  # (N, d)

        H_math    = H_l[math_indices]     # (n_math, d)
        H_generic = H_l[generic_indices]  # (n_generic, d)

        cka_intercategory[l] = compute_cka_intercategory(H_math, H_generic, seed=seed)

    return cka_intercategory


# ── Mode 3: cross-temporal CKA (RQ3 — fine-tuning) ───────────────────────────

def compute_cka_cross_temporal(
    base_hidden_states_dir: Path,
    checkpoint_hidden_states_dirs: dict[str, Path],
    n_layers: int,
    n_sub: int = 512,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Per-layer CKA between the base model and each QLoRA checkpoint.

    For each checkpoint c and layer l computes S_l^c = CKA(H_l^base, H_l^c),
    showing which layers were most reorganized by fine-tuning (RQ3).

    Args:
        base_hidden_states_dir: base-model hidden-state directory.
        checkpoint_hidden_states_dirs: mapping step -> checkpoint directory.
        n_layers: number of layers.
        n_sub: subsample size.
        seed: subsampling seed.
        device: "cpu" or "cuda".
    Returns:
        Mapping step -> array (n_layers,). The "base" key holds ones
        (CKA with itself = 1.0, a sanity check).
    """
    # Fixed subsample, shared across checkpoints. Load layer 0 just to get N.
    H_tmp = torch.load(
        base_hidden_states_dir / "layer_00.pt", map_location=device
    ).numpy()
    N = H_tmp.shape[0]
    del H_tmp

    sub_idx = subsample_indices(n_total=N, n_sub=n_sub, seed=seed)

    logger.info("Loading base-model hidden states...")
    H_base = []
    for l in range(n_layers):
        H_l = torch.load(
            base_hidden_states_dir / f"layer_{l:02d}.pt", map_location=device
        ).numpy().astype(np.float64)
        H_base.append(H_l[sub_idx])  # (n_sub, d)

    results = {}
    results["base"] = np.ones(n_layers)  # CKA(base, base) == 1.0 per layer

    for ckpt_name, ckpt_dir in checkpoint_hidden_states_dirs.items():
        logger.info("Cross-temporal CKA: base vs %s...", ckpt_name)
        cka_values = np.zeros(n_layers)

        for l in tqdm(range(n_layers), desc=f"  Layer ({ckpt_name})"):
            H_ckpt_l = torch.load(
                ckpt_dir / f"layer_{l:02d}.pt", map_location=device
            ).numpy().astype(np.float64)
            H_ckpt_sub = H_ckpt_l[sub_idx]  # (n_sub, d) — same subsample

            cka_values[l] = linear_cka(H_base[l], H_ckpt_sub)

        results[ckpt_name] = cka_values

    return results


def compute_cka_drift(cka_cross_temporal: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """CKA drift: drift(l, c) = 1 - CKA(base_l, ckpt_c_l).

    High drift means strong geometric reorganization at that layer. The "base"
    key is skipped (drift = 0).

    Args:
        cka_cross_temporal: output of compute_cka_cross_temporal().
    Returns:
        Mapping ckpt_name -> drift array (n_layers,).
    """
    drift_results = {}
    for ckpt_name, cka_values in cka_cross_temporal.items():
        if ckpt_name == "base":
            continue
        drift_results[ckpt_name] = 1.0 - cka_values
    return drift_results


# ── RQ1 robustness battery (E-G-02) ──────────────────────────────────────────
# These corroborate (or refute) an inter-category CKA divergence claim. They are
# additive helpers; linear_cka stays the biased estimator the rest of the pipeline
# (and the self-CKA==1.0 assertions) depend on.

def debiased_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased linear CKA via the unbiased HSIC estimator (Song et al. 2012).

    Uses the unbiased HSIC of Song et al. (2012), as adopted for CKA by
    Nguyen, Raghu & Kornblith (2021). Unlike `linear_cka` (which keeps the biased
    HSIC the rest of the pipeline relies on), this removes the small-sample upward
    bias and can therefore return slightly negative values for unrelated
    representations — that is expected and is NOT clipped to [0, 1].

    Args:
        X: activations (n, d_X).
        Y: activations (n, d_Y). Same n as X.
    Returns:
        Debiased CKA, typically in [0, 1] but may dip slightly below 0.
    Raises:
        ValueError: if n != n or n < 4 (the estimator is undefined for n <= 3).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples. "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}"
        )
    n = X.shape[0]
    if n < 4:
        raise ValueError(f"debiased HSIC is undefined for n < 4 (got n={n}).")

    K = X @ X.T
    L = Y @ Y.T

    def _hsic1(A: np.ndarray, B: np.ndarray) -> float:
        # Zero the diagonals (tilde matrices), then the Song et al. (2012) form.
        A = A.copy()
        B = B.copy()
        np.fill_diagonal(A, 0.0)
        np.fill_diagonal(B, 0.0)
        a_sum = A.sum()
        b_sum = B.sum()
        ab = float(np.sum(A * B))
        row_dot = float(np.dot(A.sum(axis=1), B.sum(axis=1)))
        term = (
            ab
            + a_sum * b_sum / ((n - 1) * (n - 2))
            - 2.0 / (n - 2) * row_dot
        )
        return term / (n * (n - 3))

    hsic_kl = _hsic1(K, L)
    hsic_kk = _hsic1(K, K)
    hsic_ll = _hsic1(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        raise RuntimeError(
            "Near-zero norm in debiased HSIC: representations are constant or nearly so."
        )
    return float(hsic_kl / denom)


def procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal-Procrustes disparity between two equal-shape representations.

    Column-centers and unit-Frobenius-normalizes each set, then aligns Y to X by
    the optimal rotation. The disparity 1 - (sum of singular values of Y0^T X0)^2
    is rotation-invariant: 0 when Y is an orthogonal transform of X, larger as the
    relational geometry diverges. A high-variance/outlier-driven CKA divergence
    should also show up here (cross-check for E-G-02).

    Args:
        X: activations (n, d).
        Y: activations (n, d). Same shape as X.
    Returns:
        Procrustes disparity (>= 0).
    Raises:
        ValueError: if X and Y do not share the same shape.
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must share shape. Got {X.shape} vs {Y.shape}.")

    X0 = X - X.mean(axis=0, keepdims=True)
    Y0 = Y - Y.mean(axis=0, keepdims=True)

    x_norm = np.linalg.norm(X0)
    y_norm = np.linalg.norm(Y0)
    if x_norm < 1e-10 or y_norm < 1e-10:
        raise RuntimeError("Near-zero norm: a representation is constant after centering.")
    X0 = X0 / x_norm
    Y0 = Y0 / y_norm

    M = Y0.T @ X0
    s = np.linalg.svd(M, compute_uv=False)
    disparity = 1.0 - float(s.sum()) ** 2
    return float(disparity)


def leave_k_out_influence(
    H_math: np.ndarray,
    H_generic: np.ndarray,
    k: int,
    n_iter: int,
    base_seed: int,
) -> dict[str, float]:
    """Sensitivity of inter-category CKA to dropping k samples per side.

    A divergence driven by a handful of high-variance outliers (the failure mode
    Davari et al. 2022 / Cloos et al. 2024 warn about) shows large influence; a
    content-driven divergence is stable under leave-k-out. Seeds come only from
    get_seed (project seed discipline).

    Args:
        H_math: math-stimulus activations (n1, d).
        H_generic: generic-text activations (n2, d).
        k: rows dropped per side each iteration.
        n_iter: number of leave-k-out resamples.
        base_seed: project base seed; per-iteration rng via get_seed(base_seed, "loo", i).
    Returns:
        {"base_cka", "max_abs_influence", "mean_abs_influence"}.
    """
    from src.probing.seeds import get_seed

    base = compute_cka_intercategory(H_math, H_generic, seed=base_seed)
    n1, n2 = H_math.shape[0], H_generic.shape[0]

    deltas: list[float] = []
    for i in range(n_iter):
        rng = np.random.default_rng(get_seed(base_seed, "loo", i))
        keep_math = rng.choice(n1, size=max(n1 - k, 1), replace=False)
        keep_generic = rng.choice(n2, size=max(n2 - k, 1), replace=False)
        cka_loo = compute_cka_intercategory(
            H_math[keep_math], H_generic[keep_generic], seed=base_seed
        )
        deltas.append(abs(cka_loo - base))

    return {
        "base_cka": float(base),
        "max_abs_influence": float(max(deltas)) if deltas else 0.0,
        "mean_abs_influence": float(np.mean(deltas)) if deltas else 0.0,
    }


# ── Result persistence ───────────────────────────────────────────────────────

def save_cka_results(
    cka_matrix: np.ndarray,
    output_dir: Path,
    filename_stem: str = "cka_matrix",
) -> None:
    """Save a CKA matrix atomically as .npy (programmatic) and .csv (readable).

    Args:
        cka_matrix: CKA matrix (L, L).
        output_dir: destination directory (created if absent).
        filename_stem: output filename prefix.
    """
    import os
    import tempfile

    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / f"{filename_stem}.npy"
    _atomic_save_npy(npy_path, cka_matrix)
    logger.info("Saved: %s", npy_path)

    # Atomic CSV write (matrix layout: columns = layer_00, layer_01, ...).
    csv_path = output_dir / f"{filename_stem}.csv"
    n_layers = cka_matrix.shape[0]
    header = ",".join([f"layer_{l:02d}" for l in range(n_layers)])
    fd, tmp = tempfile.mkstemp(dir=output_dir, suffix=".csv")
    os.close(fd)
    try:
        np.savetxt(tmp, cka_matrix, delimiter=",", header=header, comments="")
        os.replace(tmp, csv_path)
    except Exception:
        os.remove(tmp)
        raise
    logger.info("Saved: %s", csv_path)


__all__ = [
    "linear_cka",
    "cka_matrix_across_layers",
    "center_gram",
    "compute_cka_matrix_intramodel",
    "compute_cka_intercategory",
    "compute_cka_intercategory_all_layers",
    "compute_cka_cross_temporal",
    "compute_cka_drift",
    "debiased_linear_cka",
    "procrustes_distance",
    "leave_k_out_influence",
    "save_cka_results",
]


# ── Entry point / usage example ──────────────────────────────────────────────

if __name__ == "__main__":
    """Demo of the three CKA modes on the project's on-disk hidden states.

    Expects data/processed/<model>/layer_XX.pt plus metadata.json (with a
    "categories" list parallel to the stimuli). The real RQ1 pipeline lives in
    run_rq1.py; this block is a standalone sanity demo.
    """
    import json

    from src.config.categories import MATH_CATS, CTRL_CATS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    MODEL_NAME  = "pythia-1.4b"
    N_LAYERS    = 24
    N_SUB       = 512
    SEED        = 42
    DEVICE      = "cpu"

    BASE_DIR    = Path("data/processed") / MODEL_NAME
    RESULTS_DIR = Path("results")

    # Mode 1: intra-model CKA (L×L matrix).
    logger.info("=" * 60)
    logger.info("MODE 1 — intra-model CKA (L×L matrix)")
    logger.info("=" * 60)

    cka_matrix = compute_cka_matrix_intramodel(
        hidden_states_dir=BASE_DIR,
        n_layers=N_LAYERS,
        n_sub=N_SUB,
        seed=SEED,
        device=DEVICE,
    )

    save_cka_results(
        cka_matrix=cka_matrix,
        output_dir=RESULTS_DIR,
        filename_stem="cka_matrix_intramodel",
    )

    diag_mean = np.diag(cka_matrix).mean()
    assert abs(diag_mean - 1.0) < 1e-6, f"Sanity check failed: diag mean = {diag_mean}"
    logger.info("Sanity check OK: mean diagonal = %.6f", diag_mean)

    # Mode 2: inter-category CKA per layer (RQ1 curve).
    logger.info("=" * 60)
    logger.info("MODE 2 — inter-category CKA (per-layer curve, RQ1)")
    logger.info("=" * 60)

    # metadata["categories"] is a list parallel to stimuli_ids, with values
    # "CAT-SIGN" | "CAT-PARITY" | "CTRL-NEU" | "CTRL-NUM".
    with open(BASE_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    categories = np.array(metadata["categories"])

    math_mask    = np.isin(categories, list(MATH_CATS))
    generic_mask = np.isin(categories, list(CTRL_CATS))

    math_indices    = np.where(math_mask)[0]
    generic_indices = np.where(generic_mask)[0]

    logger.info("  Math stimuli (CAT-SIGN + CAT-PARITY): %d", len(math_indices))
    logger.info("  Control stimuli (CTRL-NEU + CTRL-NUM): %d", len(generic_indices))

    cka_inter = compute_cka_intercategory_all_layers(
        hidden_states_dir=BASE_DIR,
        n_layers=N_LAYERS,
        math_indices=math_indices,
        generic_indices=generic_indices,
        device=DEVICE,
    )

    _atomic_save_npy(RESULTS_DIR / "cka_intercategory.npy", cka_inter)
    logger.info("Inter-category CKA per layer:")
    for l, val in enumerate(cka_inter):
        logger.info("  Layer %02d: %.4f", l, val)

    # Mode 3: cross-temporal CKA (RQ3 curve).
    logger.info("=" * 60)
    logger.info("MODE 3 — cross-temporal CKA (base vs QLoRA checkpoints, RQ3)")
    logger.info("=" * 60)

    CKPT_BASE = Path("data/processed/checkpoints")
    checkpoint_dirs = {
        "checkpoint-2500":  CKPT_BASE / "checkpoint-2500",
        "checkpoint-5000":  CKPT_BASE / "checkpoint-5000",
        "checkpoint-7500":  CKPT_BASE / "checkpoint-7500",
        "checkpoint-10000": CKPT_BASE / "checkpoint-10000",
    }
    checkpoint_dirs = {k: v for k, v in checkpoint_dirs.items() if v.exists()}

    if checkpoint_dirs:
        cka_temporal = compute_cka_cross_temporal(
            base_hidden_states_dir=BASE_DIR,
            checkpoint_hidden_states_dirs=checkpoint_dirs,
            n_layers=N_LAYERS,
            n_sub=N_SUB,
            seed=SEED,
            device=DEVICE,
        )

        cka_drift = compute_cka_drift(cka_temporal)

        drift_matrix = np.stack(list(cka_drift.values()))  # (n_ckpt, n_layers)
        _atomic_save_npy(RESULTS_DIR / "cka_drift_temporal.npy", drift_matrix)

        logger.info("CKA drift (per checkpoint, layer of maximum drift):")
        for ckpt_name, drift in cka_drift.items():
            l_max = np.argmax(drift)
            logger.info("  %s: max drift at layer %02d (drift = %.4f)",
                        ckpt_name, l_max, drift[l_max])
    else:
        logger.info("  No checkpoints found on disk — skipping Mode 3.")
