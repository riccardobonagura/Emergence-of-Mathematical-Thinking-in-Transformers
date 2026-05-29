"""
isotropy.py
===========
Phase 3 of the pipeline: Isotropy Analysis (Mean Cosine Similarity).

Input:
    - data/processed/{model_name}/layer_XX.pt  : tensor (N, d) for each layer
    - data/processed/{model_name}/metadata.json : ordered list of stimulus IDs
    - data/stimuli/stimuli.jsonl                : dataset containing the "category" field

Output:
    - results/isotropy.csv
      Columns: layer, category, n_stimuli, estimator, iso_mean, iso_spread,
               ci_low, ci_high
      - iso_mean    : exact or estimated mean of cosine similarities
      - iso_spread  : std of similarities (distribution spread)
      - ci_low/high : 95% CI (NaN for exact estimator, bootstrap for MC)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
import warnings
from pathlib import Path
from typing import NamedTuple, Callable, Optional

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result structure for a single (layer, category) pair
# ---------------------------------------------------------------------------
class IsotropyResult(NamedTuple):
    layer: int
    category: str
    n_stimuli: int
    estimator: str          # "exact" | "monte_carlo"
    iso_mean: float
    iso_spread: float       # std of cosine similarities
    ci_low: float           # NaN if exact estimator
    ci_high: float          # NaN if exact estimator


# ---------------------------------------------------------------------------
# Global seed — fixed exactly once at the beginning, never inside functions
# ---------------------------------------------------------------------------
def make_rng(seed: int) -> np.random.Generator:
    """Creates a NumPy generator with an explicit seed to pass into functions."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Exact estimator (Full Gram matrix)
# ---------------------------------------------------------------------------
def isotropy_exact(
    H_cat: torch.Tensor,
    n_bootstrap: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float, float]:
    """
    Calculates the exact mean and std of cosine similarities across all pairs
    (i, j) where i ≠ j using the normalized Gram matrix.

    The point estimate has zero sampling variance over pairings, but for small
    N the *stimuli themselves* are a random draw — n_bootstrap > 0 quantifies
    that stimulus-level uncertainty (E-M-02: CI required to interpret ΔIso).

    Args:
        H_cat:       Tensor (N_cat, d) of hidden states for a given category.
        n_bootstrap: Number of stimulus-level bootstrap resamples for the 95% CI.
                     0 → no bootstrap, CI returned as NaN (backward-compatible).
        rng:         Required when n_bootstrap > 0. NumPy generator with externally
                     fixed seed (project-wide seed discipline).

    Returns:
        (iso_mean, iso_spread, ci_low, ci_high). CIs are NaN when n_bootstrap == 0.
    """
    if n_bootstrap > 0 and rng is None:
        raise ValueError("rng is required when n_bootstrap > 0.")

    N = H_cat.shape[0]
    assert N >= 2, "At least 2 stimuli are required to compute cosine similarity."

    norms = H_cat.norm(dim=1)
    zero_mask = norms < 1e-8
    if zero_mask.any():
        n_zero = zero_mask.sum().item()
        warnings.warn(
            f"{n_zero} vectors have norm < 1e-8 and will be excluded. "
            "Please check the hidden state extraction."
        )
        H_cat = H_cat[~zero_mask]
        N = H_cat.shape[0]
        if N < 2:
            return float("nan"), float("nan"), float("nan"), float("nan")

    H_norm = F.normalize(H_cat, p=2, dim=1)  # (N, d)
    C = H_norm @ H_norm.T                    # (N, N)

    mask = ~torch.eye(N, dtype=torch.bool, device=C.device)
    off_diag = C[mask]

    iso_mean = float(off_diag.mean().item())
    iso_spread = float(off_diag.std().item())

    ci_low = float("nan")
    ci_high = float("nan")
    if n_bootstrap > 0:
        # Resample stimuli (rows of H_norm), not pairings: the uncertainty we
        # care about is "another draw of N stimuli from the same distribution",
        # not "another draw of pairings from the same N stimuli".
        C_np = C.detach().cpu().numpy()
        boot_means = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.integers(0, N, size=N)
            sub = C_np[np.ix_(idx, idx)]
            # Off-diagonal mean: subtract diag contribution (always 1.0) and divide
            # by N*(N-1). Sum-based form avoids materialising a boolean mask each iter.
            total = sub.sum() - np.trace(sub)
            boot_means[b] = total / (N * (N - 1))
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

    return iso_mean, iso_spread, ci_low, ci_high


# ---------------------------------------------------------------------------
# Monte Carlo estimator (for large N_cat)
# ---------------------------------------------------------------------------
def isotropy_monte_carlo(
    H_cat: torch.Tensor,
    k_pairs: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """
    Estimates mean cosine similarity by sampling k_pairs of random pairs.
    Returns mean, std of similarities, and 95% CI via percentile bootstrap.

    Sampling uses the "shift trick" to guarantee uniform distribution
    across all ordered pairs (i, j) where i ≠ j:
        i ~ Uniform{0,...,N-1}
        j' ~ Uniform{0,...,N-2}
        j  = j' + (j' >= i)
    For every k ≠ i there is exactly one j' mapping to k, so p(j=k|i) = 1/(N-1).

    Args:
        H_cat:       Tensor (N_cat, d) of hidden states.
        k_pairs:     Number of pairs to sample.
        n_bootstrap: Number of bootstrap resamples for the CI.
        rng:         NumPy generator with an externally fixed seed.

    Returns:
        (iso_mean, iso_spread, ci_low, ci_high)
    """
    N = H_cat.shape[0]
    assert N >= 2

    # Warning if K >> available unique pairs
    max_unique_pairs = N * (N - 1)
    if k_pairs > max_unique_pairs:
        warnings.warn(
            f"k_pairs={k_pairs} > available unique pairs={max_unique_pairs} "
            f"(N={N}). Sampling occurs with implicit replacement. "
            "Consider reducing k_pairs or using the exact estimator."
        )

    # Check for zero-norm vectors
    norms = H_cat.norm(dim=1)
    zero_mask = norms < 1e-8
    if zero_mask.any():
        warnings.warn(f"{zero_mask.sum().item()} zero-norm vectors excluded.")
        H_cat = H_cat[~zero_mask]
        N = H_cat.shape[0]

    H_norm = F.normalize(H_cat, p=2, dim=1)  # (N, d)

    # Shift-trick sampling with external rng (shared state across calls)
    idx_i = rng.integers(0, N, size=k_pairs)
    idx_j_raw = rng.integers(0, N - 1, size=k_pairs)
    idx_j = np.where(idx_j_raw >= idx_i, idx_j_raw + 1, idx_j_raw)

    # Vectorized cosine similarity
    u = H_norm[torch.from_numpy(idx_i)]   # (K, d)
    v = H_norm[torch.from_numpy(idx_j)]   # (K, d)
    sims = (u * v).sum(dim=1).numpy()     # (K,)

    iso_mean = float(sims.mean())
    iso_spread = float(sims.std())

    # 95% Percentile Bootstrap CI
    boot_means = np.fromiter(
        (rng.choice(sims, size=k_pairs, replace=True).mean()
         for _ in range(n_bootstrap)),
        dtype=float,
        count=n_bootstrap,
    )
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return iso_mean, iso_spread, ci_low, ci_high


# ---------------------------------------------------------------------------
# Tensor / Metadata alignment validation
# ---------------------------------------------------------------------------
def validate_alignment(
    H_l: torch.Tensor,
    stimuli_ids: list[str],
    stimuli_jsonl_path: Path,
) -> None:
    """
    Verifies that the tensor's row count matches the number of IDs
    in metadata.json and that all IDs exist in the dataset.

    Raises a ValueError if the alignment invariant is violated.
    """
    N_tensor = H_l.shape[0]
    N_meta = len(stimuli_ids)
    if N_tensor != N_meta:
        raise ValueError(
            f"Critical misalignment: tensor has {N_tensor} rows "
            f"but metadata.json contains {N_meta} IDs. "
            "Please re-extract hidden states."
        )

    # Verify that all IDs are present in the dataset
    ids_in_jsonl = set()
    with open(stimuli_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ids_in_jsonl.add(json.loads(line)["id"])

    missing = set(stimuli_ids) - ids_in_jsonl
    if missing:
        raise ValueError(
            f"{len(missing)} IDs in metadata.json were not found in stimuli.jsonl: "
            f"{list(missing)[:5]} ..."
        )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_isotropy_analysis(
    processed_dir: str = "data/processed/pythia-1.4b",
    stimuli_path: str  = "data/processed/dataset_master_v5.jsonl",
    output_path: str   = "results/isotropy.csv",
    n_layers: int      = 24,       # Pythia-1.4B has 24 transformer layers
    exact_threshold: int = 1500,   # v5: max stimuli per category ≈ 1000
                                   # 1000x1000 Gram matrix ≈ 4 MB → use exact
                                   # for all categories (was 600 in v4,
                                   # which triggered MC for CAT-* and exact for CTRL-*)
    k_pairs: int       = 8000,     # Monte Carlo pairs (only if N_cat > threshold)
    n_bootstrap: int   = 2000,     # Bootstrap resamplings (Monte Carlo regime)
    n_bootstrap_exact: int = 1000, # Stimulus-level bootstrap when N_cat ≤ exact_threshold
    seed: int          = 42,
    layer_loader: Optional[Callable[[Path],  np.ndarray]] = None
) -> pd.DataFrame:
    """
    Executes the isotropy analysis across all model layers.

    Adaptive strategy:
        N_cat <= exact_threshold  → exact estimator (Gram matrix)
        N_cat >  exact_threshold  → Monte Carlo + bootstrap CI

    Args:
        processed_dir:    Directory containing layer_XX.pt files and metadata.json.
        stimuli_path:     Path to the dataset JSONL file.
        output_path:      Path to the output CSV.
        n_layers:         Total number of model layers.
        exact_threshold:  Threshold for estimator selection.
        k_pairs:          Pairs to sample (Monte Carlo regime).
        n_bootstrap:      Bootstrap resamplings (Monte Carlo regime).
        seed:             Global seed for reproducibility.
        layer_loader:     Injected dependency for I/O operations (test mockability).

    Returns:
        DataFrame containing the results.
    """
    # Dependency Injection pattern for I/O decoupling and test mockability
    if layer_loader is None:
        from src.probing.io_utils import load_hidden_states
        # isotropy functions require torch.Tensor; io_utils returns np.ndarray
        layer_loader = lambda p: torch.from_numpy(load_hidden_states(p))  # type: ignore[assignment,return-value]

    # Single RNG, created once — the state advances deterministically
    # across all layers and categories without internal resets
    rng = make_rng(seed)

    proc_path = Path(processed_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    stimuli_jsonl = Path(stimuli_path)

    # Load metadata
    with open(proc_path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    stimuli_ids: list[str] = metadata["stimuli_ids"]

    # ID → category mapping
    id_to_cat: dict[str, str] = {}
    with open(stimuli_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            id_to_cat[s["id"]] = s["category"]

    categories = sorted(set(id_to_cat.values()))

    # Row indices per category — built exactly once
    cat_to_indices: dict[str, list[int]] = {c: [] for c in categories}
    for row_idx, sid in enumerate(stimuli_ids):
        if sid not in id_to_cat:
            raise ValueError(
                f"ID '{sid}' in metadata.json not found in stimuli.jsonl."
            )
        cat_to_indices[id_to_cat[sid]].append(row_idx)

    log.info("Found categories: %s", categories)
    for cat, idxs in cat_to_indices.items():
        log.info("  %-20s  N=%d", cat, len(idxs))

    results: list[IsotropyResult] = []

    for l in range(n_layers):
        layer_file = proc_path / f"layer_{l:02d}.pt"
        if not layer_file.exists():
            log.warning("Layer %02d: file not found, skipping.", l)
            continue

        # Replaced direct torch.load with injected layer_loader
        H_l: torch.Tensor = layer_loader(layer_file)  # type: ignore[assignment]

        # Alignment validation (only on the first layer for efficiency,
        # we trust the extraction pipeline for the rest)
        if l == 0:
            validate_alignment(H_l, stimuli_ids, stimuli_jsonl)

        for cat in categories:
            indices = cat_to_indices[cat]
            if len(indices) < 2:
                log.warning("Layer %02d  %-20s: N=%d < 2, skipping.", l, cat, len(indices))
                continue

            H_cat = H_l[indices]  # (N_cat, d)
            N_cat = len(indices)

            if N_cat <= exact_threshold:
                iso_mean, iso_spread, ci_low, ci_high = isotropy_exact(
                    H_cat, n_bootstrap=n_bootstrap_exact, rng=rng
                )
                results.append(IsotropyResult(
                    layer=l,
                    category=cat,
                    n_stimuli=N_cat,
                    estimator="exact",
                    iso_mean=iso_mean,
                    iso_spread=iso_spread,
                    ci_low=ci_low,
                    ci_high=ci_high,
                ))
            else:
                iso_mean, iso_spread, ci_low, ci_high = isotropy_monte_carlo(
                    H_cat, k_pairs=k_pairs, n_bootstrap=n_bootstrap, rng=rng
                )
                results.append(IsotropyResult(
                    layer=l,
                    category=cat,
                    n_stimuli=N_cat,
                    estimator="monte_carlo",
                    iso_mean=iso_mean,
                    iso_spread=iso_spread,
                    ci_low=ci_low,
                    ci_high=ci_high,
                ))

        del H_l  # Explicitly free memory

        if (l + 1) % 8 == 0:
            log.info("Layer %02d/%02d completed.", l + 1, n_layers)

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    log.info("Results saved to: %s", out_file)
    return df


# ---------------------------------------------------------------------------
# Unified Public API (Module level)
# ---------------------------------------------------------------------------
def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Returns the NxN matrix of cosine similarities across all vectors.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D tensor [n, d], received shape={tuple(embeddings.shape)}")

    norms = embeddings.norm(dim=1, keepdim=True)
    valid = (norms.squeeze(-1) >= 1e-8)
    if valid.sum().item() < 2:
        raise ValueError("At least 2 vectors with non-zero norm are required.")

    E = embeddings[valid]
    E = F.normalize(E, p=2, dim=1)
    return E @ E.T


def sample_random_cosine_pairs(
    embeddings: torch.Tensor,
    num_pairs: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Samples `num_pairs` ordered pairs (i, j), where i != j, and returns cos(theta_ij).
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D tensor [n, d], received shape={tuple(embeddings.shape)}")

    norms = embeddings.norm(dim=1)
    valid = norms >= 1e-8
    E = embeddings[valid]
    n = E.shape[0]
    if n < 2:
        raise ValueError("At least 2 vectors with non-zero norm are required.")

    E = F.normalize(E, p=2, dim=1)
    idx_i = torch.randint(0, n, (num_pairs,), generator=generator)
    idx_j_raw = torch.randint(0, n - 1, (num_pairs,), generator=generator)
    idx_j = idx_j_raw + (idx_j_raw >= idx_i).to(idx_j_raw.dtype)

    u = E[idx_i]
    v = E[idx_j]
    return (u * v).sum(dim=1)


def estimate_isotropy(
    embeddings: torch.Tensor,
    *,
    method: str = "auto",
    num_pairs: int = 8000,
    exact_threshold: int = 600,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict:
    """
    Unified API for isotropy estimation.
    """
    if method not in {"auto", "exact", "sampled"}:
        raise ValueError("method must be one of: auto, exact, sampled")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D tensor [n, d], received shape={tuple(embeddings.shape)}")

    n = int(embeddings.shape[0])
    estimator = "exact" if (method == "exact" or (method == "auto" and n <= exact_threshold)) else "sampled"

    if estimator == "exact":
        # n_bootstrap > 0 here triggers stimulus-level bootstrap CI; default 0 keeps
        # the legacy NaN behaviour for callers that opt out of uncertainty quantification.
        if n_bootstrap > 0:
            iso_mean, iso_spread, ci_low, ci_high = isotropy_exact(
                embeddings, n_bootstrap=n_bootstrap, rng=make_rng(seed)
            )
        else:
            iso_mean, iso_spread, ci_low, ci_high = isotropy_exact(embeddings)
        return {
            "estimator": "exact",
            "iso_mean": float(iso_mean),
            "iso_spread": float(iso_spread),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_samples": n,
        }

    rng = make_rng(seed)
    if n_bootstrap > 0:
        iso_mean, iso_spread, ci_low, ci_high = isotropy_monte_carlo(
            embeddings,
            k_pairs=num_pairs,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
    else:
        sims = sample_random_cosine_pairs(
            embeddings=embeddings,
            num_pairs=num_pairs,
            generator=torch.Generator().manual_seed(seed),
        )
        iso_mean = float(sims.mean().item())
        iso_spread = float(sims.std().item())
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "estimator": "sampled",
        "iso_mean": float(iso_mean),
        "iso_spread": float(iso_spread),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_samples": n,
    }


__all__ = [
    "cosine_similarity_matrix",
    "sample_random_cosine_pairs",
    "estimate_isotropy",
    "isotropy_exact",
    "isotropy_monte_carlo",
    "run_isotropy_analysis",
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_isotropy_analysis()