# seeds.py — deterministic seed derivation for every stochastic operation.
# All seeds are derived from a single global base; never call random.seed() elsewhere.

SEED_OFFSETS: dict[str, int] = {
    "undersampling":          1_000,
    "train_test_split":       2_000,
    "bootstrap":              3_000,
    "permutation":           10_000,
    "global_drift_sampling": 50_000,
}


def get_seed(base_seed: int, operation: str, index: int = 0) -> int:
    """Return base_seed + offset[operation] + index.

    Args:
        base_seed: global seed from config.
        operation: key in SEED_OFFSETS.
        index:     per-call modifier (e.g. layer_idx, permutation_i).
    """
    if operation not in SEED_OFFSETS:
        raise ValueError(f"Unknown operation {operation!r}. "
                         f"Valid keys: {list(SEED_OFFSETS)}")
    return base_seed + SEED_OFFSETS[operation] + index