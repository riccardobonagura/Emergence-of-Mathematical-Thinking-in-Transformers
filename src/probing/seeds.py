"""
seeds.py — Deterministic seed derivation. Single Source of Truth for all RNG.
Never use np.random.seed(42), torch.manual_seed, or default_rng(42) directly (E-O-04).
Always call get_seed() from this module.
"""
import hashlib

def get_seed(base_seed: int, purpose: str, offset: int = 0) -> int:
    """
    Derives a deterministic, collision-resistant seed for a specific use case.

    Args:
        base_seed: global seed from config["seed"]
        purpose:   string identifying the RNG use case (e.g. "bootstrap", "undersampling")
        offset:    integer differentiator (e.g. layer_idx, split_idx)

    Returns:
        int seed in [0, 2**31)

    Example:
        get_seed(42, "bootstrap", 5)  -> reproducible seed for bootstrap at layer 5
    """
    key = f"{purpose}_{offset}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16) % (2**31)
    return (base_seed + h) % (2**31)
