"""Mappatura centralizzata degli offset per la derivazione deterministica dei seed."""

SEED_OFFSETS = {
    "undersampling": 1000,
    "train_test_split": 2000,
    "bootstrap": 3000,
    "permutation": 10000,
    "global_drift_sampling": 50000,
}

def get_seed(base_seed: int, operation: str, index: int = 0) -> int:
    """
    Deriva un seed deterministico sommando l'offset specifico dell'operazione.
    
    Args:
        base_seed: Seed globale dal config.
        operation: Chiave presente in SEED_OFFSETS.
        index: Modificatore addizionale (es. layer_idx o iterazione).
    """
    if operation not in SEED_OFFSETS:
        raise ValueError(f"Operazione '{operation}' non mappata in SEED_OFFSETS.")
    return base_seed + SEED_OFFSETS[operation] + index