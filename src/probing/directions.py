# directions.py — angular relationships between probe weight vectors.
# Used to compare sign-direction vs parity-direction across layers.

import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two flattened vectors; returns 0.0 on zero-norm."""
    v1, v2 = np.asarray(v1).ravel(), np.asarray(v2).ravel()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1 / n1, v2 / n2))


def angle_degrees(cos_sim: float) -> float:
    """Convert cosine similarity to angle in degrees [0, 180]."""
    return float(np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180.0 / np.pi)