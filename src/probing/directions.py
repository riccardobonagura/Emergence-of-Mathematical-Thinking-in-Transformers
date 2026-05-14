"""Calcolo delle relazioni angolari tra direzioni semantiche."""

import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Prodotto scalare su vettori normalizzati L2."""
    # Appiattiamo i vettori per garantire che np.dot restituisca uno scalare
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()
    
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1 / n1, v2 / n2))

def angle_degrees(cos_sim: float) -> float:
    """Converte similitudine coseno in gradi [0, 180]."""
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.arccos(cos_sim) * 180.0 / np.pi)