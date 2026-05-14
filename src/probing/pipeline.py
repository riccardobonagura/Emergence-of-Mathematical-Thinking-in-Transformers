"""
Costruzione dei classificatori lineari e algebra di denormalizzazione.
Assicura che build_pipeline e denormalize_classifier siano esportati correttamente.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Tuple

def build_pipeline(max_iter: int, C: float, solver: str, multiclass_strategy: str) -> Pipeline:
    """
    Istanzia la pipeline logistica con standardizzazione obbligatoria.
    
    Args:
        max_iter: Numero massimo di iterazioni per il solver.
        C: Inverso della forza di regolarizzazione L2.
        solver: Algoritmo di ottimizzazione (default: 'lbfgs').
        multiclass_strategy: Strategia per classi > 2 (es. 'ovr').
    
    Returns:
        Pipeline fittabile di scikit-learn.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=max_iter, 
            C=C, 
            solver=solver, 
            multi_class=multiclass_strategy
        ))
    ])

def denormalize_classifier(pipe: Pipeline) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae pesi e bias denormalizzandoli nello spazio degli hidden state originali.
    Matematica: w_orig = w_norm / sigma; b_orig = b_norm - (w_orig * mu)
    
    Args:
        pipe: Pipeline fittata con StandardScaler e LogisticRegression.
    Returns:
        Tupla (weights_original, bias_original) in float64.
    """
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    # I pesi in sklearn hanno shape (n_classes, d) o (1, d)
    w_norm = clf.coef_          
    b_norm = clf.intercept_     
    
    mean = scaler.mean_
    scale = scaler.scale_

    # Flag per gestire il caso binario (dove coef_ è 1D o 2D con una riga)
    was_1d = (w_norm.shape[0] == 1)

    # 1. Denormalizzazione pesi: w_orig = w_norm / sigma
    w_orig = w_norm / scale
    
    # 2. Denormalizzazione bias: b_orig = b_norm - (w_orig DOT mu)
    # Usiamo np.dot per gestire correttamente il caso multiclasse (matrice x vettore)
    correction = np.dot(w_orig, mean)
    b_orig = b_norm - correction

    # Se binario, riportiamo a shape piatta (d,) e scalare
    if was_1d:
        w_orig = w_orig.reshape(-1)
        b_orig = b_orig[0] if b_orig.size == 1 else b_orig

    return w_orig.astype(np.float64), b_orig.astype(np.float64)