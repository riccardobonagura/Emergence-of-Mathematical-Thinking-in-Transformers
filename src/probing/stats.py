"""Calcolo rigoroso degli intervalli di confidenza e test di permutazione."""

import numpy as np
from sklearn.base import clone
from joblib import Parallel, delayed
from typing import Tuple

from .seeds import get_seed

def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, 
    n_samples: int, ci: float, base_seed: int
) -> Tuple[float, float]:
    """Bootstrap non parametrico per l'intervallo di confidenza."""
    rng = np.random.default_rng(base_seed)
    n_test = len(y_true)
    accuracies = np.empty(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        idx = rng.choice(n_test, size=n_test, replace=True)
        accuracies[i] = np.mean(y_true[idx] == y_pred[idx])

    lower = np.percentile(accuracies, (1 - ci) / 2 * 100)
    upper = np.percentile(accuracies, (1 + ci) / 2 * 100)
    return float(lower), float(upper)

def _single_permutation(clf, X_train, y_train, X_test, y_test, seed_perm):
    """Job atomico per singola permutazione."""
    rng = np.random.default_rng(seed_perm)
    y_perm = rng.permutation(y_train)
    clf_clone = clone(clf)
    clf_clone.fit(X_train, y_perm)
    return clf_clone.score(X_test, y_test)

def permutation_test_parallel(
    clf, X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray, 
    actual_accuracy: float, n_permutations: int, 
    base_seed: int, n_jobs: int = -1
) -> Tuple[float, float]:
    """Esegue n_permutations addestramenti parallelizzati su cloni dell'estimatore."""
    seeds = [get_seed(base_seed, "permutation", i) for i in range(n_permutations)]
    
    null_accuracies = Parallel(n_jobs=n_jobs)(
        delayed(_single_permutation)(clf, X_train, y_train, X_test, y_test, s)
        for s in seeds
    )
    
    null_accuracies = np.array(null_accuracies)
    baseline_mean = np.mean(null_accuracies)
    p_value = np.sum(null_accuracies >= actual_accuracy) / n_permutations
    
    return float(baseline_mean), float(p_value)