"""
stats.py — Statistical Evaluation Module.
Contains robust statistical tests for probing classifiers, enforcing strict
random seed discipline and correct mathematical invariants.
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 1000, ci: float = 0.95,
 seed: int = 42) -> tuple[float, float]:
    """
    Calculates the bootstrap confidence interval for accuracy.
    Strictly respects the invariant: a single shared index array is sampled per iteration 
    to maintain alignment between y_true and y_pred.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    
    for _ in range(n_samples):
        # CRITICAL INVARIANT: Single shared index sampled once per iteration
        # Do not call rng.integers twice, otherwise y_true and y_pred will decouple.
        idx = rng.integers(0, n, size=n)
        
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        
        score = accuracy_score(y_true_boot, y_pred_boot)
        scores.append(score)
        
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100))
    
    return lower, upper


def permutation_test(estimator, X: np.ndarray, y: np.ndarray, cv, n_permutations: int = 100, seed: int = 42) -> float:
    """
    Calculates the p-value of the estimator's accuracy using a permutation test.
    Specifically designed for binary/multiclass classification tasks (not regression).
    """
    # Enforce strict random state routing to guarantee reproducibility
    _, _, pvalue = permutation_test_score(
        estimator, 
        X, 
        y, 
        scoring="accuracy", 
        cv=cv, 
        n_permutations=n_permutations, 
        n_jobs=-1,
        random_state=seed
    )
    
    return float(pvalue)