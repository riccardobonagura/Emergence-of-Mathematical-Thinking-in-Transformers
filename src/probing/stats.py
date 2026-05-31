"""
stats.py — bootstrap intervals, permutation testing, and multiple-comparison correction.
Seeds are passed in explicitly (no default seed parameters).

Notes:
- Permutation testing uses internal cross-validation to avoid test-set leakage.
- permutation_test_score runs with n_jobs=1 to avoid nested-parallelism deadlocks
  under joblib's loky backend when called from run_rq2.py.
"""

import numpy as np
from sklearn.model_selection import permutation_test_score
from typing import List, Tuple, Any


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int,
    ci: float,
    seed: int
) -> Tuple[float, float]:
    """
    Computes a bootstrap confidence interval for linear probing accuracy scores.
    The seed argument has no default value to prevent hidden random state drifts.

    Args:
        y_true: Ground truth target labels array.
        y_pred: Predicted labels emitted by the hyperplane classifier.
        n_samples: Number of bootstrap resamples to execute (corresponds to bootstrap_n_samples).
        ci: Confidence interval threshold (e.g., 0.95 for 95% CI).
        seed: Mandatory deterministic seed for random generator instantiation.

    Returns:
        A tuple containing (lower_bound, upper_bound) percentiles of the accuracy distribution.
    """
    rng = np.random.default_rng(seed)
    n_total = len(y_true)
    if n_total == 0:
        return 0.0, 0.0

    boot_accuracies = []
    # Vectorized element-wise accuracy correctness evaluation matrix
    correct = (y_true == y_pred).astype(np.float64)

    for _ in range(n_samples):
        # Sample sample-indices with replacement using a single index array
        # to guarantee synchronization between true and predicted arrays
        boot_idx = rng.integers(0, n_total, size=n_total)
        boot_accuracies.append(correct[boot_idx].mean())

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_accuracies, alpha * 100))
    high = float(np.percentile(boot_accuracies, (1.0 - alpha) * 100))
    return low, high


def rigorous_permutation_test(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    n_permutations: int,
    seed: int
) -> float:
    """
    Executes a label-shuffling permutation test to establish an empirical null distribution.
    Runs exclusively on the training split via internal cross-validation to strictly prevent
    test-set data leakage.

    Args:
        estimator: The scikit-learn Pipeline (StandardScaler + LogisticRegression) to evaluate.
        X: Training activations matrix.
        y: True training labels target array.
        cv: Number of cross-validation folds (e.g., 5).
        n_permutations: Total number of iterations to execute for the null hypothesis.
        seed: Mandatory deterministic seed for label permutation random state.

    Returns:
        An empirical p-value representing the fraction of shuffled trials that beat or match
        true unpermuted model accuracy.
    """
    # CRITICAL FIX: n_jobs is strictly locked to 1. Since this function is dispatched
    # in parallel by run_rq2.py across layers, nested parallelism (n_jobs=-1) would
    # choke the loky backend and induce permanent process deadlocks.
    _, _, pvalue = permutation_test_score(
        estimator=estimator,
        X=X,
        y=y,
        scoring="accuracy",
        cv=cv,
        n_permutations=n_permutations,
        random_state=seed,
        n_jobs=1
    )
    return float(pvalue)


def benjamini_hochberg_correction(p_values: List[float], fdr_level: float = 0.05) -> List[bool]:
    """
    Applies the Benjamini-Hochberg (FDR) step-down multiple comparison correction procedure.

    Args:
        p_values: Collection of raw uncorrected p-values calculated across properties/layers.
        fdr_level: Target false discovery rate threshold boundary (Default 5%).

    Returns:
        A list of boolean markers indicating whether each corresponding hypothesis remains significant.
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    significant = np.zeros(n_tests, dtype=bool)
    max_sig_idx = -1

    # Step-down verification scan loop
    for i in range(n_tests):
        threshold = (i + 1) / n_tests * fdr_level
        if sorted_p[i] <= threshold:
            max_sig_idx = i

    if max_sig_idx >= 0:
        significant[sorted_indices[:max_sig_idx + 1]] = True

    return significant.tolist()
