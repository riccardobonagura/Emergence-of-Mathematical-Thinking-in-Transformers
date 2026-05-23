# pipeline.py — scikit-learn pipeline construction and weight denormalisation.
# StandardScaler is mandatory: probing accuracy is sensitive to feature scale.

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Tuple


def build_pipeline(
    max_iter: int,
    C: float,
    solver: str,
    multiclass_strategy: str,   # kept for API compatibility; ignored for binary tasks
) -> Pipeline:
    """Return an unfitted (StandardScaler → LogisticRegression) pipeline.

    v5 note: sign and parity are binary tasks; multi_class is not forwarded
    to LogisticRegression to avoid the sklearn >= 1.5 deprecation warning.
    multiclass_strategy is accepted but unused — downstream callers need not change.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver=solver,
            # multi_class omitted: deprecated in sklearn >= 1.5, irrelevant for binary
        )),
    ])


def denormalize_classifier(pipe: Pipeline) -> Tuple[np.ndarray, np.ndarray]:
    """Extract weights and bias in the original (unscaled) hidden-state space.

    Algebra:
        w_orig = w_norm / σ
        b_orig = b_norm − w_orig · μ
    """
    scaler = pipe.named_steps["scaler"]
    clf    = pipe.named_steps["clf"]

    w_norm = clf.coef_       # (1, d) binary  |  (n_classes, d) multiclass
    b_norm = clf.intercept_

    # Track binary case before any reshape to restore shape at the end.
    is_binary = w_norm.shape[0] == 1

    w_orig = w_norm / scaler.scale_                 # broadcast over rows
    b_orig = b_norm - np.dot(w_orig, scaler.mean_)  # (n_classes,)

    if is_binary:
        w_orig = w_orig.ravel()          # (d,)
        b_orig = float(b_orig[0])        # scalar

    return w_orig.astype(np.float64), np.asarray(b_orig, dtype=np.float64)