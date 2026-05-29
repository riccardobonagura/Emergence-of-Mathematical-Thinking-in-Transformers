"""
engine.py — training, evaluation, and CI computation for one (layer, property) cell.
Stateless: all inputs arrive as numpy arrays; all outputs are strictly typed dictionaries.
"""

import numpy as np
from typing import Any, Dict, Optional, TypedDict

from .pipeline   import build_pipeline, denormalize_classifier
from .stats      import bootstrap_ci, rigorous_permutation_test
from .directions import test_confound_correlation
from .seeds      import get_seed


class LayerResult(TypedDict):
    """Explicit contract for the results emitted by a single probing layer (ARCH-03)."""
    layer: int
    property: str
    accuracy: float
    accuracy_lower_ci: float
    accuracy_upper_ci: float
    raw_p_value: float
    confound_pearson_r: Optional[float]
    confound_p_value: Optional[float]
    weights: np.ndarray
    bias: np.ndarray


class ProbingEngine:
    """Fits one probe per (layer, property) and enforces statistical rigor."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config

    def run_layer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        layer_idx: int,
        prop_name: str,
        magnitudes_test: Optional[np.ndarray] = None,
    ) -> LayerResult:
        """Fit → evaluate → denormalize → CI → Permutation → Confound test."""

        # 1. Pipeline fit
        pipe = build_pipeline(
            max_iter             = self.cfg["max_iter"],
            C                    = self.cfg["C"],
            solver               = self.cfg["solver"],
            multiclass_strategy  = self.cfg["multiclass_strategy"],
        )
        pipe.fit(X_train, y_train)

        # 2. Scoring
        accuracy = pipe.score(X_test, y_test)
        y_pred   = pipe.predict(X_test)

        # 3. Project weights back to the original (unscaled) activation space
        w_orig, b_orig = denormalize_classifier(pipe)

        # 4. Bootstrap Confidence Intervals
        # Layer-specific seed keeps bootstrap CIs independent across layers.
        lo, hi = bootstrap_ci(
            y_true    = y_test,
            y_pred    = y_pred,
            n_samples = self.cfg["bootstrap_n_samples"],
            ci        = self.cfg.get("bootstrap_ci", 0.95),
            seed      = get_seed(self.cfg["seed"], "bootstrap", layer_idx),
        )

        # 5. Rigorous Permutation Test (P1-4)
        # Executed on the training split via internal cross-validation
        # to strictly prevent test-set data leakage during significance testing.
        raw_p_value = rigorous_permutation_test(
            estimator      = pipe,
            X              = X_train,
            y              = y_train,
            cv             = 5,
            n_permutations = self.cfg.get("n_permutation_tests", 1000),
            seed           = get_seed(self.cfg["seed"], "permutation", layer_idx)
        )

        # 6. Confound N-01 Correlation Test (P0-2)
        # Correlates the probe's PREDICTIONS with the operand magnitude.
        # If Pearson's r is high and significant, the probe memorized magnitude, not abstract sign.
        r_stat, p_confound = None, None
        if magnitudes_test is not None:
            r_stat, p_confound = test_confound_correlation(magnitudes_test, y_pred)

        return {
            "layer":               layer_idx,
            "property":            prop_name,
            "accuracy":            round(float(accuracy), 4),
            "accuracy_lower_ci":   round(float(lo), 4),
            "accuracy_upper_ci":   round(float(hi), 4),
            "raw_p_value":         round(float(raw_p_value), 5),
            "confound_pearson_r":  round(float(r_stat), 4) if r_stat is not None else None,
            "confound_p_value":    round(float(p_confound), 5) if p_confound is not None else None,
            "weights":             w_orig,
            "bias":                b_orig,
        }
