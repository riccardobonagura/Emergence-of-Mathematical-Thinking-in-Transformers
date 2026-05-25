# engine.py — training, evaluation, and CI computation for one (layer, property) cell.
# Stateless: all inputs arrive as numpy arrays; all outputs are plain Python scalars + arrays.

import numpy as np

from .pipeline import build_pipeline, denormalize_classifier
from .stats    import bootstrap_ci
from .seeds    import get_seed


class ProbingEngine:
    """Fits one probe per (layer, property) and returns a result dict."""

    def __init__(self, config: dict) -> None:
        self.cfg = config

    def run_layer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        layer_idx: int,
        prop_name: str,
    ) -> dict:
        """Fit → evaluate → denormalise → CI.  Returns a flat result dict."""

        pipe = build_pipeline(
            max_iter             = self.cfg["max_iter"],
            C                    = self.cfg["C"],
            solver               = self.cfg["solver"],
            multiclass_strategy  = self.cfg["multiclass_strategy"],
        )
        pipe.fit(X_train, y_train)

        accuracy = pipe.score(X_test, y_test)
        y_pred   = pipe.predict(X_test)

        # Project weights back to the original (unscaled) activation space.
        w_orig, b_orig = denormalize_classifier(pipe)

        # Layer-specific seed keeps bootstrap CIs independent across layers.
        lo, hi = bootstrap_ci(
            y_true    = y_test,
            y_pred    = y_pred,
            n_samples = self.cfg["bootstrap_n_samples"],
            ci        = self.cfg.get("bootstrap_ci", 0.95),
            seed = get_seed(self.cfg["seed"], "bootstrap", layer_idx),
        )

        return {
            "layer":               layer_idx,
            "property":            prop_name,
            "accuracy":            round(float(accuracy), 4),
            "accuracy_lower_ci":   round(float(lo), 4),
            "accuracy_upper_ci":   round(float(hi), 4),
            "weights":             w_orig,
            "bias":                b_orig,
        }