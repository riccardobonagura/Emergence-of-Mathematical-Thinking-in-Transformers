"""
Logic Layer: Motore di addestramento e validazione statistica.
"""
import numpy as np
from .pipeline import build_pipeline, denormalize_classifier
from .metrics import bootstrap_ci
from .seeds import get_seed

class ProbingEngine:
    def __init__(self, config: dict):
        self.config = config

    def run_layer(self, X_train, y_train, X_test, y_test, layer_idx, prop_name):
        """Esegue il fit della pipeline e calcola le metriche di robustezza."""
        
        # 1. Costruzione e Training della pipeline
        pipe = build_pipeline(
            max_iter=self.config["max_iter"],
            C=self.config["C"],
            solver=self.config["solver"],
            multiclass_strategy=self.config["multiclass_strategy"]
        )
        pipe.fit(X_train, y_train)
        
        # 2. Valutazione
        accuracy = pipe.score(X_test, y_test)
        y_pred = pipe.predict(X_test)
        
        # 3. Denormalizzazione pesi (Algebra lineare per estrarre le direzioni reali)
        w_orig, b_orig = denormalize_classifier(pipe)
        
        # 4. Calcolo Intervalli di Confidenza (Bootstrap)
        # FIX: Passiamo esplicitamente 'ci' e 'base_seed' richiesti dai metadati
        low, up = bootstrap_ci(
            y_test, 
            y_pred, 
            n_samples=self.config["bootstrap_n_samples"],
            ci=self.config.get("bootstrap_ci", 0.95),
            base_seed=get_seed(self.config["seed"], "bootstrap", layer_idx)
        )
        
        return {
            "layer": layer_idx,
            "property": prop_name,
            "accuracy": float(np.round(accuracy, 4)),
            "accuracy_lower_ci": float(np.round(low, 4)),
            "accuracy_upper_ci": float(np.round(up, 4)),
            "weights": w_orig,
            "bias": b_orig
        }