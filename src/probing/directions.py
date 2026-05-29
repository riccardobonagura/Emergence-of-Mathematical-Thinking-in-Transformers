"""
directions.py — Directional analysis, angular distances, selectivity, and confound correlations.
Enforces precise geometric metrics and explicit raw selectivity gap formulations.

CORRECTED:
- Fixed type annotations for angle_degrees using typing.Optional and typing.Union to ensure
  strict compliance with the project layout and execution guidelines.
"""

import numpy as np
import scipy.stats as stats
from typing import Optional, Union


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two flat vectors.

    Args:
        v1: First vector array.
        v2: Second vector array.

    Returns:
        The scalar cosine similarity value in [-1.0, 1.0].
    """
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    norm1 = np.linalg.norm(v1_flat)
    norm2 = np.linalg.norm(v2_flat)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return float(np.dot(v1_flat, v2_flat) / (norm1 * norm2))


def angle_degrees(arg1: Union[float, np.ndarray], arg2: Optional[np.ndarray] = None) -> float:
    """
    Computes the geometric angle in degrees.
    Can accept either a pre-computed cosine similarity scalar or two separate directional vectors.

    Usage:
        angle_degrees(cos_sim) -> float
        angle_degrees(v1, v2) -> float

    Args:
        arg1: Scalar cosine similarity or the first input vector array.
        arg2: Second input vector array (optional, required if arg1 is an array).

    Returns:
        The calculated angle in degrees within [0.0, 180.0].
    """
    if arg2 is None:
        # Caller passed a single pre-computed cosine similarity scalar
        cos_sim = float(arg1)
    else:
        # Caller passed two separate directional vectors
        cos_sim = cosine_similarity(arg1, arg2)

    # Clip to avoid numerical floating-point inaccuracies outside boundaries [-1, 1]
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sim)))


def compute_selectivity(accuracy_task: float, accuracy_control: float) -> float:
    """
    Computes the Hewitt & Liang (2019) selectivity gap metric.
    Returns the raw difference between task accuracy and linguistic control accuracy.
    Callers are responsible for interpreting negative selectivity values.

    Args:
        accuracy_task: Probing accuracy achieved on the targeted mathematical task.
        accuracy_control: Probing accuracy achieved on the arbitrary language control task.

    Returns:
        The raw floating-point difference: accuracy_task - accuracy_control.
    """
    return float(accuracy_task - accuracy_control)


def test_confound_correlation(magnitudes: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    """
    Computes the Pearson correlation coefficient and associated p-value between
    a continuous confound tracking variable and probe predictions.

    Args:
        magnitudes: Array of absolute baseline values (e.g., operand magnitude or delta).
        predictions: Array of continuous logits or classification predictions from the probe.

    Returns:
        A tuple containing (pearson_r_coefficient, asymptotic_p_value).
    """
    # Defensive checks to isolate invariant vectors or tiny data samples
    if len(magnitudes) < 2 or np.var(magnitudes) == 0.0 or np.var(predictions) == 0.0:
        return 0.0, 1.0

    r_stat, p_val = stats.pearsonr(magnitudes, predictions)
    return float(r_stat), float(p_val)
