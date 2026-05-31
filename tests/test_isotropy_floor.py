"""Unit test for the random-Gaussian norm-matched isotropy floor (E-G-01)."""

import numpy as np

from src.metrics.isotropy import random_gaussian_isotropy_floor


def test_floor_near_zero_and_ci_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    H_ref = rng.standard_normal((60, 64)).astype(np.float32)
    floor_mean, ci_low, ci_high = random_gaussian_isotropy_floor(
        H_ref, n_bootstrap=30, base_seed=42
    )
    # For d >> 1 the mean-cosine floor sits near 0.
    assert abs(floor_mean) < 0.05
    assert ci_low <= floor_mean <= ci_high
