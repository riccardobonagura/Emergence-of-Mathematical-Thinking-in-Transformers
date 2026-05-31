"""Unit tests for the RQ1 CKA robustness battery (debiased CKA, Procrustes, leave-k-out)."""

import numpy as np
import pytest

from src.metrics.cka import (
    debiased_linear_cka,
    leave_k_out_influence,
    linear_cka,
    procrustes_distance,
)


def test_debiased_self_cka_is_one() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 16))
    assert debiased_linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)


def test_debiased_close_to_biased_on_correlated_pair() -> None:
    # Large-n correlated pair: debiased and biased CKA should nearly agree.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 32))
    Y = X @ rng.standard_normal((32, 32)) + 0.05 * rng.standard_normal((400, 32))
    assert abs(debiased_linear_cka(X, Y) - linear_cka(X, Y)) < 0.05


def test_debiased_raises_for_small_n() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3, 8))
    with pytest.raises(ValueError):
        debiased_linear_cka(X, X)


def test_procrustes_zero_under_rotation() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 16))
    # Random orthogonal Q via QR.
    Q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    Y = X @ Q
    assert procrustes_distance(X, Y) == pytest.approx(0.0, abs=1e-6)


def test_procrustes_positive_for_unrelated() -> None:
    rng = np.random.default_rng(4)
    X = rng.standard_normal((80, 16))
    Y = rng.standard_normal((80, 16))
    assert procrustes_distance(X, Y) > 0.1


def test_leave_k_out_influence_contract() -> None:
    rng = np.random.default_rng(5)
    H_math = rng.standard_normal((60, 16))
    H_generic = rng.standard_normal((60, 16))
    out = leave_k_out_influence(H_math, H_generic, k=10, n_iter=20, base_seed=42)
    assert set(out.keys()) == {"base_cka", "max_abs_influence", "mean_abs_influence"}
    for v in out.values():
        assert np.isfinite(v)
    assert out["max_abs_influence"] >= out["mean_abs_influence"] >= 0.0
