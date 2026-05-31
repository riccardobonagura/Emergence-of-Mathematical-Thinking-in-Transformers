"""Unit tests for the NF4 signal-to-noise interpretation helper (pure CPU)."""

import pandas as pd

from src.eval.nf4_degradation import compute_nf4_snr_interpretation


def test_snr_present_csv_math() -> None:
    df = pd.DataFrame({"geom_delta_math_rel": [0.01, 0.09, 0.05]})
    max_drift, snr, interp = compute_nf4_snr_interpretation(0.03, df)
    assert max_drift == 0.09
    assert snr == 0.09 / 0.03  # 3.0
    assert "SNR" in interp and "drift exceeds quantization floor" in interp


def test_snr_below_floor_caution() -> None:
    df = pd.DataFrame({"geom_delta_math_rel": [0.02, 0.04]})
    _, snr, interp = compute_nf4_snr_interpretation(0.03, df)
    assert snr < 3
    assert "caution" in interp


def test_missing_csv_yields_floor_only_and_null_keys() -> None:
    max_drift, snr, interp = compute_nf4_snr_interpretation(0.03, None)
    assert max_drift is None
    assert snr is None
    assert interp.startswith("floor only")
    assert "SNR not computed" in interp
