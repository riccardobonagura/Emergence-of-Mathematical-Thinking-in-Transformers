"""Unit tests for the NF4 signal-to-noise interpretation helper (pure CPU)."""

import json

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


def test_zero_floor_yields_null_snr_and_valid_json() -> None:
    df = pd.DataFrame({"geom_delta_math_rel": [0.02, 0.05]})
    max_drift, snr, interp = compute_nf4_snr_interpretation(0.0, df)
    assert snr is None
    assert "not computable" in interp
    # signal_to_noise_ratio must round-trip through JSON (no Infinity/NaN).
    summary = {
        # frozen: serialized key in nf4_degradation/summary.json (T6 byte-preservation), not an RQ label
        "rq3_max_relative_drift": round(max_drift, 6) if max_drift is not None else None,
        "signal_to_noise_ratio": round(snr, 4) if snr is not None else None,
        "interpretation": interp,
    }
    restored = json.loads(json.dumps(summary))
    assert restored["signal_to_noise_ratio"] is None
