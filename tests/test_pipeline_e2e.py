"""
test_pipeline_e2e.py — End-to-End Integration and Statistical Unit Test Suite.
Validates the full pipeline execution (RQ1 -> RQ2 -> RQ3) under strict config-driven constraints.

Enforces fixes TE-01 to TE-08: isolates test steps, hardens structural assertions,
verifies category isolation invariants, and checks hyperplane denormalization algebra.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Import centralized Single Source of Truth contracts
from src.config.categories import MATH_CATS, CTRL_CATS


# ── FIX TE-05: MODULAR SCOPED FIXTURE ENVIRONMENT ─────────────────────────────
@pytest.fixture(scope="module")
def mock_pipeline_env(tmp_path_factory) -> dict:
    """Allocates a complete mock dataset and tensor structure inside a temporary sandbox."""
    # REFACTORED: Replaced invalid mget_basetemp with native getbasetemp and removed dead minterm_dir assignment
    tmp_dir = tmp_path_factory.getbasetemp() / "e2e_sandbox"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = tmp_dir / "data/processed/pythia-1.4b"
    proc_dir.mkdir(parents=True, exist_ok=True)

    out_rq1 = tmp_dir / "results/rq1_emergence"
    out_rq2 = tmp_dir / "results/rq2_probing"

    # Compile a balanced set of mock stimuli IDs spanning all 4 categories
    stimuli_ids = []
    categories = []
    sign_labels = []
    parity_labels = []
    operand1_labels = []
    operand2_labels = []

    # 4 categories * 30 samples = 120 total mock stimuli.
    # 30/cat keeps each balanced class large enough for cv=5 permutation testing inside the engine.
    cats_pool = list(MATH_CATS) + list(CTRL_CATS)
    idx_counter = 0
    samples_per_cat = 30

    # FIX TE-03: Inject a contaminated sample into CTRL to prove filtering works.
    # A control stimulus will have an arithmetic sign label (sign=1), but its category
    # field is CTRL-NEU. The probe dataset loader must ignore it during sign training splits.
    for cat in cats_pool:
        for s_idx in range(samples_per_cat):
            sid = f"{cat}_{s_idx:03d}"
            stimuli_ids.append(sid)
            categories.append(cat)

            if cat == "CAT-SIGN":
                sign_labels.append(1 if s_idx % 2 == 0 else 0)
                parity_labels.append(-1)
                operand1_labels.append(30 + s_idx)
                operand2_labels.append(15)
            elif cat == "CAT-PARITY":
                sign_labels.append(-1)
                parity_labels.append(1 if s_idx % 2 == 0 else 0)
                operand1_labels.append(20)
                operand2_labels.append(10 + s_idx)
            elif cat == "CTRL-NEU" and s_idx == 0:
                # TE-03: Intentional label contamination payload for isolation testing
                sign_labels.append(1)
                parity_labels.append(-1)
                operand1_labels.append(99) # Out of range magnitude
                operand2_labels.append(99)
            else:
                sign_labels.append(-1)
                parity_labels.append(-1)
                operand1_labels.append(0)
                operand2_labels.append(0)

            idx_counter += 1

    # Format JSONL master file
    master_jsonl = tmp_dir / "data/processed/dataset_master_v5.jsonl"
    master_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(master_jsonl, "w", encoding="utf-8") as f:
        for idx in range(len(stimuli_ids)):
            row = {
                "id": stimuli_ids[idx],
                "category": categories[idx],
                "labels": {
                    "sign": sign_labels[idx],
                    "parity": parity_labels[idx],
                    "operand1": operand1_labels[idx],
                    "operand2": operand2_labels[idx]
                },
                "text": f"Mock string sequence placeholder {idx} ="
            }
            f.write(json.dumps(row) + "\n")

    # Format extraction metadata JSON payload structure
    meta_payload = {
        "n_layers": 24,
        "d_model": 64,  # Miniature embedding dimension for execution optimization
        "n_stimuli": len(stimuli_ids),
        "stimuli_ids": stimuli_ids,
        "categories": categories,
        "probe_strategy": "gathered_terminal",
        "labels": {
            "sign": sign_labels,
            "parity": parity_labels,
            "operand1": operand1_labels,
            "operand2": operand2_labels
        }
    }
    with open(proc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    # FIX TE-06: Distinct base and checkpoint layers to enable non-zero geometric drift loops
    # Generate 24 layers of activation tensors
    for l in range(24):
        # Base model layer activations (Layer l)
        H_base = torch.randn((len(stimuli_ids), 64)) * 0.5
        torch.save(H_base, proc_dir / f"layer_{l:02d}.pt")

        # Checkpoint model layer activations (Base + synthetic gradient shift delta)
        ckpt_dir = tmp_dir / "data/processed/checkpoints/ckpt_500"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        H_ckpt = H_base + (torch.randn_like(H_base) * 0.1) # Controlled non-zero delta injection
        torch.save(H_ckpt, ckpt_dir / f"layer_{l:02d}.pt")

    # Format localized YAML operational testing config
    config_payload = {
        "model_name": "pythia-1.4b",
        "seed": 42,
        "train_split": 0.5,
        "max_iter": 10,
        "C": 1.0,
        "solver": "lbfgs",
        "multiclass_strategy": "ovr",
        "bootstrap_n_samples": 10,
        "bootstrap_ci": 0.95,
        "n_jobs": 1,
        "n_permutation_tests": 5,
        "min_class_samples": 4,
        "output_dir": str(out_rq2),
        "figures_dir": str(tmp_dir / "results/figures/rq2"),
        "properties": {
            "sign": {"label_field": "sign", "category": "CAT-SIGN", "type": "binary"},
            "parity": {"label_field": "parity", "category": "CAT-PARITY", "type": "binary"}
        }
    }
    config_file = tmp_dir / "configs/config_test_e2e.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config_payload, f)

    return {
        "root": tmp_dir,
        "config": config_file,
        "proc_dir": proc_dir,
        "out_rq1": out_rq1,
        "out_rq2": out_rq2,
        "ckpt_dir": ckpt_dir,
        "master_jsonl": master_jsonl
    }


# ── SECTION 1 — RQ1 PIPELINE INTEGRATION RUNNER ────────────────────────────────

@patch("run_rq1.load_hidden_states")
@patch("run_rq1.run_isotropy_analysis")
def test_rq1_pipeline(mock_iso, mock_load, mock_pipeline_env, monkeypatch) -> None:
    """Validates the execution flow, outlier evaluations, and structured CSV output of RQ1."""
    env = mock_pipeline_env

    # Sized to match the fixture's full stimuli count so RQ1's index lookups don't overflow
    with open(env["proc_dir"] / "metadata.json", "r", encoding="utf-8") as f:
        n_stim = json.load(f)["n_stimuli"]
    fake_tensor = torch.randn((n_stim, 64)).numpy()
    mock_load.return_value = fake_tensor

    import run_rq1

    # Chdir into the sandbox so relative paths inside the orchestrator resolve there
    monkeypatch.chdir(env["root"])
    test_args = ["run_rq1.py", "--config", str(env["config"])]
    with patch.object(sys, "argv", test_args):
        run_rq1.main()

    out_rq1 = env["root"] / "results" / "rq1_emergence"
    result_csv = out_rq1 / "cka_results_annotated.csv"
    assert result_csv.exists(), "RQ1 failed to commit the annotated metrics table to disk."

    df = pd.read_csv(result_csv)
    required_cols = {"layer", "cka_evo_math", "cka_evo_ctrl", "cka_inter_mean", "delta_cka_evolution"}
    assert required_cols.issubset(df.columns), f"Metrics table schema mismatch. Missing: {required_cols - set(df.columns)}"
    assert len(df) == 24, "Layer indices mismatch inside the output metrics table rows."

    # Task D: cka_inter_mean is now a bootstrap mean bracketed by its percentile CI.
    ci_cols = {"cka_inter_ci_low", "cka_inter_ci_high"}
    assert ci_cols.issubset(df.columns), f"Missing inter-category CI columns: {ci_cols - set(df.columns)}"
    assert (df["cka_inter_ci_low"] <= df["cka_inter_mean"] + 1e-9).all(), "ci_low above mean."
    assert (df["cka_inter_mean"] <= df["cka_inter_ci_high"] + 1e-9).all(), "mean above ci_high."

    # M-03: per-layer inter-category CKA vector persisted as .npy
    cka_npy = out_rq1 / "cka_intercategory.npy"
    assert cka_npy.exists(), "RQ1 did not persist cka_intercategory.npy (M-03)."
    assert np.load(cka_npy).shape == (24,), "cka_intercategory.npy must have one entry per layer."

    # M-05: balanced aggregated isotropy table with equal-N math vs ctrl pools
    iso_csv = out_rq1 / "isotropy_aggregated_balanced.csv"
    assert iso_csv.exists(), "RQ1 did not write isotropy_aggregated_balanced.csv (M-05)."
    df_iso = pd.read_csv(iso_csv)
    assert {"layer", "iso_math", "iso_ctrl", "delta_iso", "n_per_side"}.issubset(df_iso.columns), \
        "Balanced isotropy table missing required columns."
    assert df_iso["n_per_side"].nunique() == 1, "Aggregated isotropy must use a single equal N across layers."


# ── SECTION 2 — RQ2 PIPELINE INTEGRATION RUNNER ────────────────────────────────

@patch("run_rq2.load_hidden_states")
def test_rq2_pipeline(mock_load, mock_pipeline_env, monkeypatch) -> None:
    """Validates the static linear probing engine, FDR constraints, and control sanity metric."""
    env = mock_pipeline_env

    def side_effect(path):
        return torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    mock_load.side_effect = side_effect

    import run_rq2

    monkeypatch.chdir(env["root"])
    test_args = ["run_rq2.py", "--config", str(env["config"])]
    with patch.object(sys, "argv", test_args):
        run_rq2.main()

    metrics_csv = env["out_rq2"] / "accuracy_metrics_corrected.csv"
    assert metrics_csv.exists(), "RQ2 failed to write the statistical metrics block to disk."

    df = pd.read_csv(metrics_csv)
    assert "is_significant" in df.columns, "Fix TE-04 Error: Multiple comparison flag column missing."
    assert "ctrl_positive_pred_rate" in df.columns, "Control prediction rate sanity column missing."
    assert "gap_robustness_delta" in df.columns, "Difficulty gradient tracking metric column missing."
    assert df["ctrl_positive_pred_rate"].between(0.0, 1.0).all(), "ctrl_positive_pred_rate out of [0,1] bounds."

    assert df["accuracy"].min() >= 0.0 and df["accuracy"].max() <= 1.0, "Accuracy metrics out of bounds."


# ── FIX TE-03: CATEGORY FILTER INVARIANT VALIDATION UNIT ─────────────────────
def test_category_filter_invariant(mock_pipeline_env) -> None:
    """Ensures contaminated cross-category tokens are strictly isolated from probing sets."""
    env = mock_pipeline_env
    from src.probing.probing_dataset import ProbingDataset

    with open(env["root"] / "data/processed/pythia-1.4b/metadata.json", "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    dataset = ProbingDataset(
        stimuli_path=env["master_jsonl"],
        stimuli_ids=meta_data["stimuli_ids"]
    )

    prop_cfg = {"label_field": "sign", "category": "CAT-SIGN", "type": "binary"}
    indices, labels = dataset._extract("sign", prop_cfg)

    with open(env["master_jsonl"], "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    for extracted_idx in indices:
        matched_stimulus = rows[extracted_idx]
        assert matched_stimulus["category"] == "CAT-SIGN", (
            f"Cross-category leakage! Probe loaded a sample from '{matched_stimulus['category']}' "
            "when filtering for 'CAT-SIGN'."
        )
        assert matched_stimulus["labels"]["operand1"] != 99, "Contaminated row leaked into active split."


# ── SECTION 3 — RQ3 PIPELINE INTEGRATION RUNNER ────────────────────────────────

@patch("run_rq3.load_hidden_states")
def test_rq3_pipeline(mock_load, mock_pipeline_env, monkeypatch) -> None:
    """Validates dynamic frozen probe evaluation and mathematical drift accumulation layers."""
    env = mock_pipeline_env

    def side_effect(path):
        return torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    mock_load.side_effect = side_effect

    import run_rq3

    monkeypatch.chdir(env["root"])
    test_args = ["run_rq3.py", "--config", str(env["config"]), "--checkpoint_dir", str(env["ckpt_dir"])]
    with patch.object(sys, "argv", test_args):
        run_rq3.main()

    trajectory_csv = env["out_rq2"] / "dynamic/trajectories_probing.csv"
    assert trajectory_csv.exists(), "RQ3 trajectory logs missing from output directory."

    df = pd.read_csv(trajectory_csv)

    for col in ["geom_delta_math", "geom_delta_ctrl", "geom_delta_math_rel", "geom_delta_ctrl_rel"]:
        assert col in df.columns, f"Drift column '{col}' missing from trajectory CSV."

    assert (df["geom_delta_math"] > 0.0).any(), "Geometric drift validation failed on active branches."
    assert (df["geom_delta_math_rel"] > 0.0).any(), "Relative Frobenius drift must be positive."


# ── FIX TE-08: PROBING DENORMALIZATION ALGEBRA VERIFIER ───────────────────────
def test_probing_algebra() -> None:
    """Enforces strict unit validation testing on the classifier space projection algebra."""
    rng = np.random.default_rng(42)
    X_raw = rng.normal(loc=5.0, scale=2.0, size=(100, 10))
    y = rng.choice([0, 1], size=100)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    clf = LogisticRegression(C=1.0, solver="lbfgs")
    clf.fit(X_scaled, y)

    w_scaled = clf.coef_[0]
    b_scaled = clf.intercept_[0]

    mu = scaler.mean_
    sigma = scaler.scale_

    w_orig = w_scaled / sigma
    b_orig = b_scaled - np.dot(w_scaled, mu / sigma)

    raw_scores = np.dot(X_raw, w_orig) + b_orig
    scaled_scores = np.dot(X_scaled, w_scaled) + b_scaled

    np.testing.assert_allclose(
        raw_scores, scaled_scores, rtol=1e-5,
        err_msg="Algebraic denormalization projection broke representation tracking invariants."
    )


# ── M-01: ISOTROPY SIGN-CONVENTION INVARIANT ──────────────────────────────────
def test_isotropy_sign_convention() -> None:
    """
    Locks the ISO formula direction documented in docs/Guida_Metodologica.md:
    ISO = mean off-diagonal cosine. High ISO → vectors collinear → anisotropic.
    Low ISO → vectors spread → isotropic. Consequently, ΔIso = ISO(math) − ISO(ctrl) < 0
    means math representations are *more isotropic* than control (not more anisotropic).
    """
    from src.metrics.isotropy import isotropy_exact

    rng = np.random.default_rng(0)

    # Near-collinear bundle: 100 copies of e_0 with tiny perturbation → ISO ≈ 1.
    base = torch.zeros(100, 64)
    base[:, 0] = 1.0
    H_aniso = base + 1e-3 * torch.randn(100, 64, generator=torch.Generator().manual_seed(0))
    iso_aniso, _, _, _ = isotropy_exact(H_aniso)
    assert iso_aniso > 0.9, (
        f"Anisotropic bundle should produce ISO close to 1.0, got {iso_aniso:.4f}. "
        "If this fails the formula sign convention has flipped."
    )

    # Random unit vectors in high d → pairwise cosines ≈ 0 → ISO ≈ 0.
    H_iso = torch.from_numpy(rng.standard_normal((500, 64)).astype(np.float32))
    iso_iso, _, _, _ = isotropy_exact(H_iso)
    assert abs(iso_iso) < 0.1, (
        f"Random high-d unit vectors should produce ISO close to 0, got {iso_iso:.4f}. "
        "If this fails the formula sign convention has flipped."
    )

    # Cross-check: the anisotropic bundle's ISO must exceed the isotropic one — the
    # ordering, not the absolute values, is what RQ1's ΔIso interpretation hangs on.
    assert iso_aniso > iso_iso, (
        f"Sign convention violated: anisotropic ISO ({iso_aniso:.4f}) is not strictly "
        f"greater than isotropic ISO ({iso_iso:.4f})."
    )
