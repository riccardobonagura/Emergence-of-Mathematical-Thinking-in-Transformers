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
    # Inject a shared anisotropic component so the mock "real" activations carry a
    # common direction → positive mean-cosine isotropy, clearly above the
    # random-Gaussian floor (which stays ≈0). Without this the random cloud would be
    # indistinguishable from its own null floor and the E-G-01 assertion would be flaky.
    base = torch.randn((n_stim, 64), generator=torch.Generator().manual_seed(0))
    base[:, 0] += 3.0
    fake_tensor = base.numpy()
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

    # Fix A: cka_inter_mean is the CLEAN full-n point estimate (NOT a duplication-
    # biased resample average), bracketed by a bias-corrected subsampling CI.
    ci_cols = {"cka_inter_ci_low", "cka_inter_ci_high"}
    assert ci_cols.issubset(df.columns), f"Missing inter-category CI columns: {ci_cols - set(df.columns)}"

    # Contract pin: the mean slot IS the single-shot point estimate. Reproduce the
    # balanced point exactly as run_rq1 does (same subsampling seed, same CKA seed)
    # and confirm the persisted mean equals it within rounding tolerance. The mock
    # feeds the SAME tensor at every layer, so the point is identical across rows —
    # any return to a resample-average would break this equality.
    from src.metrics.cka import compute_cka_intercategory
    from src.probing.seeds import get_seed

    cats = np.array(json.load(open(env["proc_dir"] / "metadata.json"))["categories"])
    m_raw = np.where(np.isin(cats, list(MATH_CATS)))[0]
    c_raw = np.where(np.isin(cats, list(CTRL_CATS)))[0]
    n_sub = min(m_raw.size, c_raw.size)
    rng_sub = np.random.default_rng(get_seed(42, "rq1_subsampling", 0))
    m_idx = np.sort(rng_sub.choice(m_raw, size=n_sub, replace=False))
    c_idx = np.sort(rng_sub.choice(c_raw, size=n_sub, replace=False))
    H64 = fake_tensor.astype(np.float64)  # run_rq1 casts to float64 before CKA
    point = compute_cka_intercategory(H64[m_idx], H64[c_idx], seed=42)
    assert np.allclose(df["cka_inter_mean"].to_numpy(), round(point, 6), atol=2e-6), \
        "cka_inter_mean must equal the clean point estimate (Fix A: no duplication-biased average)."

    # Bias-corrected band brackets the point by construction; assert the PROPERTY, not
    # tight bounds (the rank-deficient d=64 mock has a wide finite-sample CKA bias).
    # Rounding to 6dp can nudge an equal bound by <=1e-6, so allow that slack.
    assert (df["cka_inter_ci_low"] <= df["cka_inter_mean"] + 1e-6).all(), "ci_low above mean post-correction."
    assert (df["cka_inter_mean"] <= df["cka_inter_ci_high"] + 1e-6).all(), "mean above ci_high post-correction."
    assert (df["cka_inter_ci_high"] > df["cka_inter_ci_low"]).all(), "CI must be non-degenerate (ci_high > ci_low)."

    # Task C: matched-baseline + robustness battery columns.
    battery_cols = {
        "delta_vs_ctrl_baseline", "divergence_exceeds_baseline",
        "cka_inter_debiased", "procrustes_math_ctrl", "leave_k_influence",
    }
    assert battery_cols.issubset(df.columns), f"Missing battery columns: {battery_cols - set(df.columns)}"
    assert df["divergence_exceeds_baseline"].dtype == bool, "divergence_exceeds_baseline must be boolean."

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

    # E-G-01: random-Gaussian isotropy floor persisted as three additive columns.
    floor_cols = {"iso_floor", "iso_floor_ci_low", "iso_floor_ci_high"}
    assert floor_cols.issubset(df_iso.columns), f"Missing isotropy floor columns: {floor_cols - set(df_iso.columns)}"
    # Real activations sit ABOVE their random-Gaussian floor; on the mock the floor
    # is the null mean-cosine of an isotropic cloud (≈0), strictly below iso_math.
    assert (df_iso["iso_floor"] < df_iso["iso_math"]).all(), \
        "Isotropy floor must sit below iso_math (real data above its random-Gaussian floor)."


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


# ── RQ4: DETERMINIZATION METRIC ALGEBRA ───────────────────────────────────────
def test_rq4_metric_functions() -> None:
    """Pins the closed-form values of the three RQ4 logit metrics on known inputs."""
    from src.eval.determinization import (next_token_entropy, top1_top2_margin,
                                          prob_of_target)

    vocab = 50
    # Uniform logits → maximum entropy log(vocab), zero margin.
    uniform = np.zeros((1, vocab), dtype=np.float32)
    assert np.isclose(next_token_entropy(uniform)[0], np.log(vocab), atol=1e-4)
    assert np.isclose(top1_top2_margin(uniform)[0], 0.0, atol=1e-5)

    # One-hot-ish logits → entropy ≈ 0, P(target)=1, large margin.
    onehot = np.full((1, vocab), -1e4, dtype=np.float32)
    onehot[0, 7] = 1e4
    assert np.isclose(next_token_entropy(onehot)[0], 0.0, atol=1e-4)
    assert np.isclose(prob_of_target(onehot, np.array([7]))[0], 1.0, atol=1e-6)
    assert top1_top2_margin(onehot)[0] > 1e3

    # Hand-checked 2-row softmax: logits [0, ln2] → p = [1/3, 2/3].
    logits = np.array([[0.0, np.log(2.0)], [np.log(2.0), 0.0]], dtype=np.float32)
    np.testing.assert_allclose(prob_of_target(logits, np.array([0, 0])),
                               [1 / 3, 2 / 3], atol=1e-5)
    # margin = |ln2 - 0| = ln2 for both rows.
    np.testing.assert_allclose(top1_top2_margin(logits),
                               [np.log(2.0)] * 2, atol=1e-5)


# ── RQ4: TARGET-TOKEN BUILDER (stub tokenizer) ────────────────────────────────
class _StubTokenizer:
    """Maps exact strings to id lists so build_targets needs no real tokenizer."""
    def __init__(self, table: dict) -> None:
        self._table = table

    def encode(self, text: str):
        return self._table[text]


def test_rq4_build_targets() -> None:
    """Prefix-strip picks continuation[0]; single-token mask flags one-token results."""
    from src.eval.determinization import build_targets

    table = {
        "A =": [1, 2],          # single-token positive
        "A = 29": [1, 2, 99],
        "B =": [3, 4],          # multi-token negative: first token is the sign " -"
        "B = -29": [3, 4, 5, 6],
        "C =": [7],             # control row — must be filtered out, never encoded for result
    }
    stimuli = [
        {"id": "s0", "category": "CAT-SIGN", "text": "A =", "labels": {"result": 29}},
        {"id": "s1", "category": "CTRL-NEU", "text": "C =", "labels": {"result": -1}},
        {"id": "s2", "category": "CAT-PARITY", "text": "B =", "labels": {"result": -29}},
    ]
    target_ids, single = build_targets(_StubTokenizer(table), stimuli)

    # Math rows only, in source order: [A(single), B(multi)].
    assert target_ids.tolist() == [99, 5]
    assert single.tolist() == [True, False]


def test_rq4_build_targets_non_prefix_raises() -> None:
    """encode(text) not a prefix of encode(text+' '+result) must fail fast."""
    from src.eval.determinization import build_targets

    table = {"A =": [1, 2], "A = 29": [9, 9, 9]}  # prefix mismatch
    stimuli = [{"id": "s0", "category": "CAT-SIGN", "text": "A =", "labels": {"result": 29}}]
    with pytest.raises(ValueError, match="not a prefix"):
        build_targets(_StubTokenizer(table), stimuli)


# ── RQ4: LOGIT EXTRACTOR (stub model) ─────────────────────────────────────────
class _StubTok:
    pad_token_id = 0
    eos_token_id = 0


class _StubModel:
    """Returns a fixed [B,T,V] logit cube where logits[b,t,k] = 100b + 10t + k,
    so the gather position is verifiable. pad_id = 0; BOS shares it (column 0)."""
    tokenizer = _StubTok()

    def to_tokens(self, texts, prepend_bos=True):
        # Row 0 ends at index 2 (index 3 is pad); row 1 ends at index 3.
        return torch.tensor([[0, 5, 6, 0], [0, 7, 8, 9]])

    def __call__(self, tokens, attention_mask=None, return_type="logits"):
        b, t = tokens.shape
        v = 10
        grid = (100 * torch.arange(b).view(b, 1, 1)
                + 10 * torch.arange(t).view(1, t, 1)
                + torch.arange(v).view(1, 1, v))
        return grid.to(torch.float16)


def test_rq4_extract_eq_logits() -> None:
    """Gathers the last non-pad ('=') position per row and returns CPU float32."""
    from src.eval.determinization import extract_eq_logits

    stimuli = [
        {"id": "m0", "category": "CAT-SIGN", "text": "x ="},
        {"id": "c0", "category": "CTRL-NUM", "text": "y ."},   # filtered out
        {"id": "m1", "category": "CAT-PARITY", "text": "z ="},
    ]
    out = extract_eq_logits(_StubModel(), stimuli, batch_size=8)

    assert out.shape == (2, 10)
    assert out.dtype == np.float32
    # Row 0 gathered at t=2 → 100*0 + 10*2 + k; row 1 at t=3 → 100*1 + 10*3 + k.
    np.testing.assert_allclose(out[0], 20 + np.arange(10))
    np.testing.assert_allclose(out[1], 130 + np.arange(10))


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
