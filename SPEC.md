
Repo Spec — Geometric Dynamics in Transformer Internal Representations

Project: Bachelor thesis CS, Federico II Napoli, Bonagura N46007216
Model: Pythia-1.4B (EleutherAI), GPT-NeoX arch, 24 layers, d_model=2048
Hardware: RTX 5080 16GB, WSL2, conda transformer_thesis
Codebase: ~7,900 LOC Python, branch dev

1. Research Questions

RQ1 — Where does arithmetic structure emerge geometrically?
Measure per-layer isotropy differential (ΔIso) and inter-category CKA between math stimuli and linguistic controls. ΔIso < 0 = math representations more isotropic. CKA drop from 1.0 = structural divergence. Purely correlative — no causal claims.

RQ2 — Which arithmetic properties are linearly decodable, and at which layers?
Train one logistic regression probe per (layer, property) pair. Properties: sign of result (binary), parity of result (binary). Statistical gating: bootstrap CI (N=1000), permutation test (N=1000, train-only CV), Benjamini-Hochberg FDR over 48 tests. Confound checks: cosine(w_sign, w_magnitude), Pearson(probe_logits, operand1).

RQ3 — Does QLoRA fine-tuning produce measurable geometric reorganization?
Fine-tune with QLoRA (NF4, r=16, QKV-only, MLP frozen) on MetaMathQA. At each checkpoint: merge adapter, re-extract via TransformerLens, apply frozen RQ2 probes, measure Frobenius drift (dim-normalized and relative). Evaluate 0-shot GSM8K at each step. Compare drift to NF4 quantization baseline (T16).

---
2. Dataset — data/processed/dataset_master_v5.jsonl

3000 stimuli, 4 categories, contrastive minimal-pair design (BLiMP-inspired):

┌────────────┬────────────┬───────────────────┬────────────┬─────────────┐
│  Category  │     N      │      Purpose      │  Operands  │  Operators  │
├────────────┼────────────┼───────────────────┼────────────┼─────────────┤
│ CAT-SIGN   │ 1000 (500  │ Binary sign of    │ a,b ∈      │ Subtraction │
│            │ pairs)     │ result            │ [10,50]    │             │
├────────────┼────────────┼───────────────────┼────────────┼─────────────┤
│ CAT-PARITY │ 1000 (500  │ Binary parity of  │ a,b ∈      │ Add + Sub   │
│            │ pairs)     │ result            │ [10,50]    │             │
├────────────┼────────────┼───────────────────┼────────────┼─────────────┤
│ CTRL-NEU   │ 500        │ Linguistic        │ —          │ —           │
│            │            │ baseline (prose)  │            │             │
├────────────┼────────────┼───────────────────┼────────────┼─────────────┤
│ CTRL-NUM   │ 500        │ Numeric context   │ —          │ —           │
│            │            │ baseline          │            │             │
└────────────┴────────────┴───────────────────┴────────────┴─────────────┘

Record schema (one JSON per line):
id, text, category, template_id, labels{result, sign, parity, operand1, operand2},
contrast{pair_id, varying_axis, controlled_axes},
token_fields{n_tokens, token_ids, equals_sign_index}, dataset_version: "v5"

Controls use sentinel label -1 for sign/parity. Pair members (e.g. SIGN-0000-A/B) differ only on the target property. All operands are single-token under GPT-NeoX tokenizer. 3 templates per category (template_id).

---
3. Extraction — src/extraction/extract_states.py

Input: dataset_master_v5.jsonl + Pythia-1.4B via TransformerLens
Output: data/processed/pythia-1.4b/layer_00.pt … layer_23.pt + metadata.json

- TransformerLens HookedTransformer.from_pretrained(fold_ln=True, dtype=fp16)
- Right-padding via to_tokens(prepend_bos=True); pad_id = eos_id = 0 (Pythia quirk)
- Terminal token gathered via _last_token_indices(): scans from right for first non-pad, NOT [:, -1, :]
- validate_extraction_tokens(): pre-flight check that gathered token decodes to "=" for math stimuli
- attention_mask explicitly computed; column 0 forced to 1 (BOS shares pad's id)
- Hooks on blocks.{l}.hook_resid_post — residual stream after each transformer block
- Each layer tensor: [3000, 2048] FP16, persisted via torch.save
- metadata.json contract (ExtractionMetadata TypedDict): n_layers, d_model, n_stimuli, stimuli_ids, categories, labels{sign, parity, operand1, operand2}, probe_strategy: "gathered_terminal", dataset_version

Critical constraint: transformers>=4.46,<4.49 — GPT-NeoX vmap/SDPA bug in ≥4.49.

---
4. RQ1 Pipeline — run_rq1.py

Input: extracted tensors + metadata
Output: results/rq1_emergence/ — CSVs + .npy arrays

1. Isotropy analysis via isotropy_exact(): mean off-diagonal cosine similarity per (layer, category). ISO high = anisotropic, ISO low = isotropic. ΔIso = ISO(math) − ISO(ctrl); negative = math more isotropic.
2. Evolutionary CKA: linear_cka(H_l, H_{l-1}) — structural change between consecutive layers. Separate curves for math and ctrl. Dip = structural reorganization.
3. Inter-category CKA: compute_cka_intercategory(H_math, H_ctrl) — similarity between math and ctrl subspaces. Hardcoded OUT_DIR = results/rq1_emergence (separate from RQ2).
4. Baselines (reviewer-mandated): CKA(CTRL-NEU, CTRL-NUM) and within-math across-template CKA to bound positional confound. All subsampling via get_seed().
5. Balanced isotropy aggregation — equal N per side (math vs ctrl).

---
5. RQ2 Pipeline — run_rq2.py

Input: extracted tensors + dataset_master_v5.jsonl + config
Output: results/rq2_probing/ — accuracy CSV, probe weights, test indices, config hash

Flow

1. Load metadata → validate probe_strategy == "gathered_terminal" (fail-fast)
2. Build ProbingDataset — aligns JSONL records to metadata stimuli_ids order
3. For each property (sign, parity):
  - get_property_split(): category filter → balanced undersampling → pair-aware group split (contrastive pair members stay together in train or test)
  - save_test_indices() — frozen before any training
  - Sample control inputs from CTRL categories for sanity metric
4. Parallel dispatch via joblib.Parallel across 24 layers × 2 properties
5. Per-layer worker (process_task):
  - ProbingEngine.run_layer(): StandardScaler → LogisticRegression(C=10, lbfgs) → denormalize weights to original space → bootstrap CI → permutation test → confound correlation
  - ctrl_positive_pred_rate: frozen probe applied to control inputs, fraction predicted positive (~0.5 = not spuriously active)
  - gap_robustness_delta: accuracy(easy quartile) − accuracy(hard quartile) by operand gap
6. BH FDR correction over all 48 p-values
7. Write accuracy_metrics_corrected.csv + rq2_config_hash.json

TypedDict contracts (ARCH-03)

- LayerResult: engine output — layer, property, accuracy, CI bounds, p-value, confound stats, weights, bias
- RQ2Result: extends LayerResult with ctrl_positive_pred_rate, gap_robustness_delta, is_significant
- PropConfig: label_field (required), type (required), category (optional)

Probing internals (src/probing/)

┌────────────────────┬────────────────────────────────────────────────────┐
│       Module       │                        Role                        │
├────────────────────┼────────────────────────────────────────────────────┤
│ seeds.py           │ MD5-based deterministic seed derivation.           │
│                    │ get_seed(base, purpose, offset) → int              │
├────────────────────┼────────────────────────────────────────────────────┤
│                    │ build_pipeline() → StandardScaler +                │
│ pipeline.py        │ LogisticRegression. denormalize_classifier() →     │
│                    │ project w,b back to unscaled space: w_orig = w/σ,  │
│                    │ b_orig = b − w·μ/σ                                 │
├────────────────────┼────────────────────────────────────────────────────┤
│                    │ ProbingEngine.run_layer(): fit → score →           │
│ engine.py          │ denormalize → bootstrap CI → permutation test →    │
│                    │ confound test. Config validated at construction.   │
├────────────────────┼────────────────────────────────────────────────────┤
│                    │ ProbingDataset: JSONL↔metadata alignment, category │
│ probing_dataset.py │  filtering, balanced undersampling, pair-aware     │
│                    │ splitting via contrast.pair_id                     │
├────────────────────┼────────────────────────────────────────────────────┤
│                    │ bootstrap_ci(), rigorous_permutation_test()        │
│ stats.py           │ (train-only CV, n_jobs=1 to avoid nested           │
│                    │ parallelism), benjamini_hochberg_correction()      │
├────────────────────┼────────────────────────────────────────────────────┤
│ directions.py      │ cosine_similarity(), angle_degrees(),              │
│                    │ test_confound_correlation()                        │
├────────────────────┼────────────────────────────────────────────────────┤
│                    │ MetadataHandler, load_hidden_states(),             │
│ io_utils.py        │ _atomic_write_csv/json/npy,                        │
│                    │ save/load_test_indices, save_weights               │
└────────────────────┴────────────────────────────────────────────────────┘

---
6. Confound Validation — src/probing/run_confound_checks.py + run_parity_confound_checks.py

Standalone diagnostic scripts (not called by orchestrators). Run after RQ2.

N-01 (sign confound): Does the sign probe decode abstract sign or operand1 magnitude?
- V1: Linear regression R² for operand1 from hidden states (permutation-gated)
- V2: cosine(w_sign, w_op1_probe) — direction alignment
- V3: Pearson(frozen sign logits, operand1 values) — direct triangulation
- BH correction across layers on op1 R² p-values

N-02 (parity confound): Does the parity probe decode result parity or operand2 parity?
- V1: operand2 value decodability (R², permuted)
- V2: operand2 parity decodability (LogisticRegression accuracy + cosine alignment)
- V3: Pearson(frozen parity logits, operand2 parity)
- Dataset protection diagnostic: first-operand parity balance + ground-truth corr(result_parity, op2_parity)

Both modules include index-space assertion: metadata.stimuli_ids order == JSONL line order.

---
7. Fine-Tuning — src/finetuning/train_qlora.py

Input: config + lora_config + MetaMathQA (HuggingFace Hub)
Output: data/processed/checkpoints/checkpoint-{step}/ + final_checkpoint/

- BitsAndBytes NF4 quantization (double_quant=True, bfloat16 compute)
- prepare_model_for_kbit_training() → LoRA config from ModelProfile.target_modules (= ["query_key_value"])
- r=16, alpha=32, dropout=0.1, bias=none
- Left-padding tokenizer (correct for causal LM training)
- MetaMathQA 95/5 train/val split via get_seed(seed, "dataset_splitting", 0)
- max_seq_length=1024, 1 epoch, cosine LR schedule, save every 500 steps
- Config key for output: checkpoints_dir (not output_dir, avoids collision with RQ2)

---
8. Checkpoint Loop — src/extraction/checkpoint_loop.py

Input: data/processed/checkpoints/ + config
Output: data/processed/checkpoints_extracted/{ckpt_name}/ (same format as base extraction)

For each checkpoint:
1. deepcopy(base_hf) (pre-loaded once on CPU)
2. PeftModel.from_pretrained() → merge_and_unload()
3. HookedTransformer.from_pretrained(hf_model=merged, fold_ln=True, fp16) — same config as base extraction
4. extract_from_model() — reuses the main extraction pipeline
5. subprocess.run(["python", "run_rq3.py", ...]) — triggers dynamic evaluation
6. GPU cleanup between checkpoints

---
9. RQ3 Pipeline — run_rq3.py

Input: checkpoint extracted tensors + base tensors + frozen RQ2 weights/test_indices + config
Output: results/rq2_probing/dynamic/trajectories_probing.csv

1. Parse checkpoint step from directory name
2. Validate seed consistency via rq2_config_hash.json
3. Subsample math/ctrl indices for drift evaluation
4. Per layer:
  - Load H_base and H_ckpt
  - compute_geometric_drift() → returns (dim_normalized, relative) Frobenius for both math and ctrl subsets
  - For each property: load frozen weights → apply to H_ckpt[test_idx] → accuracy
5. Append-or-replace results in CSV (idempotent per step)

TypedDict: RQ3TrajectoryRow — step, layer, property, probing_acc, geom_delta_math, geom_delta_ctrl, geom_delta_math_rel, geom_delta_ctrl_rel

Dual Frobenius metrics:
- geom_delta_math: ||H_ckpt − H_base||_F / (N × d) — dim-normalized
- geom_delta_math_rel: ||H_ckpt − H_base||_F / ||H_base||_F — scale-invariant, used for T16 comparison

---
10. Evaluation — src/eval/

eval_gsm8k.py

- lm_eval.simple_evaluate(model="hf", tasks=["gsm8k"], num_fewshot=0)
- Seed from get_seed(config["seed"], "gsm8k_evaluation", 0) passed to all three RNG slots
- Wald binomial CI assuming N=1319 test samples
- Appends GSM8K accuracy to trajectories_probing.csv (merges on step column)
- Supports loading strategies: peft (unmerged adapter), merged_cpu, merged_direct

nf4_degradation.py (T16)

- Compares FP16 vs NF4 representations using native HF forward hooks (NOT TransformerLens — incompatible with quantized weights)
- Left-padding + [:, -1, :] extraction (valid: last position = terminal token with left-padding)
- Balanced stratified sampling: 75 stimuli × 4 categories = 300
- Reports frobenius_dist_relative and frobenius_dist_normalized_dim per layer
- Interpretation thresholds from Dettmers et al. (2023): <3% negligible, <5% minor, >5% significant

---
11. Config Files

configs/config_rq2.yaml (master config for RQ1/RQ2/RQ3)

model_name: pythia-1.4b
seed: 42
train_split: 0.80
C: 10.0                    # Selected from C-sweep
max_iter: 1000
solver: lbfgs
bootstrap_n_samples: 1000
n_permutation_tests: 1000
n_jobs: -1
output_dir: results/rq2_probing
eval_subset_size: 200
properties:
  sign:   {label_field: sign,   category: CAT-SIGN,   type: binary}
  parity: {label_field: parity, category: CAT-PARITY, type: binary}

configs/lora_config.yaml

r=16, alpha=32, dropout=0.1, bits=4, lr=2e-4, 1 epoch, save every 500 steps. target_modules delegated to ModelProfile.

---
12. Data Layout

data/processed/
├── dataset_master_v5.jsonl              # 3000 stimuli
├── pythia-1.4b/                         # Base extraction
│   ├── metadata.json                    # ExtractionMetadata
│   └── layer_00.pt … layer_23.pt       # [3000, 2048] FP16
├── checkpoints/                         # QLoRA training output
│   ├── checkpoint-500/ … checkpoint-N/
│   └── final_checkpoint/
└── checkpoints_extracted/               # Re-extracted checkpoint tensors
    └── checkpoint-500/
        ├── metadata.json
        └── layer_00.pt … layer_23.pt

results/
├── rq1_emergence/
│   ├── isotropy_pythia.csv
│   ├── cka_results_annotated.csv
│   ├── cka_intercategory.npy
│   ├── cka_math_evol.npy, cka_ctrl_evol.npy
│   └── isotropy_aggregated_balanced.csv
├── rq2_probing/
│   ├── accuracy_metrics_corrected.csv
│   ├── emergence_summary.json
│   ├── direction_angles.csv
│   ├── weights/
│   │   ├── layer_XX_sign.npy, layer_XX_sign_bias.npy
│   │   ├── layer_XX_parity.npy, layer_XX_parity_bias.npy
│   │   └── rq2_config_hash.json
│   ├── test_indices/
│   │   ├── sign_test_idx.npy
│   │   └── parity_test_idx.npy
│   ├── dynamic/
│   │   └── trajectories_probing.csv
│   ├── confound_checks_hardened.csv
│   └── parity_confound_checks.csv
├── gsm8k/
│   └── gsm8k_{tag}.json
├── nf4_degradation/
│   ├── per_layer_stats.csv
│   └── summary.json
└── figures/
    ├── rq1_emergence/rq1_emergence.html
    ├── rq2/accuracy_curves.png, .html, orthogonality_*.png, .html
    └── rq3/rq3_dashboard.html

---
13. Visualization — src/viz/

Module: probing_viz.py
Reads: RQ2 accuracy DataFrame
Produces: accuracy_curves.png/.html, orthogonality heatmap
────────────────────────────────────────
Module: plot_rq1_emergence.py
Reads: isotropy CSV + CKA .npy
Produces: 2-panel Plotly dashboard (ΔIso + evolutionary CKA)
────────────────────────────────────────
Module: plot_rq3_trajectory.py
Reads: trajectories_probing.csv
Produces: 3-panel Plotly dashboard (accuracy trajectory, drift heatmap,
  drift↔Δacc scatter)
────────────────────────────────────────
Module: pca_umap_viz.py
Reads: layer tensors + metadata
Produces: PCA/UMAP 2D scatter colored by category

---
14. Testing — tests/test_pipeline_e2e.py

6 tests, all CPU-only, no GPU required:

┌────────────────────────────────┬────────────────────────────────────────┐
│              Test              │           What it validates            │
├────────────────────────────────┼────────────────────────────────────────┤
│                                │ RQ1 runs end-to-end on synthetic       │
│ test_rq1_pipeline              │ 120-stimulus fixture, outputs correct  │
│                                │ CSV schema                             │
├────────────────────────────────┼────────────────────────────────────────┤
│                                │ RQ2 runs with real probing engine on   │
│ test_rq2_pipeline              │ random tensors, BH correction applied, │
│                                │  ctrl_positive_pred_rate present       │
├────────────────────────────────┼────────────────────────────────────────┤
│                                │ RQ3 reads RQ2 weights, applies frozen  │
│ test_rq3_pipeline              │ probe to checkpoint tensors, outputs   │
│                                │ drift columns including _rel variants  │
├────────────────────────────────┼────────────────────────────────────────┤
│ test_category_filter_invariant │ CTRL stimulus with contaminated sign   │
│                                │ label is excluded from CAT-SIGN split  │
├────────────────────────────────┼────────────────────────────────────────┤
│                                │ Denormalization math: w_orig·X_raw +   │
│ test_probing_algebra           │ b_orig == w_scaled·X_scaled + b_scaled │
│                                │  within 1e-5                           │
├────────────────────────────────┼────────────────────────────────────────┤
│                                │ ISO(collinear bundle) ≈ 1.0,           │
│ test_isotropy_sign_convention  │ ISO(random vectors) ≈ 0.0, ordering    │
│                                │ preserved                              │
└────────────────────────────────┴────────────────────────────────────────┘

Fixture: 120 stimuli (30 per category), 24 layers, d_model=64. load_hidden_states monkeypatched with random/real tensors. Module-scoped fixture shared across RQ1→RQ2→RQ3.

---
15. Cross-Module Contracts

dataset_master_v5.jsonl
    │  id, category, labels, contrast.pair_id
    ▼
extract_states.py ──► metadata.json (stimuli_ids, categories, labels, probe_strategy)
    │                  layer_XX.pt  [n_stimuli, d_model]
    ▼
run_rq1.py ◄── isotropy.py, cka.py
    │  reads: layer tensors + metadata
    │  writes: isotropy CSV, CKA .npy
    │
run_rq2.py ◄── ProbingDataset, ProbingEngine, stats.py
    │  reads: layer tensors + metadata + JSONL
    │  writes: accuracy CSV, weights/*.npy, test_indices/*.npy, rq2_config_hash.json
    │
    ├── run_confound_checks.py  (reads: weights, test_indices, JSONL, tensors)
    ├── run_parity_confound_checks.py
    │
train_qlora.py ──► data/processed/checkpoints/
    │
checkpoint_loop.py
    │  reads: checkpoints/ + base model + stimuli
    │  writes: checkpoints_extracted/{ckpt}/layer_XX.pt + metadata.json
    │  triggers: run_rq3.py via subprocess
    │
run_rq3.py
    │  reads: base tensors + checkpoint tensors + frozen weights + test_indices + config_hash
    │  writes: trajectories_probing.csv (append-or-replace per step)
    │
eval_gsm8k.py
    │  reads: model checkpoint + config
    │  writes: gsm8k_{tag}.json + appends to trajectories_probing.csv
    │
nf4_degradation.py
       reads: base model + config + stimuli
       writes: per_layer_stats.csv + summary.json (T16 baseline)

---
16. Invariants (Enforced Everywhere)

┌─────────────────┬──────────────────────────────────────────────────────┐
│      Rule       │                      Mechanism                       │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Seed discipline │ get_seed(base, purpose, offset) from seeds.py. Never │
│                 │  raw seed(42) or default_rng(42).                    │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Atomic writes   │ _atomic_write_csv/json, _atomic_save_npy — tempfile  │
│                 │ + os.replace. No raw open().write().                 │
├─────────────────┼──────────────────────────────────────────────────────┤
│ UTF-8 encoding  │ Every open() call has encoding="utf-8".              │
├─────────────────┼──────────────────────────────────────────────────────┤
│ TypedDicts      │ ExtractionMetadata, LayerResult, RQ2Result,          │
│ (ARCH-03)       │ RQ3TrajectoryRow, PropConfig, ModelProfile           │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Category SSOT   │ categories.py: MATH_CATS, CTRL_CATS, ALL_CATS,       │
│                 │ LABEL_SENTINEL=-1                                    │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Model SSOT      │ models.py: get_model_profile() → hf_path,            │
│                 │ target_modules, extract_batch_size                   │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Comments        │ English only, inline, no change markers, no Italian  │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Git             │ Branch dev only. Never touch main. User pushes       │
│                 │ manually.                                            │
├─────────────────┼──────────────────────────────────────────────────────┤
│ transformers    │ >=4.46,<4.49 (vmap/SDPA GPT-NeoX bug)                │
└─────────────────┴──────────────────────────────────────────────────────┘

---
17. Execution Sequence

# 0. Setup
pip install -e . && pip install -r requirements.txt
pytest tests/                                        # CPU, no GPU

# 1. Dataset (already built: data/processed/dataset_master_v5.jsonl)

python -m src.dataset.build_stimuli + build_control + merge_stimuli

# 2. Base extraction (GPU)
python -m src.extraction.extract_states --config configs/config_rq2.yaml

# 3. Validate extraction
python -m src.utils.validate_configs --extraction data/processed/pythia-1.4b

# 4. RQ1 (GPU for tensor loading)
python run_rq1.py --config configs/config_rq2.yaml

# 5. RQ2 (GPU for tensor loading)
python run_rq2.py --config configs/config_rq2.yaml

# 6. Confound checks (GPU)
python -m src.probing.run_confound_checks --config configs/config_rq2.yaml
python -m src.probing.run_parity_confound_checks --config configs/config_rq2.yaml

# 7. NF4 baseline (GPU, T16)
python -m src.eval.nf4_degradation --config configs/config_rq2.yaml

# 8. Fine-tuning (GPU, ~3-4h on RTX 5080)
python -m src.finetuning.train_qlora --config configs/config_rq2.yaml --lora_config configs/lora_config.yaml

# 9. Checkpoint loop → extract + RQ3 per checkpoint (GPU)
python -m src.extraction.checkpoint_loop --config configs/config_rq2.yaml

# 10. GSM8K evaluation per checkpoint (GPU)
python -m src.eval.eval_gsm8k --model_path EleutherAI/pythia-1.4b --tag baseline --config configs/config_rq2.yaml
python -m src.eval.eval_gsm8k --model_path data/processed/checkpoints/checkpoint-500 --tag ckpt_500 --config configs/config_rq2.yaml --loading_strategy peft
# ... repeat for each checkpoint

# 11. Visualization
python -m src.viz.plot_rq1_emergence
python -m src.viz.plot_rq3_trajectory

---
18. Known Limitations (Documented)

┌──────────────────────────────┬─────────────────┬───────────────────────┐
│          Limitation          │      Type       │       Reference       │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ Single model (Pythia-1.4B)   │ Scope           │ R-I-02                │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ 3 templates, domain [10,50]  │ Scope           │ E-P-04                │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ Correlative, not causal      │ Epistemological │ E-O-01                │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ Extraction at "=" = expected │ Methodological  │ E-P-02                │
│  result, not computed        │                 │                       │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ QLoRA QKV-only, MLP frozen   │ Structural      │ RQ3 scope             │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ 0-shot GSM8K (5-15% lower    │ Comparability   │ E-F-01                │
│ than 5-shot literature)      │                 │                       │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ Positional asymmetry: math   │                 │ extract_states.py     │
│ ends in "=", ctrl ends in    │ RQ1 caveat      │ comment               │
│ word/"."                     │                 │                       │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ NF4 degradation (T16) uses   │                 │ Relative Frobenius    │
│ native HF hooks, not         │ Comparability   │ resolves scale        │
│ TransformerLens              │                 │                       │
├──────────────────────────────┼─────────────────┼───────────────────────┤
│ 500 pairs vs 1000 BLiMP      │ Sample size     │ Computational         │
│ standard                     │                 │ constraints           │
└──────────────────────────────┴─────────────────┴───────────────────────