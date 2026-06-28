# CLAUDE.md — Geometric Dynamics in Transformer Internal Representations

## Project identity
Bachelor thesis CS, Federico II Napoli, Bonagura N46007216.
Model: Pythia-1.4B (EleutherAI/pythia-1.4b), 24 layers, d_model=2048.
Hardware: RTX 5080 16GB, WSL2, conda env `transformer_thesis`.
Language: codebase English, thesis Italian. Branch: always `dev`.

## Setup
```bash
pip install -e .
pip install -r requirements.txt
# CRITICAL: transformers>=4.46,<4.49 — GPT-NeoX vmap/SDPA bug in >=4.49
```

## Key commands
```bash
# Master config is configs/config_rq2.yaml (there is no configs/config.yaml).
pytest tests/                          # CPU-only, no GPU required
python run_rq1.py --config configs/config_rq2.yaml
python run_rq2.py --config configs/config_rq2.yaml
python run_rq4.py --config configs/config_rq2.yaml --checkpoint_dir data/processed/checkpoints_extracted/<ckpt>
python -m src.extraction.extract_states                                      # GPU: base-model states
python -m src.finetuning.train_qlora                                         # GPU: QLoRA NF4 fine-tune
python -m src.extraction.checkpoint_loop --config configs/config_rq2.yaml    # GPU: merge+re-extract+run_rq4 per ckpt
python -m src.eval.nf4_degradation --config configs/config_rq2.yaml          # GPU: NF4 degradation baseline (T16)
GSM8K_BATCH_SIZE=16 python -m src.eval.eval_gsm8k --config configs/config_rq2.yaml \
    --tag <tag> --model_path <path> --loading_strategy {peft|merged_direct}  # GPU: 0-shot GSM8K
python run_rq5.py --config configs/config_rq2.yaml                            # GPU: RQ5 determinization at "="
python -m src.viz.plot_rq4_trajectory                                        # RQ4 dashboard (CPU)
python run_rq3.py --config configs/config_rq2.yaml                            # CPU: RQ3 FT geometry dynamics
python -m src.viz.plot_rq3_ft_dynamics                                       # RQ3 dashboard (CPU)
```

## Authority order (conflicts resolved top→bottom)

1. @docs/Guida_Metodologica.md — epistemological principles + overview (E-G-*, E-M-*, E-F-*, E-O-*)
2. @docs/Approccio_Architetturale.md — design hierarchy + architectural decisions (ARCH-03 active, 01/02 deferred)
3. Source code — always question choices



## Mandatory invariants — never violate
- Seeds: always `get_seed()` from `src/probing/seeds.py` — never `np.random.seed(42)` or `default_rng(42)` directly
- IO: always `_atomic_write_csv` / `_atomic_write_json` / `_atomic_save_npy` for all result writes
- Encoding: always `open(..., encoding="utf-8")`
- LoRA target modules: always from `get_model_profile()`, never hardcoded
- Comments: short, English, inline only — no Italian, no change markers (# Modifica, # Fix)
- TypedDicts: explicit contracts on all inter-module handoffs (ARCH-03)

## Git rules
- Active branch: `dev`. Never touch `main`.
- Never run `git checkout main` or `git merge` without explicit user instruction.
- Commit after every completed task with a descriptive message.

## Architecture
src/
config/     categories.py (SSOT categories), models.py (ModelProfile registry)
dataset/    build_stimuli.py, build_control.py, merge_stimuli.py, regenerate_dataset.py
extraction/ extract_states.py, checkpoint_loop.py
metrics/    cka.py, isotropy.py
probing/    seeds.py, pipeline.py, directions.py, stats.py, probing_dataset.py,
engine.py, io_utils.py, run_confound_checks.py, run_parity_confound_checks.py
finetuning/ train_qlora.py
eval/       eval_gsm8k.py, nf4_degradation.py
viz/        plot_rq1_emergence.py, plot_rq4_trajectory.py, plot_rq3_ft_dynamics.py,
pca_umap_viz.py, probing_viz.py
utils/      validate_configs.py, io_smoke_test.py
run_rq1.py, run_rq2.py, run_rq3.py, run_rq4.py, run_rq5.py   # checkpoint loop is src/extraction/checkpoint_loop.py (no root copy)
tests/test_pipeline_e2e.py

## Pipeline sequence
1. Dataset construction → `data/processed/dataset_master_v5.jsonl` (4 categories, 3000 stimuli)
2. Extraction → `data/processed/pythia-1.4b/layer_XX.pt` (FP16, [n_stimuli, 2048]) + `metadata.json`
3. RQ1 → isotropy + evolutionary CKA → `results/rq1_emergence/`
4. RQ2 → linear probing sign/parity → `results/rq2_probing/` (weights, accuracy, direction angles)
5. Fine-tuning → QLoRA NF4 on MetaMathQA → `data/processed/checkpoints/` (4 ckpts + `final_adapter`, step 12343)
6. Checkpoint loop → merge adapter → re-extract → `data/processed/checkpoints_extracted/`
7. RQ4 → frozen probe on checkpoints + dual Frobenius drift → `results/rq4_drift/trajectories_probing.csv`
8. NF4 degradation baseline (T16) → bf16-ref vs NF4 per-layer Frobenius/cosine → `results/nf4_degradation/`
9. GSM8K 0-shot eval → per checkpoint (baseline + 4 ckpts + final_adapter) → `results/gsm8k/gsm8k_<tag>.json`,
   merged into the step 7 trajectory CSV (`gsm8k_acc`, `gsm8k_ci_*`)
10. Visualization → `src/viz/` dashboards (RQ1 emergence, RQ4 trajectory) → `results/figures/`

RQ3 — FT geometry dynamics (dynamics of the RQ1 geometry across fine-tuning):
- `run_rq3.py` recomputes RQ1 geometry (ΔIso, inter-category CKA) + cross-temporal
  CKA(base→ckpt) on the extracted checkpoints → `results/rq3_ft_dynamics/rq3_dynamics.csv`
  (reuse-only: no edits to run_rq1/run_rq4/cka/isotropy). Dashboard via
  `src.viz.plot_rq3_ft_dynamics` → `results/figures/rq3/rq3_ft_dynamics.html`.
  GSM8K overlays are descriptive only (n=6); evolutionary layer-to-layer CKA out of scope.

## Data layout
data/processed/
├── dataset_master_v5.jsonl
├── pythia-1.4b/
│   ├── metadata.json
│   └── layer_00.pt … layer_23.pt
└── checkpoints_extracted/
└── checkpoint-<step>/

## Key contracts (ARCH-03)
- `ExtractionMetadata` TypedDict: `probe_strategy` and `dataset_version` are required fields
- `ModelProfile` TypedDict: required/optional split — `hf_path`, `target_modules` always required
- `PropConfig` TypedDict: `label_field` and `type` required, `category` optional
- `LayerResult` TypedDict: all metric fields explicitly typed in `engine.py`
- `rq2_config_hash.json`: saved by `run_rq2.py` after weights, verified by `run_rq4.py`

## Testing
`tests/test_pipeline_e2e.py` — full RQ1→RQ2→RQ3 on synthetic CPU data, no GPU needed.
Monkeypatches `load_hidden_states` with random tensors. All fixtures generated inline.


## Research scope — what results mean
- Correlative approach, NOT causal: we localize where information is 
  accessible, not which circuits produce it
- Results valid for: Pythia-1.4B, GPT-NeoX tokenizer, domain [10,50],
  3 syntactic templates, operators as specified per category
- Extraction at "=" token = representation of expected result, 
  not computed result (ontologically relevant distinction)
- QLoRA targets QKV only — MLP frozen — RQ3 claims are 
  scoped to attention projections, not full architecture
- 0-shot GSM8K throughout — not comparable to 5-shot literature baselines