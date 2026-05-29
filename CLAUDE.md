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
pytest tests/                          # CPU-only, no GPU required
python run_rq1.py --config configs/config.yaml
python run_rq2.py --config configs/config.yaml
python run_rq3.py --config configs/config.yaml --checkpoint_dir 
python -m src.extraction.extract_states   # GPU required
python -m src.finetuning.train_qlora      # GPU required
```

## Authority order (conflicts resolved top→bottom)

1. @README_metodologico - overview and direction
2. @HO_metodologico.md — epistemological principles (E-G-*, E-M-*, E-F-*, E-O-*)
3. @Comandamenti — design hierarchy (O-*, S-*, C-*, A-*, B-*)
4. @Approccio_architetturale — architectural decisions (ARCH-03 active, 01/02 deferred)
5. Source code — always question choices



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
dataset/    build_stimuli.py, build_control.py, merge_stimuli.py
extraction/ extract_states.py, checkpoint_loop.py
metrics/    cka.py, isotropy.py
probing/    seeds.py, pipeline.py, directions.py, stats.py,
probing_dataset.py, engine.py, io_utils.py, run_confound_checks.py
finetuning/ train_qlora.py
eval/       eval_gsm8k.py, nf4_degradation.py
utils/      validate_configs.py, io_smoke_test.py
run_rq1.py, run_rq2.py, run_rq3.py, checkpoint_loop.py
tests/test_pipeline_e2e.py

## Pipeline sequence
1. Dataset construction → `data/processed/dataset_master_v5.jsonl` (4 categories, 3000 stimuli)
2. Extraction → `data/processed/pythia-1.4b/layer_XX.pt` (FP16, [n_stimuli, 2048]) + `metadata.json`
3. RQ1 → isotropy + evolutionary CKA → `results/rq1_emergence/`
4. RQ2 → linear probing sign/parity → `results/rq2_probing/` (weights, accuracy, direction angles)
5. Fine-tuning → QLoRA NF4 on MetaMathQA → `data/processed/checkpoints/`
6. Checkpoint loop → merge adapter → re-extract → `data/processed/checkpoints_extracted/`
7. RQ3 → frozen probe on checkpoints + Frobenius drift → `results/rq2_probing/dynamic/`

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
- `rq2_config_hash.json`: saved by `run_rq2.py` after weights, verified by `run_rq3.py`

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