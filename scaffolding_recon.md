# scaffolding_recon.md — READ-ONLY recon for the RQ-numbering rename

Generated on branch `scaffolding`. No files modified, nothing committed except this report.
Static analysis only (git / grep / file reads). **Mapping code→thesis is explicitly out of scope** —
this report documents the code *as it exists*.

---

## 1. Branch state

```
$ git status
On branch scaffolding
nothing to commit, working tree clean

$ git log --oneline -5
b03c7dd added orchestrator
2cdcc78 oracolo: honor --dry-run for pytest step in run_sequence (fix fork bomb)
c43da4a tests: oracolo smoke — pin registry + drift hooks (D3, D6)
d02d19b recon: resolve §8 Q4 — intentional optional override (Branch 1)
fe640e1 scripts: add oracolo.py — Il Banco della Pizia orchestrator

$ git rev-list --left-right --count dev...scaffolding
0	0
```

- **`scaffolding` is identical to `dev`** (0 ahead, 0 behind). The rename will be the first divergence.
- Local branches: `dev`, `main`, `pre-rq-compliance`, `scaffolding` (current). Remotes mirror dev/main/scaffolding.
- ⚠️ Note CLAUDE.md says "Active branch: `dev`" — recon/rename work is happening on `scaffolding`, which currently equals dev.

---

## 2. Directory tree (2 levels, key files)

```
run_rq1.py  run_rq1_dynamics.py  run_rq2.py  run_rq3.py  run_rq4.py   # 5 root orchestrators
CLAUDE.md  README.md  pyproject.toml
scripts/oracolo.py          # single-file menu orchestrator ("Il Banco della Pizia")
logs/oracolo                # runtime logs (not source)

src/
  __init__.py
  config/      categories.py  models.py  schemas.py
  dataset/     build_stimuli.py  build_control.py  merge_stimuli.py
               regenerate_dataset.py  test_dataset.py
  extraction/  extract_states.py  checkpoint_loop.py
  finetuning/  train_qlora.py
  metrics/     cka.py  isotropy.py
  probing/     engine.py  pipeline.py  directions.py  stats.py  seeds.py
               io_utils.py  probing_dataset.py
               run_confound_checks.py  run_parity_confound_checks.py
  eval/        eval_gsm8k.py  nf4_degradation.py  determinization.py
               rq3_drift_specificity.py
  viz/         plot_rq1_emergence.py  plot_rq2_probing.py  plot_rq3_trajectory.py
               plot_rq4_determinization.py  plot_ft_geometry_dynamics.py
               pca_umap_viz.py  probing_viz.py
  utils/       validate_configs.py  io_smoke_test.py

configs/   config_rq2.yaml (master)  config_template.yaml  config_test.yaml  lora_config.yaml

tests/     test_pipeline_e2e.py  test_cka_robustness.py  test_isotropy_floor.py
           test_nf4_snr.py  test_rq3_drift_specificity.py  test_viz_smoke.py
           test_oracolo_smoke.py  check_hardware.py  check_interface.py
           generate_fixtures.py

docs/      Pipeline_Dataflow.md (mermaid)  RECON.md (prior recon)  Specifica_Progetto.md
           Guida_Metodologica.md  Approccio_Architetturale.md  images/

results/   (validated artifacts — see §10)
```

> **No `*.mmd` file exists.** The data-pipeline diagram lives as an embedded mermaid `flowchart`
> inside `docs/Pipeline_Dataflow.md` (see §11). `docs/RECON.md` also embeds a mermaid block.

**Discrepancy vs CLAUDE.md:** CLAUDE.md describes a 3-RQ pipeline. The actual code has **RQ4
(`determinization`)** and several modules CLAUDE.md never mentions: `src/eval/determinization.py`,
`src/eval/rq3_drift_specificity.py`, `src/config/schemas.py`, `scripts/oracolo.py`,
`src/viz/plot_rq2_probing.py`, `src/viz/plot_rq4_determinization.py`. The rename surface is larger
than CLAUDE.md implies.

---

## 3. Entry points (runnable CLIs)

### Root orchestrators (all take `--config`, all use argparse)

| Path | Purpose (from docstring) | Inputs | Output paths |
|---|---|---|---|
| `run_rq1.py` | "RQ1 orchestrator: Emergence threshold location (l*)" — balanced isotropy, evolutionary + inter-category CKA | `--config`; reads `data/processed/`, `dataset_master_v5.jsonl` | `results/rq1_emergence/` → `isotropy_pythia.csv`, `isotropy_aggregated_balanced.csv`, `cka_results_annotated.csv`, `cka_intercategory.npy` |
| `run_rq1_dynamics.py` | "Supplementary, post-hoc exploratory" — recompute RQ1 geometry per checkpoint + cross-temporal CKA(base→ckpt) | `--config`; reads `data/processed/checkpoints_extracted/`, `results/nf4_degradation/summary.json` | `results/rq1_emergence/dynamic/rq1_dynamics.csv` |
| `run_rq2.py` | "RQ2 orchestrator: static linear probing with strict statistical gating" | `--config`; reads `data/processed/`, `dataset_master_v5.jsonl` | `results/rq2_probing/` → `accuracy_metrics_corrected.csv`, `weights/*.npy`, `weights/rq2_config_hash.json`, `test_indices/*.npy` |
| `run_rq3.py` | "RQ3 orchestrator: frozen probe evaluation on QLoRA checkpoints" | `--config`, `--checkpoint_dir` (required); reads `output_dir/weights`, `test_indices`, checkpoint states | `results/rq2_probing/dynamic/trajectories_probing.csv` (appends per step) |
| `run_rq4.py` | "RQ4 driver: behavioral determinization at the '=' token" | `--config`; reads `data/processed/checkpoints`, `dataset_master_v5.jsonl` | `results/rq4_determinization/` → `determinization.csv` + `determinization_step_*.json` |

### `python -m src...` module CLIs

| Module | Purpose | Inputs | Outputs |
|---|---|---|---|
| `src.extraction.extract_states` | activation extraction at terminal "=" token | `dataset_master_v5.jsonl` | `data/processed/pythia-1.4b/layer_XX.pt` + `metadata.json` |
| `src.extraction.checkpoint_loop` | "driver for RQ3 dynamic eval": merge LoRA → re-extract → **subprocess `run_rq3.py`** | `data/processed/checkpoints` | `data/processed/checkpoints_extracted/`, then trajectory CSV via run_rq3 |
| `src.finetuning.train_qlora` | QLoRA NF4 on MetaMathQA | `lora_config.yaml` | `data/processed/checkpoints/` (`checkpoint-<step>`, `final_checkpoint`) |
| `src.eval.eval_gsm8k` | 0-shot GSM8K eval | `--tag --model_path --loading_strategy` | `results/gsm8k/gsm8k_<tag>.json`; merges into trajectory CSV |
| `src.eval.nf4_degradation` | NF4 degradation baseline (T16) | config | `results/nf4_degradation/per_layer_stats.csv`, `summary.json`; reads trajectory CSV |
| `src.eval.determinization` | RQ4 metrics library (invoked by `run_rq4.py`) | — | (library; `RQ4DeterminizationRow`) |
| `src.eval.rq3_drift_specificity` | standalone post-RQ3 diagnostic: math-vs-ctrl drift specificity | reads trajectory CSV | `results/rq2_probing/dynamic/drift_specificity_summary.json` |
| `src.probing.run_confound_checks` | N-01 sign confound diagnostics | `output_dir/weights`, `test_indices/sign_test_idx.npy` | `results/rq2_probing/confound_checks_hardened.csv` |
| `src.probing.run_parity_confound_checks` | N-02 parity confound diagnostics | `test_indices/parity_test_idx.npy` | `results/rq2_probing/parity_confound_checks.csv` |
| `src.dataset.regenerate_dataset` | rebuild dataset; can chain extraction/RQ2 via **subprocess** | raw stimuli | `data/processed/dataset_master_v5.jsonl` |
| `src.viz.plot_rq1_emergence` | RQ1 dashboard | `results/rq1_emergence/*` | `results/figures/rq1_emergence/rq1_emergence.html` |
| `src.viz.plot_rq2_probing` | RQ2 dashboard | `results/rq2_probing` | `results/figures/rq2/*` |
| `src.viz.plot_rq3_trajectory` | RQ3 dashboard | trajectory CSV | `results/figures/rq3/rq3_dashboard.html` |
| `src.viz.plot_rq4_determinization` | RQ4 dashboard | `determinization.csv` (+ trajectory CSV for gsm8k overlay) | `results/figures/rq4/rq4_determinization.html` |
| `src.viz.plot_ft_geometry_dynamics` | supplementary dashboard | `rq1_dynamics.csv`, `cka_results_annotated.csv`, trajectory CSV, nf4 summary | `results/figures/supplementary_ft_dynamics.html` |
| `src.viz.pca_umap_viz` | PCA/UMAP viz | `data/processed/pythia-1.4b` | `results/figures/pca/*` |
| `src.utils.validate_configs` | config schema validation | config | — |
| `src.utils.io_smoke_test` | atomic-IO smoke test | — | `results/io_smoke_test` |
| `scripts/oracolo.py` | menu orchestrator cataloguing **all 28 entrypoints** (subprocess each) | menu/CLI | delegates to all of the above |

---

## 4. RENAME SURFACE

Grep over `*.py *.yaml *.yml *.md *.mmd *.toml` excluding `.git/ results/ data/ __pycache__/ logs/`.

### Per-token hit counts (whole surface incl. docs/README)

| token | hits | token | hits |
|---|---|---|---|
| `rq1` | 218 | `rq1_dynamics` | 28 |
| `rq2` | 243 | `dynamic` | 103 |
| `rq3` | 191 | `drift` | 187 |
| `rq4` | 137 | `frobenius` | 54 |
| `determinization` | 66 | `cka_intercategory` | 33 |
| `supplementary` | 25 | `cross_temporal` | 6 |
| `trajectories_probing` | 39 | | |

These tokens are pervasive — a rename is **NOT local**. The high-frequency tokens appear as: directory
names (`results/rq1_emergence`, `results/rq2_probing`, `results/rq4_determinization`), config keys
(`rq3_trajectory_csv`, `rq4_batch_size`, `rq4_output_dir`), filenames (`run_rqN.py`,
`plot_rqN_*.py`, `rq3_drift_specificity.py`), TypedDict names (`RQ3TrajectoryRow`,
`RQ4DeterminizationRow`), CSV column names (`rq3_max_relative_drift`), oracolo entrypoint keys, test
names, and docs.

### Distinct code/config files touched per token (excl. docs/README, excl. tests where noted)

- **rq1** (15 files): `configs/config_rq2.yaml`, `run_rq1.py`, `run_rq1_dynamics.py`, `scripts/oracolo.py`, `src/dataset/merge_stimuli.py`, `src/extraction/extract_states.py`, `src/metrics/cka.py`, `src/viz/pca_umap_viz.py`, `src/viz/plot_ft_geometry_dynamics.py`, `src/viz/plot_rq1_emergence.py`, + tests (`check_interface.py`, `test_cka_robustness.py`, `test_oracolo_smoke.py`, `test_pipeline_e2e.py`, `test_viz_smoke.py`).
- **rq2** (26 files): configs (`config_rq2.yaml`, `config_template.yaml`), `run_rq1.py`, `run_rq1_dynamics.py`, `run_rq2.py`, `run_rq3.py`, `scripts/oracolo.py`, `src/dataset/regenerate_dataset.py`, `src/eval/{eval_gsm8k,nf4_degradation,rq3_drift_specificity}.py`, `src/extraction/extract_states.py`, `src/finetuning/train_qlora.py`, `src/probing/{io_utils,run_confound_checks,run_parity_confound_checks,stats}.py`, `src/utils/{io_smoke_test,validate_configs}.py`, `src/viz/{plot_ft_geometry_dynamics,plot_rq2_probing,plot_rq3_trajectory,plot_rq4_determinization}.py`, + 3 tests.
- **rq3** (21 files): configs, `run_rq1_dynamics.py`, `run_rq3.py`, `run_rq4.py`, `scripts/oracolo.py`, `src/eval/{eval_gsm8k,nf4_degradation,rq3_drift_specificity}.py`, `src/extraction/{checkpoint_loop,extract_states}.py`, `src/metrics/cka.py`, `src/utils/validate_configs.py`, `src/viz/{plot_ft_geometry_dynamics,plot_rq3_trajectory,plot_rq4_determinization}.py`, + 4 tests.
- **rq4** (9 files): configs, `run_rq4.py`, `scripts/oracolo.py`, `src/eval/determinization.py`, `src/viz/plot_rq4_determinization.py`, + 3 tests.
- **determinization** (8 files): configs, `run_rq4.py`, `scripts/oracolo.py`, `src/eval/determinization.py`, `src/viz/plot_rq4_determinization.py`, + 2 tests.
- **drift** (16 files): `config_template.yaml`, `run_rq1_dynamics.py`, `run_rq3.py`, `scripts/oracolo.py`, `src/eval/{eval_gsm8k,nf4_degradation,rq3_drift_specificity}.py`, `src/extraction/extract_states.py`, `src/metrics/cka.py`, `src/probing/stats.py`, `src/viz/{plot_ft_geometry_dynamics,plot_rq3_trajectory}.py`, + 4 tests.
- **frobenius** (10 files): `run_rq1_dynamics.py`, `run_rq3.py`, `scripts/oracolo.py`, `src/eval/{nf4_degradation,rq3_drift_specificity}.py`, `src/extraction/extract_states.py`, `src/metrics/cka.py`, `src/viz/{plot_ft_geometry_dynamics,plot_rq3_trajectory}.py`, + 1 test.

### Low-frequency tokens — full file:line:text (code/config only)

**`supplementary`**
```
scripts/oracolo.py:258  Entrypoint("viz-supp", "Cruscotto supplementare", "Supplementary dashboard", CAT.VIZ,
scripts/oracolo.py:261      outputs=[R / "figures/supplementary_ft_dynamics.html"],
scripts/oracolo.py:262      description="Supplementary FT-geometry dashboard."),
src/viz/plot_ft_geometry_dynamics.py:1   docstring "Supplementary dashboard (exploratory)"
src/viz/plot_ft_geometry_dynamics.py:24  OUT_HTML = Path("results/figures/supplementary_ft_dynamics.html")
src/viz/plot_ft_geometry_dynamics.py:82,85,86  HTML title strings
run_rq1_dynamics.py:2    docstring "Supplementary, post-hoc exploratory analysis"
run_rq1_dynamics.py:64   ArgumentParser(description="Supplementary fine-tuning geometry dynamics")
```

**`trajectories_probing`** (the canonical RQ3 CSV — high blast radius)
```
configs/config_rq2.yaml:41                rq3_trajectory_csv: results/rq2_probing/dynamic/trajectories_probing.csv
run_rq3.py:26 (docstring), :163 (comment), :167  traj = dyn_dir / "trajectories_probing.csv"  (WRITER)
src/eval/eval_gsm8k.py:203                csv_path = .../trajectories_probing.csv  (READ+MERGE gsm8k cols)
src/eval/nf4_degradation.py:274           reads rq3_trajectory_csv default
src/eval/rq3_drift_specificity.py:100     reads rq3_trajectory_csv default
src/viz/plot_rq3_trajectory.py:20,26      reader
src/viz/plot_rq4_determinization.py:9,21  reader (gsm8k overlay)
src/viz/plot_ft_geometry_dynamics.py:22   reader
scripts/oracolo.py:222,250,397,400,401,678  entrypoint IO decls + drift guards
tests/test_pipeline_e2e.py:401, test_rq3_drift_specificity.py:65,103, test_viz_smoke.py:53
```

**`rq1_dynamics`**
```
run_rq1_dynamics.py:9 (docstring out path), :34 (logger name), :127 (seed purpose), :173 out_csv = out_dir / "rq1_dynamics.csv"  (WRITER)
src/viz/plot_ft_geometry_dynamics.py:5,20,41   DYN_CSV reader
scripts/oracolo.py:193,195,260                 entrypoint IO decls
```

**`cka_intercategory`**
```
src/metrics/cka.py:222 def compute_cka_intercategory, :254 _all_layers, :277,286,288,497,505, __all__ :560,561, :639,647 (WRITER cka_intercategory.npy)
run_rq1.py:24,78,86,251,255,295,299,354, :415 _atomic_save_npy(... "cka_intercategory.npy")  (WRITER)
run_rq1_dynamics.py:24,151
scripts/oracolo.py:190
tests/test_pipeline_e2e.py:220,231,251,252,253
```

**`cross_temporal`** (only in one module)
```
src/metrics/cka.py:293 def compute_cka_cross_temporal, :354 compute_cka_drift, :361,366, __all__ :562, :667
```

> Full line-level dump for every token (1356 lines) was generated transiently; the path-defining
> lines are exhaustively captured in §5 below, and the high-frequency tokens reduce to the directory
> + filename + config-key + symbol renames enumerated in this section.

---

## 5. Hardcoded `data/` and `results/` path literals (file:line)

### `results/` write/read literals
```
run_rq1.py:117                            results/rq1_emergence  (OUT_DIR, line 117/118)
run_rq1.py:179,409,415,461                isotropy_pythia.csv, cka_results_annotated.csv, cka_intercategory.npy, isotropy_aggregated_balanced.csv
run_rq1_dynamics.py:75                    results/rq1_emergence/dynamic
run_rq1_dynamics.py:186                   results/nf4_degradation/summary.json  (read)
src/metrics/isotropy.py:308              results/isotropy.csv  (legacy __main__)
src/metrics/cka.py:593,647,679           RESULTS_DIR=Path("results"); cka_intercategory.npy; cka_drift_temporal.npy  (legacy __main__)
src/probing/run_confound_checks.py:61    results/rq2_probing  (default output_dir)
src/probing/run_parity_confound_checks.py:72  results/rq2_probing
src/viz/plot_rq1_emergence.py:42,43      RESULTS_DIR / OUT_DIR
src/viz/plot_rq2_probing.py:43,81,82     results/rq2_probing, results/figures/rq2
src/viz/plot_rq3_trajectory.py:20,21     trajectories_probing.csv, results/figures/rq3
src/viz/plot_rq4_determinization.py:21,41,42  trajectory csv, determinization.csv, results/figures/rq4
src/viz/plot_ft_geometry_dynamics.py:20-24    rq1_dynamics.csv, cka_results_annotated.csv, trajectory csv, nf4 summary, supplementary html
src/viz/pca_umap_viz.py:195              results/figures/pca
src/eval/nf4_degradation.py:159,274      results/nf4_degradation; trajectory csv
src/eval/eval_gsm8k.py:132,203           results/gsm8k; trajectory csv
src/eval/rq3_drift_specificity.py:100    trajectory csv
src/utils/io_smoke_test.py:63            results/io_smoke_test
run_rq4.py:169                           rq4_output_dir default results/rq4_determinization
scripts/oracolo.py:244,397,403,660,668,673,678,689,702,716  all results/* paths (see §7)
```

### `data/` literals
```
src/extraction/extract_states.py:216,229   dataset_master_v5.jsonl, data/processed
src/extraction/checkpoint_loop.py:129,131,136  checkpoints_extracted, dataset, checkpoints
src/finetuning/train_qlora.py:71           data/processed/checkpoints (checkpoints_dir default)
src/metrics/cka.py:592,657                 data/processed, data/processed/checkpoints
src/dataset/{regenerate_dataset,build_control,build_stimuli}.py  data/processed, data/raw
run_rq1.py:115,116  run_rq2.py:116,129  run_rq3.py:85  run_rq4.py:48,49  data/processed + dataset
scripts/oracolo.py:139,140,146,163,164,169,292,403   data/raw + data/processed/checkpoints[_extracted]
src/metrics/isotropy.py:306,307            data/processed/pythia-1.4b, dataset (legacy __main__)
src/viz/pca_umap_viz.py:194                data/processed/pythia-1.4b
```

### Named path constants
- `OUT_DIR` — `run_rq1.py:117`, `src/viz/plot_rq1_emergence.py:43`
- `RESULTS_DIR` — `src/viz/plot_rq1_emergence.py:42`, `src/metrics/cka.py:593`
- `output_dir` (config-driven) — `run_rq3.py:64`, `run_confound_checks.py:61`, `run_parity_confound_checks.py:72`, `train_qlora.py:71` (`checkpoints_dir`), via `io_utils.py:71/153/159/175`
- Dynamic trajectory CSV — single canonical literal `results/rq2_probing/dynamic/trajectories_probing.csv`, repeated in 10+ files (see §4, §7).

---

## 6. Config keys + reader modules

### `configs/config_rq2.yaml` (master)

| key | value | read by (grep) |
|---|---|---|
| `model_name` | pythia-1.4b | extract_states, train_qlora, all run_rq* via get_model_profile |
| `seed` | 42 | run_rq1/2/3, get_seed callers |
| `train_split` | 0.80 | run_rq2 / probing pipeline |
| `n_permutation_tests` | 1000 | probing engine/stats |
| `bootstrap_n_samples` | 1000 | probing engine/stats |
| `bootstrap_ci` | 0.95 | probing engine |
| `max_iter` | 1000 | probing engine (LogReg) |
| `C` | 1.0 (overridden from 10.0) | probing engine |
| `solver` | lbfgs | probing engine |
| `multiclass_strategy` | ovr | probing engine |
| `n_jobs` | -1 | probing engine |
| `properties.{parity,sign}` | type/label_field/category/class_names | run_rq2, confound checkers, `PropConfig` |
| `output_dir` | results/rq2_probing | run_rq2, run_rq3, confound checkers, validate_configs |
| `figures_dir` | results/figures/rq2_probing | viz, validate_configs |
| `eval_subset_size` | 200 | eval_gsm8k |
| `total_training_steps` | 12343 | run_rq3, eval, oracolo D6 guard |
| `iso_floor_bootstrap_n` | 200 | run_rq1 (isotropy floor) |
| `rq3_trajectory_csv` | results/rq2_probing/dynamic/trajectories_probing.csv | run_rq3, eval_gsm8k, nf4_degradation, rq3_drift_specificity (via `.get`) |
| `rq4_batch_size` | 32 | run_rq4 (via `.get`) |
| `rq4_output_dir` | results/rq4_determinization | run_rq4 (via `.get`) |

### `configs/lora_config.yaml`

| key | value | read by |
|---|---|---|
| `model_name` | pythia-1.4b | train_qlora |
| `r` | 16 | train_qlora (LoRA) |
| `lora_alpha` | 32 | train_qlora |
| `lora_dropout` | 0.1 | train_qlora |
| `bits` | 4 | train_qlora (BitsAndBytes) |
| `double_quant` | true | train_qlora |
| `quant_type` | nf4 | train_qlora, nf4_degradation |
| `learning_rate` | 2.0e-4 | train_qlora |
| `batch_size` | 8 | train_qlora |
| `gradient_accumulation` | 4 | train_qlora |
| `num_epochs` | 1 | train_qlora |
| `warmup_ratio` | 0.03 | train_qlora |
| `lr_scheduler` | cosine | train_qlora |
| `save_steps` | 500 | train_qlora |
| `max_seq_length` | 512 | train_qlora, oracolo D3 drift guard |

Other configs: `config_template.yaml` (documents rq2/rq3/rq4/drift keys), `config_test.yaml` (test fixture).

---

## 7. Data-flow contracts (writer → file → reader) + subprocess flags

| Result file | Writer | Reader(s) |
|---|---|---|
| `rq1_emergence/isotropy_pythia.csv` | run_rq1 | plot_rq1_emergence |
| `rq1_emergence/isotropy_aggregated_balanced.csv` | run_rq1 | plot_rq1_emergence; oracolo (drift checks) |
| `rq1_emergence/cka_results_annotated.csv` | run_rq1 | plot_rq1_emergence; plot_ft_geometry_dynamics; oracolo |
| `rq1_emergence/cka_intercategory.npy` | run_rq1 (cka.py) | test_pipeline_e2e |
| `rq1_emergence/dynamic/rq1_dynamics.csv` | run_rq1_dynamics | plot_ft_geometry_dynamics |
| `rq2_probing/accuracy_metrics_corrected.csv` | run_rq2 | oracolo; plot_rq2_probing |
| `rq2_probing/weights/*.npy` + `rq2_config_hash.json` | run_rq2 | run_rq3 (hash-verified), confound checkers |
| `rq2_probing/test_indices/{sign,parity}_test_idx.npy` | run_rq2 | run_rq3, confound checkers |
| `rq2_probing/confound_checks_hardened.csv` | run_confound_checks | plot_rq2_probing; oracolo |
| `rq2_probing/parity_confound_checks.csv` | run_parity_confound_checks | plot_rq2_probing |
| **`rq2_probing/dynamic/trajectories_probing.csv`** | run_rq3 (writer) | eval_gsm8k (merges gsm8k cols), nf4_degradation, rq3_drift_specificity, plot_rq3_trajectory, plot_rq4_determinization, plot_ft_geometry_dynamics |
| `rq2_probing/dynamic/drift_specificity_summary.json` | rq3_drift_specificity | (terminal) |
| `rq4_determinization/determinization.csv` + `_step_*.json` | run_rq4 (determinization.py) | plot_rq4_determinization; oracolo |
| `nf4_degradation/{per_layer_stats.csv,summary.json}` | nf4_degradation | run_rq1_dynamics, plot_ft_geometry_dynamics, plot_rq3_trajectory, test_nf4_snr |
| `gsm8k/gsm8k_*.json` | eval_gsm8k | (merged into trajectory CSV); oracolo |

**Subprocess invocations (rename-sensitive — string-built commands):**
- `src/extraction/checkpoint_loop.py:91-96` → `[sys.executable, "run_rq3.py", "--config", ..., "--checkpoint_dir", ...]`, run with `timeout=3600` (line 100).
- `src/dataset/regenerate_dataset.py:109` → `-m src.dataset.merge_stimuli`; `:189` → `run_rq2.py`; `:192` → `-m src.probing.run_confound_checks`; `:194` → `-m src.probing.run_parity_confound_checks`.
- `scripts/oracolo.py` — builds and `subprocess.Popen`/`run` for **all 28 entrypoints** (PY=sys.executable). Entrypoint registry hardcodes script names + result paths at lines ~139-292 (inputs/outputs) and drift-guard checks at 397-716. **This file is the single largest rename hotspot.**

---

## 8. Tests (assertions + rename/path hardcoding)

| File | Test fn | Asserts | rq/path hardcode |
|---|---|---|---|
| `test_pipeline_e2e.py` | `test_rq1_pipeline` | run_rq1 produces cka csv/npy + balanced iso, deterministic CKA | ⚠ `results/rq1_emergence`, `run_rq1.*` patches |
| | `test_rq2_pipeline` | run_rq2 writes accuracy_metrics_corrected.csv | ⚠ `results/rq2_probing`, output_dir |
| | `test_category_filter_invariant` | category filtering stable | — |
| | `test_propconfig_single_source_of_truth` | PropConfig SSOT | — |
| | `test_validator_category_guardrail_intact` | validate_configs guardrail | — |
| | `test_propconfig_runtime_tolerates_missing_category` | optional category | — |
| | `test_rq3_pipeline` | run_rq3 writes trajectory CSV | ⚠ `dynamic/trajectories_probing.csv` |
| | `test_rq4_pipeline` + `test_rq4_*` (metric/build_targets/extract_eq_logits/single_restricted) | RQ4 determinization metrics | ⚠ rq4 names |
| | `test_probing_algebra`, `test_isotropy_sign_convention` | numeric invariants | — |
| `test_cka_robustness.py` | debiased self-CKA=1, biased proximity, small-n raise, Procrustes rotation/unrelated, leave-k-out contract | — | imports cka |
| `test_isotropy_floor.py` | floor≈0, CI brackets mean | — |
| `test_nf4_snr.py` | SNR present/below-floor/missing-csv/zero-floor | ⚠ key `rq3_max_relative_drift` |
| `test_rq3_drift_specificity.py` | 4 verdict ladders + dedup/summary + missing-step | ⚠ `from src.eval.rq3_drift_specificity`, `trajectories_probing.csv` |
| `test_viz_smoke.py` | rq1/rq3/rq4/rq2-accuracy/pca/rq2-confound dashboards render | ⚠ rq names + `trajectories_probing.csv` |
| `test_oracolo_smoke.py` | registry has all 28 keys; every entrypoint dry-runs; composite rites; unknown key fails; drift D3/D6; dry-run skips pytest | ⚠ **hardcodes 28 entrypoint keys incl. `rq1,rq2,rq3,rq4,rq1-dyn,viz-rq1..4,solo_rq4`** (lines 44-53); `configs/config_rq2.yaml` |
| non-test helpers | `check_hardware.py`, `check_interface.py`, `generate_fixtures.py` | hardware/interface/fixtures | check_interface refs rq1 |

---

## 9. SSOT + contracts

| Item | Location |
|---|---|
| Categories SSOT | `src/config/categories.py` — `MATH_CATS`, `CTRL_CATS`, `ALL_CATS`, `PROBE_PROPERTIES`, `LABEL_SENTINEL=-1` |
| Model registry | `src/config/models.py` — `ModelProfile` / `ModelProfileOptional` TypedDict + `get_model_profile()` |
| Seeds | `src/probing/seeds.py` — `get_seed(base_seed, purpose, offset=0)` |
| `PropConfig` / `_PropConfigOptional` | `src/config/schemas.py:12-22` |
| `ModelProfile` / `ModelProfileOptional` | `src/config/models.py:8-12` (required: `hf_path`, `target_modules`, `extract_batch_size`, `needs_pad_token_fix`) |
| `ExtractionMetadata` / `ExtractionMetadataLabels` | `src/extraction/extract_states.py:33,26` (required `probe_strategy`, `dataset_version`) |
| `LayerResult` | `src/probing/engine.py:15` |
| `RQ3TrajectoryRow` | `run_rq3.py:25` → fields: step, layer, property, probing_acc, geom_delta_math/ctrl(+_rel) |
| `RQ4DeterminizationRow` | `src/eval/determinization.py:29` |
| `LayerDriftRow` | `src/eval/rq3_drift_specificity.py:39` |

> The prompt's "RQ2Result" TypedDict does not exist by that name; the RQ2 contract is `LayerResult`
> (engine.py). CSV column contracts (see §10) are an additional implicit rename surface.

---

## 10. Validated artifacts on disk (PRESERVE by migration — never recompute)

Total `results/` = ~30.8 MB (29.9 MB is HTML figures). Cuomo-validated computed outputs:

**CSV / JSON / NPY (the real data — small, must be migrated):**
```
rq1_emergence/cka_results_annotated.csv            3233 B
rq1_emergence/cka_intercategory.npy                 320 B
rq1_emergence/isotropy_pythia.csv                  9725 B
rq1_emergence/isotropy_aggregated_balanced.csv     2406 B
rq1_emergence/dynamic/rq1_dynamics.csv            12838 B
rq2_probing/accuracy_metrics_corrected.csv         3026 B
rq2_probing/accuracy_matrix_C10_backup.csv         1176 B
rq2_probing/confound_checks_hardened.csv           3212 B
rq2_probing/parity_confound_checks.csv             3639 B
rq2_probing/weights/rq2_config_hash.json            885 B
rq2_probing/weights/layer_00..23_{sign,parity}.npy  16512 B each (96 files) + *_bias.npy 136 B each
rq2_probing/test_indices/{sign,parity}_test_idx.npy 1728 B each
rq2_probing/dynamic/trajectories_probing.csv      23588 B   ← UNIFIED RQ3 CSV (12-col w/ gsm8k)
rq2_probing/dynamic/drift_specificity_summary.json 3631 B
rq2_sweep/C_{0.01,0.1,1.0,10.0}/accuracy_matrix.csv ~1.2 KB each
rq4_determinization/determinization.csv            1293 B
rq4_determinization/determinization_step_{0,2500,5000,7500,10000,12343}.json ~830 B each
nf4_degradation/per_layer_stats.csv                 966 B
nf4_degradation/summary.json                        519 B
gsm8k/gsm8k_{baseline,ckpt_2500,5000,7500,10000,final_adapter}.json ~290 B each
```

**HTML/PNG figures (regenerable from CSVs, but currently present):**
```
figures/rq1_emergence/rq1_emergence.html  figures/rq2/{accuracy_curves,confound_effect_vs_significance}.{html,png}
figures/rq3/rq3_dashboard.html  figures/rq4/rq4_determinization.html
figures/supplementary_ft_dynamics.html  figures/pca/pca_{2class,4way}_layer_23.{html,png}
```

**CSV header contracts (rename must keep columns or migrate):**
- `trajectories_probing.csv`: `step,layer,property,probing_acc,geom_delta_math,geom_delta_ctrl,geom_delta_math_rel,geom_delta_ctrl_rel,gsm8k_acc,gsm8k_ci_lower,gsm8k_ci_upper`
- `determinization.csv`: `step,category,n_rows,n_single_token,entropy_mean,margin_mean,p_first_token_mean,p_correct_single,p_correct_single_ci_lo,p_correct_single_ci_hi,entropy_mean_single,margin_mean_single`

> Log files present but disposable: `probing.log(.bak)`, `run_20260530_1705.log`, `nf4_degradation/probing.log`.

---

## 11. Current data-pipeline diagram

- **No `.mmd` file exists.** The pipeline diagram is an embedded mermaid `flowchart LR` in
  **`docs/Pipeline_Dataflow.md`** (lines 9-124).
- Subgraph/RQ labels in that flowchart:
  - `RQ1` — "emergence geometry (two distinct metrics)" → ISO (`isotropy_aggregated_balanced.csv`), CKA (`cka_results_annotated.csv`).
  - `RQ2` — "linear probing (two probes, two confounds)" → SIGN/PAR (`accuracy_metrics_corrected.csv`), SCONF (`confound_checks_hardened.csv`), PCONF (`parity_confound_checks.csv`), WTS (`weights/*.npy`).
  - `RQ3` — "fine-tuning dynamics (accuracy + drift)" → R3ACC + R3DRIFT (`dynamic/trajectories_probing.csv`).
  - `RQ4` — "behavioral determinization at '=' (inference-only)" → R4DET + R4SINGLE (`rq4_determinization/determinization.csv`).
  - `SUPP` — "Supplementary — FT geometry dynamics" → SUPPCSV (`rq1_emergence/dynamic/rq1_dynamics.csv`).
  - Plus shared nodes NF4 (T16), GSM8K, TRAJ (unified CSV), VIZ1-4 + SUPPV.
- `docs/RECON.md` is a **prior recon report** (mermaid at line 111+) and already discusses the rq1_dynamics/RQ4 ambiguity — useful prior art for the next step.

---

## 12. Hazards (what makes the rename non-local)

1. **RQ3 data lives under `results/rq2_probing/dynamic/`**, not `results/rq3*/`. The unified
   trajectory CSV, drift summary, and RQ3 logs are all nested inside the RQ2 directory. A
   semantically clean rename of RQ3 would have to either relocate this dir (breaking 10+ readers +
   the `rq3_trajectory_csv` config key + the `output_dir / "dynamic"` derivation in `run_rq3.py:165`)
   or deliberately leave RQ3 output under the RQ2 tree. **This is the single biggest collision.**
2. **Supplementary output lives under `results/rq1_emergence/dynamic/`** (`rq1_dynamics.csv`) — same
   pattern: RQ-labelled subdir hosting a different analysis.
3. **`config_rq2.yaml` is the master config for ALL of RQ1/RQ2/RQ3/RQ4** (despite the rq2 name).
   Renaming the file breaks every `--config configs/config_rq2.yaml` invocation across docs, oracolo,
   tests (`test_oracolo_smoke.py:41`), and CLAUDE.md key commands.
4. **`scripts/oracolo.py`** hardcodes a 28-entry registry of script names + input/output paths +
   drift guards; **`tests/test_oracolo_smoke.py` asserts the exact 28 keys** (incl. `rq1,rq2,rq3,rq4,
   rq1-dyn,viz-rq1..4,solo_rq4`). Any rename of entrypoints/paths must update both in lockstep.
5. **TypedDict + CSV-column names encode RQ numbers** (`RQ3TrajectoryRow`, `RQ4DeterminizationRow`,
   `rq3_max_relative_drift` key, `geom_delta_*` columns) — symbol renames, not just file renames.
6. **`run_rq3.py` derives paths from `output_dir`** (config `output_dir=results/rq2_probing`) →
   `output_dir / "dynamic"`, `output_dir / "weights"`, `output_dir / "test_indices"`. RQ3 has **no
   directory of its own**; it is structurally a sub-analysis of RQ2's output_dir.
7. **Legacy `__main__` blocks** in `src/metrics/cka.py` (writes `results/cka_intercategory.npy`,
   `results/cka_drift_temporal.npy`) and `src/metrics/isotropy.py` (`results/isotropy.csv`) write to
   non-namespaced paths — stale/duplicate path literals to reconcile.
8. **CLAUDE.md is stale**: it documents a 3-RQ pipeline and omits RQ4, determinization, oracolo,
   rq3_drift_specificity, schemas.py. The rename must also bring CLAUDE.md into truth.
9. **`checkpoint_loop.py` shells out to `run_rq3.py` by literal filename** — renaming `run_rq3.py`
   breaks this subprocess silently (only caught at runtime).

---

## Current code RQ semantics (AS THEY EXIST IN CODE — no thesis mapping)

| identifier | what it computes | result paths |
|---|---|---|
| **rq1** | `run_rq1.py` — category-balanced isotropy (ΔIso), evolutionary CKA(l vs l−1) + inter-category CKA(math↔ctrl) with debiased/Procrustes/leave-k-out/matched-terminal robustness | `results/rq1_emergence/{isotropy_pythia.csv, isotropy_aggregated_balanced.csv, cka_results_annotated.csv, cka_intercategory.npy}` |
| **rq2** | `run_rq2.py` — static linear probing (sign, parity) with bootstrap CI + permutation test + BH; frozen weights/bias + test indices + config hash; N-01/N-02 confound checkers | `results/rq2_probing/{accuracy_metrics_corrected.csv, weights/*.npy, weights/rq2_config_hash.json, test_indices/*.npy, confound_checks_hardened.csv, parity_confound_checks.csv}` |
| **rq3** | `run_rq3.py` (+ checkpoint_loop driver) — frozen-probe accuracy on QLoRA checkpoints + dual Frobenius drift (dim-normalized + relative, math vs ctrl); gsm8k merged in; rq3_drift_specificity verdict | `results/rq2_probing/dynamic/{trajectories_probing.csv, drift_specificity_summary.json}` (nested under RQ2!) |
| **rq4** | `run_rq4.py` (+ determinization.py) — inference-only "=" next-token determinization per (step, category): entropy↓, top1−top2 margin↑, P(answer)↑, single-token-restricted subset | `results/rq4_determinization/{determinization.csv, determinization_step_*.json}` |
| **supplementary** | `run_rq1_dynamics.py` — recomputes RQ1 geometry (ΔIso, inter-cat CKA) + cross-temporal CKA(base→ckpt) on extracted checkpoints; explicitly exploratory, not an RQ | `results/rq1_emergence/dynamic/rq1_dynamics.csv` → `results/figures/supplementary_ft_dynamics.html` |
