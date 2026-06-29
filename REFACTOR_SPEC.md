# REFACTOR_SPEC.md — Align codebase RQ numbering to thesis RQ numbering

**Branch:** `scaffolding` only. Never `dev`/`main`. The student pushes manually.
**Nature:** semantic rename only. No logic, numbers, seeds, or results change. RQ1 and RQ2 stay frozen.
**If a logic bug is found:** report it separately, do not fix it here.
**Authority:** the student approves the flagged decisions in §9 before this runs.

---

## 1. Verified mapping (code → thesis), re-confirmed against the `.tex`

| Thesis (in `.tex`, do NOT renumber) | Code today | Becomes |
|---|---|---|
| RQ1 — Geometric Emergence (static geometry) | `rq1` (`run_rq1`, `results/rq1_emergence`) | **unchanged** |
| RQ2 — Linear Decodability of Sign & Parity (probing) | `rq2` (`run_rq2`, `results/rq2_probing`) | **unchanged** |
| RQ3 — Dynamics of the RQ1 Geometry Across FT | code **"supplementary"** (`run_rq1_dynamics`, `plot_ft_geometry_dynamics`; recomputes RQ1 CKA + $\Delta$Iso per checkpoint) | **rq3** |
| RQ4 — Geometric Reorganization (Frobenius drift + frozen-probe decay + GSM8K + NF4) | code **`rq3`** (`run_rq3`, `checkpoint_loop` driver, `rq3_drift_specificity`, `trajectories_probing.csv`) | **rq4** |
| RQ5 — Behavioral Determinization at `=` | code **`rq4`** (`run_rq4`, `determinization.py`, `results/rq4_determinization`) | **rq5** |

Net renames: `rq4 → rq5`, `rq3 → rq4`, `supplementary → rq3`.
**`rq3` and `rq4` both already exist**, so a naive rename clobbers. Collision-safe order is enforced by the commit plan (§8): **rq4→rq5 first, then rq3→rq4, then supplementary→rq3.** This ordering holds at every level: filenames, directories, symbols, config keys, oracolo keys, test functions, and figure dirs.

---

## 2. Invariants to preserve (do NOT touch)

- **SSOT:** `src/config/categories.py`, `src/config/models.py` (`ModelProfile`, `target_modules=["query_key_value"]`), `src/probing/seeds.py` (`get_seed`).
- **Contracts:** `src/config/schemas.py` (`PropConfig`), `src/probing/engine.py` (`LayerResult`), `src/extraction/extract_states.py` (`ExtractionMetadata`). (Note: the contract type is `LayerResult`, there is no `RQ2Result`.)
- **Determinism:** `get_seed(base_seed, purpose, offset)` discipline; atomic-write helpers (`_atomic_save_npy`, etc.).
- **Env pin:** `transformers>=4.46,<4.49`.
- Comments architectural, English only.

---

## 3. FROZEN tokens (the determinism / data-contract carve-out) — READ BEFORE RENAMING

Two classes of token contain old RQ numbers but **must NOT be renamed**, because renaming them would change a seed or mutate a preserved artifact. They are carved out of the grep gate (§ test plan T3) with an inline comment `# frozen: <reason>, not an RQ label` and an explicit file:line entry in the final report.

1. **`get_seed(...)` purpose strings.** Any `purpose=` literal passed to `get_seed` (e.g. the one at `run_rq1_dynamics.py:127`). Changing the string changes the derived seed, which violates "no seeds change." **Freeze every seed-purpose literal**, even if it reads `rq1_dynamics`/`rq3`/`rq4`. First action: grep all `get_seed(` call sites, list every purpose literal, freeze those containing lineage tokens.

2. **Serialized keys inside preserved artifacts.** A key that is written *into* a validated file under `results/` cannot be renamed without changing that file's bytes (and the no-recompute hash test, T6). The known candidate is **`rq3_max_relative_drift`** (asserted in `test_nf4_snr.py`).
   - **Action:** check whether `rq3_max_relative_drift` appears *inside* a preserved file (likely `results/nf4_degradation/summary.json`).
     - If it is **not** serialized (only a code/test identifier) → rename to `rq4_max_relative_drift` normally (it joins the `rq3→rq4` group).
     - If it **is** serialized → **freeze it** (keep the literal in the writer/reader/test, comment it, whitelist it), so `summary.json` stays byte-identical. Report this as a resolved data-contract conflict.

CSV **column** names (`geom_delta_math/ctrl(_rel)`, `gsm8k_acc`, `gsm8k_ci_*`, determinization columns) contain no RQ number and are frozen data contracts: leave them.

---

## 4. Exhaustive old → new rename table

Bare `rq1` (218 hits) and `rq2` (243 hits) are **mostly legitimate and stay**. Only match the **compound lineage tokens** below. Never run a blind `rq1→` or `rq2→` substitution.

### 4.1 Root entry points
| old | new |
|---|---|
| `run_rq4.py` | `run_rq5.py` |
| `run_rq3.py` | `run_rq4.py` |
| `run_rq1_dynamics.py` | `run_rq3.py` |
| `run_rq1.py`, `run_rq2.py` | unchanged |

### 4.2 `src/` modules (RQ-numbered filenames)
| old | new |
|---|---|
| `src/eval/rq3_drift_specificity.py` | `src/eval/rq4_drift_specificity.py` |
| `src/viz/plot_rq3_trajectory.py` | `src/viz/plot_rq4_trajectory.py` |
| `src/viz/plot_rq4_determinization.py` | `src/viz/plot_rq5_determinization.py` |
| `src/viz/plot_ft_geometry_dynamics.py` | `src/viz/plot_rq3_ft_dynamics.py` *(approved §9.3)* |
| `src/eval/determinization.py` | unchanged filename (descriptive); symbol inside renames (§4.5) |
| `src/eval/eval_gsm8k.py`, `src/eval/nf4_degradation.py` | unchanged filenames (shared); internal `RQ3`/`RQ4` docstrings/comments + path literals repoint |
| `src/extraction/checkpoint_loop.py` | unchanged filename; subprocess target + docstring update (§4.7) |
| `src/viz/plot_rq1_emergence.py`, `plot_rq2_probing.py`, `src/probing/run_confound_checks.py`, `run_parity_confound_checks.py` | unchanged (RQ1/RQ2) |

### 4.3 Result directories + data files (git mv only — see §5)
| old | new |
|---|---|
| `results/rq4_determinization/` (+ `determinization.csv`, `determinization_step_*.json`) | `results/rq5_determinization/` (contents move with dir) |
| `results/rq2_probing/dynamic/trajectories_probing.csv` | `results/rq4_drift/trajectories_probing.csv` |
| `results/rq2_probing/dynamic/drift_specificity_summary.json` | `results/rq4_drift/drift_specificity_summary.json` |
| `results/rq1_emergence/dynamic/rq1_dynamics.csv` | `results/rq3_ft_dynamics/rq3_dynamics.csv` *(approved §9.2)* |
| emptied `results/rq2_probing/dynamic/`, `results/rq1_emergence/dynamic/` | remove |
| `results/rq1_emergence/*` (static), `results/rq2_probing/*` (minus migrated `dynamic/`) | **untouched** |

New dirs to create: `results/rq3_ft_dynamics/`, `results/rq4_drift/`.

### 4.4 Figure dirs/files (git mv; regenerable, lower criticality)
| old | new |
|---|---|
| `results/figures/rq4/rq4_determinization.html` | `results/figures/rq5/rq5_determinization.html` |
| `results/figures/rq3/rq3_dashboard.html` | `results/figures/rq4/rq4_dashboard.html` |
| `results/figures/supplementary_ft_dynamics.html` | `results/figures/rq3/rq3_ft_dynamics.html` *(approved §9.3)* |
| `results/figures/rq1_emergence/`, `figures/rq2/`, `figures/pca/` | unchanged |

### 4.5 Symbols (classes / TypedDicts / keys)
| old | new |
|---|---|
| `RQ3TrajectoryRow` (`run_rq3.py:25`) | `RQ4TrajectoryRow` |
| `RQ4DeterminizationRow` (`src/eval/determinization.py:29`) | `RQ5DeterminizationRow` |
| `LayerDriftRow` (`rq3_drift_specificity.py:39`) | unchanged (no RQ number) |
| `rq3_max_relative_drift` | `rq4_max_relative_drift` **iff not serialized** (else freeze, §3) |

### 4.6 Config keys (`configs/config_rq2.yaml`, `config_template.yaml`)
| old key | new key | value change |
|---|---|---|
| `rq3_trajectory_csv` | `rq4_trajectory_csv` | `results/rq2_probing/dynamic/trajectories_probing.csv` → `results/rq4_drift/trajectories_probing.csv` |
| `rq4_batch_size` | `rq5_batch_size` | — |
| `rq4_output_dir` | `rq5_output_dir` | `results/rq4_determinization` → `results/rq5_determinization` |
| `output_dir` (= `results/rq2_probing`), `figures_dir` | unchanged (RQ2-owned) |

**Config file name `config_rq2.yaml`: NOT renamed in this refactor** (approved §9.4 — master config; `rq2` is an aligned number).

### 4.7 Subprocess targets (string-built commands)
- `src/extraction/checkpoint_loop.py:91-96` shells `run_rq3.py` → change literal to **`run_rq4.py`** (verified, §9.5); `timeout=3600` untouched; docstring "driver for RQ3 dynamic eval" → "RQ4".
- `src/dataset/regenerate_dataset.py:189` shells `run_rq2.py` (unchanged), `:192/:194` confound checkers (unchanged), `:109` `merge_stimuli` (unchanged). No change unless a lineage token appears.
- `scripts/oracolo.py` — subprocess for all 28 entrypoints; registry + path decls + drift guards update per §4.8/§4.3/§4.4.

### 4.8 Oracolo registry keys (`scripts/oracolo.py`) + `tests/test_oracolo_smoke.py` (lockstep)
Count stays 28; only relabel.
| old key | new key |
|---|---|
| `rq3` | `rq4` |
| `rq4` | `rq5` |
| `rq1-dyn` | `rq3` |
| `viz-rq3` | `viz-rq4` |
| `viz-rq4` | `viz-rq5` |
| `viz-supp` | `viz-rq3` |
| `solo_rq4` | `solo_rq5` |
| `rq1`, `rq2`, `viz-rq1`, `viz-rq2`, + the ~17 non-RQ keys | unchanged |
Update each entrypoint's input/output **path decls** and **drift guards** (lines ~139–292, ~397–716) to the new result/figure paths. `test_oracolo_smoke.py:44-53` asserts the exact key set → update it inside the same commit as the registry change so the suite stays green per commit.

### 4.9 Tests
| file | change |
|---|---|
| `tests/test_rq3_drift_specificity.py` | → `tests/test_rq4_drift_specificity.py`; `from src.eval.rq3_drift_specificity` → `rq4_drift_specificity`; `trajectories_probing.csv` path → `results/rq4_drift/...` |
| `tests/test_pipeline_e2e.py` | `test_rq3_pipeline` → `test_rq4_pipeline`; `test_rq4_*` (`_pipeline`, `_metric`, `_build_targets`, `_extract_eq_logits`, `_single_restricted`) → `test_rq5_*`; path asserts (`dynamic/trajectories_probing.csv` → `rq4_drift/...`; `rq4_determinization` → `rq5_determinization`). Function-rename order inside the file: rq4→rq5 first, then rq3→rq4. RQ1/RQ2 tests untouched. |
| `tests/test_nf4_snr.py` | key `rq3_max_relative_drift` → freeze-aware (§3) |
| `tests/test_viz_smoke.py` | rq3/rq4 dashboard names + `trajectories_probing.csv` path |
| `tests/test_oracolo_smoke.py` | 28-key set (§4.8); `configs/config_rq2.yaml` path unchanged |
| `test_cka_robustness.py`, `test_isotropy_floor.py`, `check_interface.py` | unchanged (RQ1/RQ2 / `rq1` bare) |

### 4.10 Docs (Commit 4)
- `docs/Pipeline_Dataflow.md` embedded mermaid (lines 9-124): subgraph relabel `SUPP→RQ3`, `RQ3→RQ4`, `RQ4→RQ5`; node path literals repointed (`dynamic/trajectories_probing.csv` → `rq4_drift/...`; `rq1_emergence/dynamic/rq1_dynamics.csv` → `rq3_ft_dynamics/rq3_dynamics.csv`; `rq4_determinization` → `rq5_determinization`; `supplementary_ft_dynamics.html` → `figures/rq3/rq3_ft_dynamics.html`). *(The authoritative standalone `.mmd` is a later deliverable; this embedded copy is made truthful.)*
- `docs/Specifica_Progetto.md`, `Guida_Metodologica.md`, `Approccio_Architetturale.md`, `README.md`: RQ-number refs where they label the renamed experiments.
- `CLAUDE.md`: update slash-command / skill / key-command RQ refs only. **Do NOT** rewrite its stale 3-RQ description here (report separately, §10).
- `docs/RECON.md`: leave as historical prior-art; exclude from the grep gate.

### 4.11 The "supplementary" word (25 hits)
Hard tokens (`viz-supp` key, `supplementary_ft_dynamics.html`, `run_rq1_dynamics` docstrings/argparse description) rename to the RQ3 lineage. Per §9.7, the "supplementary/exploratory" docstrings are reframed fully to "RQ3 (FT geometry dynamics)".

---

## 5. Results migration map (git mv only — NEVER recompute)

The two nested migrations are the highest-risk step. **Migrating RQ3/RQ4 content must not disturb the RQ1/RQ2 artifacts the dirs also hold.**

```
# RQ5 (whole dir)
git mv results/rq4_determinization            results/rq5_determinization

# RQ4 drift (out of the RQ2 tree, into a new dir)
mkdir -p results/rq4_drift
git mv results/rq2_probing/dynamic/trajectories_probing.csv      results/rq4_drift/trajectories_probing.csv
git mv results/rq2_probing/dynamic/drift_specificity_summary.json results/rq4_drift/drift_specificity_summary.json
rmdir  results/rq2_probing/dynamic            # must now be empty

# RQ3 FT-dynamics (out of the RQ1 tree, into a new dir)
mkdir -p results/rq3_ft_dynamics
git mv results/rq1_emergence/dynamic/rq1_dynamics.csv  results/rq3_ft_dynamics/rq3_dynamics.csv   # approved §9.2: rename
rmdir  results/rq1_emergence/dynamic          # must now be empty

# figures (regenerable; git mv preserves bytes)
git mv results/figures/rq4 results/figures/rq5 && git mv results/figures/rq5/rq4_determinization.html results/figures/rq5/rq5_determinization.html
git mv results/figures/rq3 results/figures/rq4 && git mv results/figures/rq4/rq3_dashboard.html        results/figures/rq4/rq4_dashboard.html
mkdir -p results/figures/rq3 && git mv results/figures/supplementary_ft_dynamics.html results/figures/rq3/rq3_ft_dynamics.html
```

**Writer/reader repointing that accompanies the migration:**

- **`run_rq4.py` (was `run_rq3.py`) write-path decoupling.** It currently derives `dyn_dir = output_dir / "dynamic"` from `output_dir = results/rq2_probing` (`:165-167`). Change the **write** base to the `rq4_trajectory_csv` directory (`results/rq4_drift`). **Keep** the **reads** of `output_dir/"weights"`, `output_dir/"test_indices"`, and the `rq2_config_hash.json` verification from `results/rq2_probing` (these are RQ2 inputs RQ4 legitimately consumes).
- **`run_rq3.py` (was `run_rq1_dynamics.py`) write-path decoupling.** Change the **write** base from `results/rq1_emergence/dynamic` (`:75`) and `out_csv` (`:173`) to `results/rq3_ft_dynamics/rq3_dynamics.csv`. **Keep** the **reads** of `results/rq1_emergence/cka_results_annotated.csv` and `results/nf4_degradation/summary.json` and the trajectory CSV (now `results/rq4_drift/...`).
- **6 trajectory-CSV readers** repoint to `results/rq4_drift/trajectories_probing.csv` via the renamed `rq4_trajectory_csv` key: `eval_gsm8k.py:203`, `nf4_degradation.py:274`, `rq4_drift_specificity.py:100`, `plot_rq4_trajectory.py:20/26`, `plot_rq5_determinization.py:9/21` (gsm8k overlay), `plot_rq3_ft_dynamics.py:22`.
- **1 dynamics-CSV reader:** `plot_rq3_ft_dynamics.py:5/20/41` → `results/rq3_ft_dynamics/rq3_dynamics.csv`.

---

## 6. What does NOT move

- `results/rq1_emergence/{isotropy_pythia.csv, isotropy_aggregated_balanced.csv, cka_results_annotated.csv, cka_intercategory.npy}` — RQ1, frozen.
- `results/rq2_probing/{accuracy_metrics_corrected.csv, weights/*, test_indices/*, confound_checks_hardened.csv, parity_confound_checks.csv, accuracy_matrix_C10_backup.csv}`, `results/rq2_sweep/*` — RQ2, frozen.
- `results/nf4_degradation/*`, `results/gsm8k/*` — shared baselines, names carry no RQ lineage token, frozen.
- Legacy non-namespaced `__main__` writes in `src/metrics/cka.py` (`results/cka_intercategory.npy`, `cka_drift_temporal.npy`) and `isotropy.py` (`results/isotropy.csv`) — no RQ token; report separately (§10), do not touch.

---

## 7. (reserved)

---

## 8. Commit plan (one rename group per commit, collision-safe, revertible, `scaffolding` only)

> Each commit is internally complete: all references for that group are updated within the commit, so `pytest tests/` is green **after every commit**. Capture the pre-refactor hash manifest (test plan T6) **before Commit 1**.

**Commit 1 — `rq4 → rq5` (determinization → thesis RQ5).**
`run_rq4.py`→`run_rq5.py`; `plot_rq4_determinization.py`→`plot_rq5_determinization.py`; `RQ4DeterminizationRow`→`RQ5DeterminizationRow`; config `rq4_batch_size/rq4_output_dir`→`rq5_*` (+ value `rq4_determinization`→`rq5_determinization`); `git mv results/rq4_determinization → results/rq5_determinization`; `figures/rq4 → figures/rq5` (+ html); oracolo `rq4→rq5`,`viz-rq4→viz-rq5`,`solo_rq4→solo_rq5` + path decls; tests `test_rq4_*→test_rq5_*`, viz/oracolo key updates; `RQ4` docstrings→`RQ5`. Do not touch `run_rq3`/`rq3` tokens.

**Commit 2 — `rq3 → rq4` (drift/probe/GSM8K/NF4 → thesis RQ4).**
`run_rq3.py`→`run_rq4.py`; `rq3_drift_specificity.py`→`rq4_drift_specificity.py`; `plot_rq3_trajectory.py`→`plot_rq4_trajectory.py`; `RQ3TrajectoryRow`→`RQ4TrajectoryRow`; config `rq3_trajectory_csv`→`rq4_trajectory_csv` (+ value→`results/rq4_drift/...`); `rq3_max_relative_drift` per §3; **results migration** of `trajectories_probing.csv` + `drift_specificity_summary.json` into `results/rq4_drift/` + write-path decoupling in `run_rq4`; **6 readers repointed**; `checkpoint_loop.py` subprocess `run_rq3.py`→`run_rq4.py`; `figures/rq3 → figures/rq4` (+ html); oracolo `rq3→rq4`,`viz-rq3→viz-rq4` + path decls + drift guards; tests `test_rq3_drift_specificity.py`→`test_rq4_...`, `test_rq3_pipeline`→`test_rq4_pipeline`, viz/nf4/oracolo updates; `RQ3` docstrings→`RQ4`. Do not touch `run_rq1_dynamics`/`supplementary`/`rq1-dyn`/`viz-supp`. Bare `rq1`/`rq2` untouched.

**Commit 3 — `supplementary (rq1_dynamics) → rq3` (FT-geometry dynamics → thesis RQ3).**
`run_rq1_dynamics.py`→`run_rq3.py`; `plot_ft_geometry_dynamics.py`→`plot_rq3_ft_dynamics.py` (+ OUT_HTML→`figures/rq3/rq3_ft_dynamics.html`); **results migration** `rq1_dynamics.csv`→`results/rq3_ft_dynamics/rq3_dynamics.csv` + write-path decoupling (keep RQ1/nf4 reads); reader repoint; `git mv supplementary_ft_dynamics.html → figures/rq3/...`; oracolo `rq1-dyn→rq3`,`viz-supp→viz-rq3` + IO decls + descriptions; tests oracolo/viz updates; "supplementary" docstrings per §9.8. **Freeze the `get_seed` purpose string** at the former `run_rq1_dynamics.py:127` (§3).

**Commit 4 — docs + `CLAUDE.md` reference alignment** (§4.10). No logic.

Revert is `git revert <commit>` per group. Never `dev`/`main`; never push.

---

## 9. Decisions (approved by the student)

1. **New result dir names — APPROVED:** `results/rq3_ft_dynamics/`, `results/rq4_drift/`, `results/rq5_determinization/`.
2. **`rq1_dynamics.csv` → `rq3_dynamics.csv` — APPROVED** (rename; "rq1" in the old name meant "the RQ1 geometry tracked over time").
3. **Viz renames — APPROVED:** `plot_ft_geometry_dynamics.py` → `plot_rq3_ft_dynamics.py`; `supplementary_ft_dynamics.html` → `rq3_ft_dynamics.html`.
4. **`config_rq2.yaml` filename — APPROVED to LEAVE AS-IS.** Master config; `rq2` is an aligned number; renaming would hit every `--config` call, oracolo, tests, docs, CLAUDE.md. Not renamed in this refactor.
5. **`checkpoint_loop` subprocess target — VERIFIED against the recon.** `checkpoint_loop.py:91-96` currently shells `run_rq3.py`, which is the frozen-probe + Frobenius-drift driver (code `rq3` = thesis RQ4), called with `--config` + `--checkpoint_dir`. Post-rename the same file is `run_rq4.py` with an unchanged interface, so the literal becomes **`run_rq4.py`**. checkpoint_loop does not drive the supplementary/RQ3 path (run independently off `checkpoints_extracted/`).
6. **Grep-gate whitelist — APPROVED.** Frozen seed-purpose strings + serialized keys are carved out of the "zero old tokens" assertion, each documented file:line with an inline `# frozen` comment.
7. **"Supplementary/exploratory" docstrings — APPROVED to reframe fully** to "RQ3 (FT geometry dynamics)".
8. **`rq3_max_relative_drift` (auto-handled, no decision needed):** if serialized in `nf4_degradation/summary.json` → **freeze** (preserve bytes); else rename to `rq4_max_relative_drift`. Claude Code reports which case held.

---

## 10. Report separately (do NOT fix in this refactor)

- `CLAUDE.md` is substantively stale (documents only 3 RQs; omits RQ4/RQ5, `determinization.py`, `oracolo.py`, `rq*_drift_specificity`, `schemas.py`). Needs a content pass, separate from the rename.
- `CLAUDE.md` "Active branch: `dev`" note — branch state, the student's call.
- Legacy non-namespaced `__main__` writes in `cka.py` / `isotropy.py` — pre-existing smell, no RQ token.
- `config_rq2.yaml` misleading name, if left (§9.4).
- Any logic bug encountered — report, do not fix.

---

## 11. POST-REFACTOR TEST PLAN (Claude Code runs at the end; explicit pass criteria)

**T0 — Pre-refactor snapshot (before Commit 1).** `git status` clean on `scaffolding`; record `git rev-parse HEAD`. Compute and store `sha256` + `mtime` for every data file to be migrated (the CSV/JSON/NPY in §5). This manifest is the no-recompute oracle for T6.

**T1 — pytest.** `pytest tests/ -q` exits 0, 0 failures, 0 errors, no collection/import errors. The CPU suite (6 files: `test_cka_robustness`, `test_isotropy_floor`, `test_nf4_snr`, `test_rq4_drift_specificity`, `test_viz_smoke`, `test_oracolo_smoke`) all green; `test_pipeline_e2e` green on whatever tier it normally runs. **Pass:** exit 0. Green after each commit, not only at the end.

**T2 — Import smoke, every entry point.**
`python -c "import run_rq1, run_rq2, run_rq3, run_rq4, run_rq5"` (new: `run_rq3`=FT-dynamics, `run_rq4`=drift, `run_rq5`=determinization).
Import each `src...` module CLI incl. `src.eval.rq4_drift_specificity`, `src.viz.plot_rq4_trajectory`, `src.viz.plot_rq5_determinization`, `src.viz.plot_rq3_ft_dynamics`, `src.extraction.checkpoint_loop`.
`python scripts/oracolo.py --list` (or dry-run) resolves **exactly 28** keys, no `KeyError`/path error. **Pass:** all imports exit 0; 28 keys present with the new labels.

**T3 — Grep gate (zero stale lineage tokens).** Over tracked files, excluding `.git/ results/ data/ __pycache__/ logs/ docs/RECON.md` and the documented frozen whitelist (§3). Assert **0 hits** for each OLD compound token in its old sense:
```
run_rq1_dynamics | rq1[-_]dyn(amics)? | viz-supp | supplementary_ft_dynamics
plot_rq3_trajectory | rq3_drift_specificity | rq3_trajectory_csv | RQ3TrajectoryRow | rq3_dashboard | figures/rq3(as drift)
run_rq3\.py(as drift driver) | rq3_max_relative_drift (unless frozen)
run_rq4\.py(as determ) | rq4_determinization | RQ4DeterminizationRow | rq4_output_dir | rq4_batch_size | plot_rq4_determinization | solo_rq4 | viz-rq4(as determ) | figures/rq4(as determ)
```
**Do NOT** grep bare `rq1`/`rq2` (they legitimately remain). **Pass:** every old compound token → 0 hits outside the whitelist; whitelist hits match the reported file:line list exactly.

**T4 — Config keys resolve.** Load `config_rq2.yaml`: `rq4_trajectory_csv`, `rq5_batch_size`, `rq5_output_dir` present; `rq3_trajectory_csv`, `rq4_batch_size`, `rq4_output_dir` absent; values point to `results/rq4_drift/...` and `results/rq5_determinization`. `python -m src.utils.validate_configs` exits 0. **Pass:** key presence/absence + values as specified; validator green.

**T5 — Subprocess target.** `grep -n "run_rq" src/extraction/checkpoint_loop.py` → the shelled script is `run_rq4.py`; **0** hits for `run_rq3.py` there. `regenerate_dataset.py` still targets `run_rq2.py` + confound checkers. **Pass:** checkpoint_loop shells `run_rq4.py`.

**T6 — Migrated files unchanged (no recompute).** Re-compute `sha256` for every migrated data file at its new path; compare to the T0 manifest. **Pass:** `sha256(new) == sha256(old)` for every migrated CSV/JSON/NPY; `git mv` preserves `mtime`. Figures (regenerable) also expected byte-equal. **Any** data-file hash mismatch ⇒ a recompute happened ⇒ **FAIL**: revert, fix the spec, do not edit the file. (This is why `rq3_max_relative_drift`, if serialized, is frozen rather than renamed, §3.)

**T7 — RQ1/RQ2 untouched proof.** `git diff --stat <T0 HEAD>..HEAD` shows no changes to `run_rq1.py`, `run_rq2.py`, RQ1/RQ2 metric cores, `categories.py`, `models.py`, `seeds.py`, `engine.py`, `schemas.py`, and the RQ1/RQ2 result files (`results/rq1_emergence/*` minus moved `dynamic/`; `results/rq2_probing/*` minus moved `dynamic/`). Any unavoidable touch (a genuinely shared reader) is explicitly enumerated in the report. **Pass:** RQ1/RQ2 code + frozen result files byte-identical.

**Overall pass:** T1–T7 all pass. **If any test fails, the SPEC is wrong: fix the spec (and the rename), never the data.** Never edit a validated file to satisfy a hash test.

---

## 12. COPY-PASTE EXECUTION PROMPT (hand to Claude Code after the student signs off §9)

> You are on branch `scaffolding` of the Pythia-1.4B arithmetic interpretability repo. Execute the semantic RQ-renumbering described in `REFACTOR_SPEC.md` (read it fully first). Rules: rename only, no logic/numbers/seeds/results change; RQ1 and RQ2 stay frozen; if you find a logic bug, report it, do not fix it; all work on `scaffolding`, never `dev`/`main`, never push.
>
> 1. Confirm `git status` clean on `scaffolding`; record HEAD. Run test plan **T0**: store `sha256`+`mtime` for every data file in §5. Grep all `get_seed(` purpose strings and list any containing lineage tokens; **freeze** them (§3). Check whether `rq3_max_relative_drift` is serialized in `results/nf4_degradation/summary.json`; apply §3.
> 2. Execute the commits in **strict order C1 `rq4→rq5`, C2 `rq3→rq4`, C3 `supplementary→rq3`, C4 docs** (§8), one commit per group, using the §4 table, the §5 migration map (`git mv` only), and the §4.8 oracolo+test lockstep. After each commit, run `pytest tests/ -q` and confirm green.
> 3. Apply the §9 decisions exactly as the student approved them (dir names, `rq3_dynamics.csv`, viz renames, `config_rq2.yaml` left as-is, `checkpoint_loop`→`run_rq4.py`, whitelist, docstring framing).
> 4. Run the full test plan **T1–T7** (§11). Report: per-commit one-line summary; the grep-gate output; the T6 hash-manifest comparison (old vs new path, equal/unequal); the frozen-token whitelist actually applied (file:line); the resolution of `rq3_max_relative_drift`; and any §10 items to report separately.
> 5. **Stop and ask** (do not improvise) if: a data file hash would change, a serialized key forces a data-contract conflict, the 28 oracolo keys cannot be preserved, or you hit a logic bug. Leave the branch in a clean, committed, revertible state.