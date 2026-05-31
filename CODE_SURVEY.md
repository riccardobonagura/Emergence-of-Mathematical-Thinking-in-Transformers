# Code Survey — design/style issues (survey only, no changes made)

Scanned `src/` against the project's own stated rules (CLAUDE.md "Mandatory
invariants", comment policy; Comandamenti design hierarchy). Grouped by severity.
Each item gives `file:line` and the rule it touches. **No fixes applied** — this
is a report. (The pipeline-task code edits committed/staged separately are out of
scope here.)

---

## A. Violations of stated "never violate" invariants

### A1. Change markers in comments — invariant: *"no change markers (# Modifica, # Fix)"*
~30 occurrences of `FIX N-0x`, `CL-0x`, `FIX IO-0x`, `N-0x:` baked into comments.
These are review/changelog artifacts that the invariant explicitly bans from code.
- `src/eval/nf4_degradation.py` — `FIX N-01/02/04/05`, `N-03/06/07:` (lines 67, 94, 111, 125, 161, 168, 175, 191, 214)
- `src/extraction/checkpoint_loop.py` — `CL-01..CL-07` (lines 55, 74, 78, 91, 100, 113, 130, 146, 161)
- `src/probing/io_utils.py` — `FIX IO-01`, `FIX IO-04` (lines ~133, 142)
- `src/utils/validate_configs.py:62` — `HARDENED: Added ...`
- `src/extraction/extract_states.py:6` — `HARDENED: Restores ...`

**Why it matters:** changelog-in-comments rots; git history is the changelog.
Recommend stripping all `FIX/CL/N-/HARDENED` tags, keeping only the *what/why* in
plain terms.

### A2. Italian comments & print strings — invariant: *"Comments: short, English"*
Concentrated almost entirely in **`src/metrics/cka.py`** (~80 hits): e.g.
`Questa matrice cattura la struttura relazionale` (141), `Caso degenere` (161),
`BLOCCO 4 — Uso 1: CKA intra-modello` (252), `Caricamento del primo layer` (287),
`La matrice è simmetrica` (313), `Salvataggio .npy (formato binario)` (549),
plus Italian `print()` output (301, 470, 552, 666). `cka.py` reads as if authored
in a different style/era than the rest of the (English) codebase.

### A3. Non-atomic result writes — invariant: *"always `_atomic_write_csv`/`_atomic_save_npy` for all result writes"*
The atomic helpers exist in `io_utils.py` but these result-producing paths bypass them:
- `src/metrics/isotropy.py:393` — `df.to_csv(out_file, index=False)` (RQ1 isotropy CSV)
- `src/metrics/cka.py:551, 665, 701` — `np.save(...)` direct (RQ1 CKA `.npy` outputs)
- `src/dataset/merge_stimuli.py:318` — `json.dump(...)` direct (dataset metadata)

**Why it matters:** a crash mid-write leaves a truncated/corrupt RQ1 artifact —
exactly what the atomicity invariant is meant to prevent.

### A4. `open()` without explicit encoding — invariant: *"always `encoding='utf-8'`"*
- `src/metrics/cka.py:641` — `open(BASE_DIR / "metadata.json", "r")` (no `encoding=`).
(All other `open()` calls checked pass; the binary `os.fdopen(fd,"wb")` in io_utils is correctly exempt.)

---

## B. Comment / prose style — *"short, English, inline"*

### B1. Grandiose / over-claiming prose (~50 hits)
The codebase narrates itself in marketing/over-formal register that obscures simple
operations and over-claims rigor:
- CLI descriptions: `"Strict NF4 Quantization Degradation Verifier"` (nf4_degradation.py:95),
  `"Strict GSM8K Evaluation"` (eval_gsm8k.py:117), `"Strict production-grade YAML configuration validator"` (validate_configs.py:188).
- Section banners: `RIGOROUS METRICS ASSESSMENT LOOPS`, `DUAL METRIC REPORTING PARADIGM`,
  `HARDENED ALLIGNED NF4 EXTRACTION` (also a typo: "ALLIGNED") in nf4_degradation.py.
- Inflated verbs/nouns: `Eradicates contiguous slice bias`, `seamless comparison`,
  `hardware orchestration`, `Commencing parallel text compilation mapping`,
  `metrics payloads`, `OS replacement swaps`, `frozen spatial isolation evaluations`.

**Why it matters:** for a thesis codebase, plain comments ("compute relative
Frobenius per layer") are more credible than "RIGOROUS METRICS ASSESSMENT LOOPS",
and the over-claiming ("Strict", "production-grade") invites scrutiny the code
can't always back (cf. the NF4 NaN bug shipped under "RIGOROUS").

---

## C. Design / robustness smells (judgment calls, not rule violations)

### C1. Hardcoded base-model id — `src/eval/eval_gsm8k.py:144`
`model_args = f"pretrained=EleutherAI/pythia-1.4b,peft={args.model_path}"` hardcodes
the HF path, while `get_model_profile()["hf_path"]` and `config["model_name"]` exist
for exactly this. Inconsistent with the otherwise config/profile-driven design;
silently wrong if the model ever changes.

### C2. Hardcoded `d_model` fallback — `src/probing/io_utils.py:46`
`get_d_model(self, default: int = 2048)` — a magic 2048 default. If metadata is
missing the code proceeds with a possibly-wrong dimension instead of failing loud.

### C3. `src/metrics/cka.py` is doing too much — 709 lines (largest module)
Bundles intra-model CKA, inter-category CKA, checkpoint-drift CKA, and save/load
(its own `BLOCCO 4/5/7` sections). Mixed `print()` progress vs. the `logging` used
elsewhere. Candidate to split and to standardize on the logger.

### C4. Stale identifiers after this session's NF4 fix — `src/eval/nf4_degradation.py`
The reference pass is now **bfloat16**, but variables/temp dirs are still named
`tmp_fp16` / `H_fp16` and one comment still says "standard relative Frobenius
... FP16". Cosmetic, but misleading — rename to `*_ref`. *(Flagged for honesty: I
introduced this when patching the NaN bug.)*

---

## D. Minor / non-issues (recorded to prevent false alarms)

- **`io_utils.py` atomic writers** use bare `except Exception:` — but they
  `os.remove(tmp); raise`, i.e. cleanup-then-reraise. **Correct**, not error-swallowing.
- `src/viz/*` use `plt.savefig`/`fig.savefig` directly (not atomic) — acceptable
  for figures (regenerable, not result data), though could be unified.
- `config.get("model_name", "pythia-1.4b")` defaults appear in several entrypoints
  (extract_states, checkpoint_loop, confound checks) — defensible defaults, but
  the literal repeats; a single config-required key would be cleaner.

---

## Suggested priority if you later act on this
1. **A2 + A3 in `cka.py`** (Italian + non-atomic) — single highest-value cleanup;
   it's the outlier file and produces RQ1 artifacts.
2. **A1 change markers** — mechanical strip across nf4/checkpoint_loop/io_utils.
3. **A4 / C1 / C2** — one-line correctness/consistency fixes.
4. **B1 prose** — optional, but improves the thesis-defense optics.
5. **C3 / C4** — refactors, lowest urgency.
