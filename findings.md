# RQ4 RECON — findings (read-only, dev branch)

Goal of eventual patch (context): per-checkpoint, inference-only determinization at the "="
token — next-token **entropy**, **P(correct answer token)**, **top1−top2 logit margin** — math
rows only (CAT-SIGN, CAT-PARITY). Recon only; nothing edited/committed.

Environment verified: `transformers 4.48.3` (inside `<4.49` guard ✓), TransformerLens present,
`EleutherAI/pythia-1.4b` cached under `~/.cache/huggingface/hub` (runs offline with `HF_HUB_OFFLINE=1`).

---

## 1. Tokenizer single-token subset (DERIVED empirically, not assumed)

Loaded the real tokenizer. For every math row I encoded the prefix (`text`, which ends in the
`" ="` token id 426) and the continuation `text + " " + str(result)`, then isolated the tokens
beyond the prefix. The space-led separator is the correct one: the model's natural continuation
after `" ="` (`Ġ=`) is a space-prefixed number (`Ġ29`), matching how operands themselves tokenize
(`" 13"` = id 2145). Encoding with `""` gives `"29"` (id 1717) — wrong target; **use `" "`**.

Result (2000 math rows, `data/processed/dataset_master_v5.jsonl`, `labels.result`):

| group | single-token | multi-token |
|---|---|---|
| CAT-SIGN | 500 | **500** |
| CAT-PARITY | 1000 | 0 |
| by sign: positive (1485) | 1485 | 0 |
| by sign: zero (15) | 15 | 0 |
| by sign: **negative (500)** | 0 | **500** |

- **Negatives** (`"13 - 42 ="` → `-29`) tokenize as **two** tokens: first `" -"` (id **428**, the same
  `Ġ-` used as the subtraction operator), then `"29"` (id 1717). So for negatives the first answer
  token is **always** `" -"`. All 500 multi-token rows are negative and all live in CAT-SIGN;
  CAT-PARITY results are all ≥0 → all single-token.
- Positives/zero → exactly one token `" N"` (e.g. `" 29"` id 3285, `" 0"`).
- file:line — gather column is the `" ="` position via `extract_states._last_token_indices`
  (`src/extraction/extract_states.py:54`), pre-flight-asserted to decode to `"="` by
  `validate_extraction_tokens` (`extract_states.py:65`).

**Patch decision.**
- Define the **correct answer token id** per row as `tokenizer.encode(text + " " + str(result))[len(encode(text)):][0]`
  (the *first* continuation token), precomputed once into an array aligned to `metadata.stimuli_ids`.
- **single-token-result subset criterion**: `len(continuation) == 1`. Report `P(correct)`, entropy,
  margin on the full math set, but flag/group by this subset — on it `P(first-token)` == `P(full answer)`.
- **P(first-answer-token) fallback** (the required definition for multi-token rows): always score
  `P(continuation[0])`. For positives/zero this is the whole answer; for negatives it is `P(" -")`,
  i.e. the probability the model commits to *starting a negative number* — a clean, well-defined
  proxy that needs no autoregressive rollout. Entropy and margin are intrinsic to the `"="` logits
  and need no answer token at all.

---

## 2. Logit return + off-by-one + center_unembed

- `HookedTransformer.forward` defaults to `return_type="logits"`
  (`…/transformer_lens/HookedTransformer.py:504`) and accepts `attention_mask`
  (`HookedTransformer.py:511`); shape is `[batch, pos, d_vocab]`. The exact call mirroring the
  extraction masking is `model(tokens, attention_mask=attention_mask, return_type="logits")` reusing
  the `extract_states.py:108-116` idiom (`to_tokens(prepend_bos=True)`; mask `= (tokens != pad_id)`;
  force `mask[:,0]=1`). No `run_with_hooks` needed — RQ4 wants the output, not an internal cache.
- **Off-by-one (verified on real decode):** prefix ids end `[…, 426]` where 426 = `" ="` at position
  `last_idx`. `logits[:, t, :]` predicts token `t+1`; the empirical continuation's first token after
  the `"="` position is the answer first token. So `logits[row, last_idx, :]` = P(next token after "=")
  = the RQ4 target. ✓
- **center_unembed:** `from_pretrained` default is `True` (`HookedTransformer.py:1123`); the
  `checkpoint_loop` merge call does not override it, so RQ4 inherits `True`. center_unembed sets
  `W_U` column-mean to zero → subtracts a **per-position constant** from every logit. Entropy,
  softmax `P(token)`, and the top1−top2 margin are **all invariant** to adding a constant to a logit
  row. So all three RQ4 metrics are unaffected; no need to touch the flag.

**Patch decision.** Reuse the merge→HookedTransformer idiom; gather logits at `last_idx`, recompute
`last_idx`/`attention_mask` exactly as extraction does. Do not pass `center_unembed` — default True is safe.

---

## 3. Checkpoint reality

On disk under `data/processed/checkpoints/`: `checkpoint-2500`, `checkpoint-5000`,
`checkpoint-7500`, `checkpoint-10000`, `final_adapter` (plus a `_skip` dir that the
`"checkpoint" in d.name` filter at `checkpoint_loop.py:140` already excludes). Same set already
re-extracted under `data/processed/checkpoints_extracted/`.

- Step parsing: `checkpoint-N` → `int(name.split("-")[-1])` (`checkpoint_loop.py:141`); terminal
  `final_adapter`/`final_checkpoint` → `config.get("total_training_steps", 2000)` in
  `run_rq3.py:71-73` (config_rq2 sets `total_training_steps: 12343`).
- **Step-0 / base semantics:** the existing RQ3 trajectory already carries `step ∈ {0, 2500, 5000,
  7500, 10000, 12343}`. Step 0 = the **base, pre-FT** model: `run_rq3` produces it by being pointed at
  the base extracted dir `data/processed/pythia-1.4b/` (`"pythia-1.4b".split("-")[-1]` → not int →
  `step_num=0`, drift 0). GSM8K confirms the same anchor: `gsm8k_baseline.json` acc **0.0** →
  `final_adapter` **0.171** (trajectory `gsm8k_acc`: 0.0→0.099→0.136→0.152→0.161→0.171). RQ4 must
  include step 0 to stay consistent with the "from 0 baseline" story — for step 0 it loads the
  **base** `HookedTransformer` (no adapter merge), for the 5 adapters it uses the merged model.
- The `checkpoint_loop.process_checkpoint` path (`checkpoint_loop.py:60-71`) yields a merged
  `HookedTransformer` (fp16, fold_ln=True) that RQ4 can reuse as-is; RQ4 only swaps the post-merge
  body (logit gather instead of `extract_from_model`'s hidden-state hooks).

**Patch decision.** Enumerate via the `checkpoint_loop.main()` idiom (sorted `checkpoint-*` + terminal
adapter); add a synthetic **step-0 base pass** (no PEFT merge) so RQ4's trajectory starts at 0 like RQ3/GSM8K.

---

## 4. Test surface

- `tests/test_pipeline_e2e.py` mocks `load_hidden_states` with synthetic CPU tensors
  (`@patch("run_rq3.load_hidden_states")` at line 333; fixture `mock_pipeline_env` at line 27 builds
  the JSONL/metadata/per-layer `.pt` fixtures). **No real model or tokenizer is ever loaded** — RQ4's
  model+logit path cannot be exercised the RQ3 way.
- **Pure-function unit tests** (the bulk of coverage): entropy / margin / P(token) computed on a
  small synthetic `[n, vocab]` logit matrix with hand-checkable values (uniform→max entropy & zero
  margin; one-hot→zero entropy & P=1). Live next to `test_probing_algebra`
  (`tests/test_pipeline_e2e.py:362`) as `test_rq4_*`.
- **e2e**: mirror `test_rq3_pipeline` (line 333) but `@patch` the **RQ4 logit-extractor** (the function
  that returns the `[n_rows, vocab]` "=" logits) with a synthetic matrix, then assert the per-step CSV
  exists with the three metric columns. Same file, SECTION after the RQ3 runner.

**Patch decision.** Factor RQ4 so the logit gather is one mockable function (input: model+stimuli →
output: `[n_math, vocab]` float array at "="); metrics are pure functions of that array + a target-id
array. Both layers unit-testable on CPU with no GPU/tokenizer.

---

## Skeptical sweep

- **fp16 vs float32:** model runs fp16, but compute entropy/margin/P in **float32** — cast the gathered
  `[n, ~50k]` logit row to float32 before `logsumexp`/softmax. fp16 logsumexp over a 50k vocab is
  numerically fragile (overflow in exp, lost precision in the sum); the cast is cheap (one row per stimulus).
- **Bare `config[key]` / `default_rng(seed)` temptations:**
  - New keys MUST be `config.get("rq4_...", default)` — `configs/config_test.yaml` has **no** rq4 keys
    and **no** `total_training_steps` (verified), so any bare `config["rq4_..."]` or `config["total_training_steps"]`
    breaks the e2e config. Reuse `run_rq3.py:73`'s `config.get("total_training_steps", 2000)` for the terminal step.
  - RQ4 is deterministic over the **full** math set → ideally **no RNG at all**. If a subsample/CI is
    added, it must go through `get_seed(config["seed"], "<purpose>")` (`src/probing/seeds.py:9`), never
    `np.random.default_rng(42)` directly (E-O-04).
  - Do **not** reuse `bootstrap_ci` (`src/probing/stats.py:16`): it is accuracy-specific
    (`(y_true==y_pred)` resampling) and read by run_rq2 — wrong tool for continuous means; if a CI on a
    mean is wanted, write a separate continuous-mean bootstrap.
- **No geometry touched:** RQ4 imports none of `linear_cka`/`center_gram` (`src/metrics/cka.py`),
  `isotropy_exact`/`cka_inter_mean`/`run_isotropy_analysis` (`src/metrics/isotropy.py`). It reads only
  logits → confirmed the geometry metrics are untouched.
- **New TypedDict, not per-layer:** RQ4 is per-step (aggregate over examples), so define a fresh
  `RQ4DeterminizationRow` (e.g. `step, n_rows, n_single_token, entropy_mean, p_correct_mean,
  margin_mean, …`) — do **not** extend `RQ3TrajectoryRow` (`run_rq3.py:25`, which is per-layer×property).
  Persist with `_atomic_write_csv(path, df.to_dict("records"), df.columns.tolist())` + the
  append-and-replace-by-step idiom (`run_rq3.py:169-173`).

---

## Open questions for the patch author

1. **Aggregation granularity:** one row per step over all math rows, or split CAT-SIGN vs CAT-PARITY
   (and single- vs multi-token subset)? The 500 negative/multi-token rows behave differently
   (`P(" -")` proxy) — recommend reporting per-category **and** a single-token-only column so the
   "P(correct)" headline is unambiguous.
2. **Where do the per-row target token ids live?** Precompute once and persist (e.g. a small
   `rq4_target_tokens.npy` aligned to `stimuli_ids`), or recompute from the tokenizer each run?
   Persisting avoids loading the tokenizer in the metric path and keeps unit tests tokenizer-free.
3. **Step-0 base pass:** load base `HookedTransformer` separately (no PEFT) inside the RQ4 runner, or
   read it from a pre-extracted artifact? RQ3 gets step 0 by pointing at `data/processed/pythia-1.4b/`,
   but that dir holds *hidden states*, not logits — RQ4 must run the base model live for step 0.
4. **Orchestration home:** a standalone `run_rq4.py` invoked per `--checkpoint_dir` (subprocess, like
   `run_rq3`), or folded into the GPU `checkpoint_loop` so the merged model is reused without a second
   merge? Reuse is cheaper but couples RQ4 to the GPU loop; standalone is testable and matches RQ3's shape.
5. **GSM8K correlation:** is drift↔determinization↔GSM8K correlation in scope for RQ4, or is RQ4 purely
   descriptive of behavioral sharpening? (n=6 steps — same small-n caveat as the supplementary dynamics.)
