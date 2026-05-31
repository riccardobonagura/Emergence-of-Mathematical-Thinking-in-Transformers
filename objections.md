We're adding a SUPPLEMENTARY, POST-HOC exploratory analysis that bridges RQ1 and RQ3
— it is NOT part of either RQ's confirmatory design. First revise the plan at
/home/ricbng/.claude/plans/iterative-prancing-reef.md with the edits below, show me
the diff, and on my approval implement it. Branch dev only. Reuse-only: do not edit
run_rq1.py, run_rq3.py, src/metrics/cka.py, or src/metrics/isotropy.py.

PLAN EDITS (apply to the plan doc, then carry into implementation):

1. Reframe + rename. Title the deliverable "Supplementary — Fine-tuning geometry
   dynamics (exploratory bridge between RQ1 and RQ3)". Drop the "RQ1 + RQ3" framing
   that implies confirmatory scope. Rename viz -> src/viz/plot_ft_geometry_dynamics.py
   and output -> results/figures/supplementary_ft_dynamics.html. CSV stays at
   results/rq1_emergence/dynamic/rq1_dynamics.csv.

2. Checkpoint enumeration. Do NOT hardcode steps 2500/5000/7500/10000. Enumerate
   data/processed/checkpoints_extracted/ at runtime, parse step from dir name, map
   final_adapter via the same convention run_rq3.py already uses. Add a printed note:
   "n = <count> extracted states (design specifies ~25 at save-every-500); resolution
   is bound by what is on disk. Extracting the intermediate saved checkpoints would
   raise n with no methodological change." Assert the stimuli hash of every checkpoint
   metadata equals the base (ad7886c9...) before computing anything.

3. Seeds. Inter-category CKA at step 0 MUST reuse get_seed(seed, "rq1_subsampling", 0)
   so step-0 rows reproduce results/rq1_emergence/ exactly. Cross-temporal CKA uses a
   NEW distinct purpose: get_seed(seed, "rq1_dynamics_crosstemporal", 0), shared across
   all steps. Never raw seeds.

4. Baseline noise floor (E-G-02). On BOTH CKA panels, overlay the expected-variance band
   from the existing RQ1 reviewer baselines: CKA(CTRL-NEU, CTRL-NUM) and within-math
   across-template CKA (already in results/rq1_emergence/, e.g. cka_results_annotated.csv).
   A CKA change inside that band is noise, not signal. Panel intros must NOT describe a
   high inter-category CKA as "divergence" — it means similar representations.

5. Caveat text in each panel intro, citing the registry verbatim-in-spirit:
   - Cross-temporal CKA panel: E-G-02 (CKA = similarity, scale deltas vs expected
     variance; linear CKA invariant to rotation + isotropic scaling, which is the FEATURE
     that lets it isolate restructuring from rotation — the complement to RQ3 Frobenius);
     E-F-03 (cite the T16 NF4 degradation when reporting any drift).
   - Inter-category CKA panel: E-G-02 + the positional-asymmetry caveat (math read at "=",
     ctrl at terminal word/".").
   - Isotropy panel: E-G-01 (ΔIso is RELATIVE, anisotropy != semantic richness, may reflect
     corpus token density per Ethayarajh) AND extend the positional-asymmetry caveat here
     too (ΔIso is also a math-vs-ctrl contrast).
   - All GSM8K-overlay companions: E-G-04 + E-M-02 + E-O-04 — descriptive only; n is too
     small for any correlation coefficient or significance; name the MetaMath<->GSM8K
     distributional-overlap third-variable confound; no causal language; report magnitude
     (effect size), not coefficients. E-F-02: trajectory is within 1 epoch (~12k steps) and
     may be pre-convergence, so the endpoint is not a saturated geometry.

6. Add one line acknowledging evolutionary (layer-to-layer) CKA — the third RQ1 CKA notion
   — is intentionally out of scope for this supplement.

IMPLEMENTATION (after approval):
- New: run_rq1_dynamics.py (root, mirrors run_rq3.py structure) and
  src/viz/plot_ft_geometry_dynamics.py.
- Reuse: isotropy_exact; linear_cka / compute_cka_intercategory / compute_cka_cross_temporal
  (+ subsample_indices) from cka.py; MetadataHandler, load_hidden_states, _atomic_write_csv
  from io_utils.py; get_seed from seeds.py; MATH_CATS/CTRL_CATS from the categories SSOT.
- CSV columns: step, layer, iso_math, iso_ctrl, delta_iso, ci_low_*, ci_high_*, cka_inter,
  cka_vs_base. Atomic write, UTF-8, English inline comments only.
- Pull gsm8k per step from results/rq2_probing/dynamic/trajectories_probing.csv; assert the
  GSM8K steps cover the dynamics steps (incl. final_adapter) — warn on any gap.

VERIFICATION (run and show output):
1. python run_rq1_dynamics.py --config configs/config_rq2.yaml
2. Assert step-0 consistency: delta_iso@L15 ~= -0.1056 vs isotropy_aggregated_balanced.csv;
   cka_inter matches cka_results_annotated.csv; cka_vs_base == 1.0 for all layers at step 0.
3. Print mean cross-temporal drift per step (expect monotone-ish increase, echoing RQ3
   Frobenius 0.0 -> ~0.690) and confirm it exceeds the baseline noise band.
4. python -m src.viz.plot_ft_geometry_dynamics; confirm 3 panels render, baseline band shown,
   GSM8K overlay present, every caveat block displays.
5. Update CLAUDE.md pipeline stages and docs/pipeline_dataflow.md with the new node.
Do not push; I push manually. Do not commit.