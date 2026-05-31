# Peer-Review Brief — Geometric Dynamics in Transformer Internal Representations

**Thesis (BSc CS, Federico II Napoli — Bonagura N46007216).** Model: Pythia-1.4B
(EleutherAI, 24 layers, d_model=2048). Synthetic controlled arithmetic dataset,
operands in [10,50], 3 syntactic templates. All numbers below are from the
**regenerated pipeline (2026-05-31)** on the v5 tensors. *English; an Italian
version is available on request.*

> **Status note.** GSM8K accuracy and the NF4-quantization baseline (T16) are
> **not yet included** — both require GPU re-runs that are pending a hardware
> reset. Everything else (RQ1, RQ2, confound analysis, RQ3 geometric drift) is
> final and verified.

---

## 1. Methods (one paragraph)

Hidden states are extracted at the **`=` token** (FP16, one vector per stimulus
per layer). Two binary properties are probed with **L2-regularized logistic
regression** (C=1.0), pair-aware train/test split, 5-fold CV, StandardScaler,
bootstrap CIs, permutation test on the training CV, and **Benjamini–Hochberg**
correction across 24 layers × 2 properties = 48 tests. RQ1 measures **isotropy**
(per-category) and **evolutionary CKA** (layer-to-layer similarity, linear CKA).
RQ3 freezes the RQ2 probes and re-applies them to QLoRA-NF4 checkpoints
(MetaMathQA, QKV-only adapters) while tracking **relative Frobenius drift** of
the attention projections.

**Scope / epistemic frame.** All claims are **correlative**: a property being
linearly decodable at layer *l* means it is *accessible*, not that the model
*uses* it (Belinkov 2022; Hewitt & Liang 2019). No causal claims are made.

---

## 2. RQ1 — Isotropy & CKA

**Isotropy (ΔIso = iso_math − iso_ctrl; lower iso = more isotropic).**
Math representations are **more isotropic than control** across layers 7–22,
with the trough **ΔIso = −0.106 at layer 15** (CIs non-overlapping, n=1000/side).
The sign flips **positive at layer 23 (+0.036)**. Layers 0–5 show only small
negative ΔIso (−0.01 to −0.04).

> *Reviewer caveat (E-G-01):* anisotropy ≠ semantic richness; ΔIso should be read
> as a **relative** difference between categories, not an absolute structural
> measure. A confound (token-frequency distribution in the training corpus,
> Ethayarajh 2019) cannot be excluded.

**Evolutionary CKA (self-similarity layer l vs l−1).** Math and control reorganize
**similarly** through the stack. At the terminal layer, CKA_math = 0.926 vs
CKA_ctrl = 0.887 (Δ = 0.039). **This terminal divergence is *not* statistically
distinguishable from baseline layer-to-layer variation**: background ΔCKA over
layers 1–22 is −0.012 ± 0.027, giving the layer-23 point a **Z ≈ −1.03**.

> *Honest reading:* RQ1-CKA is effectively a **null result** on "math diverges at
> the end" — the geometry of math and control evolves comparably, and the small
> terminal gap is within noise. We recommend reporting it as such rather than as
> evidence of late-stage mathematical specialization.

---

## 3. RQ2 — Linear probing (sign, parity)

| Property | Emergence (acc>0.7) | Peak | Peak acc | Significant cells (BH) |
|---|---|---|---|---|
| **Sign** (result <0 vs ≥0) | **layer 0** (acc 0.93) | layer 3 | **1.00** | 24/24 |
| **Parity** (result even/odd) | layer 6 | layer 15 | **0.99** | 18/24 |

- **Sign** is decodable essentially from the input: **0.93 at layer 0** (embedding
  layer, before any transformer block), saturating to 1.00 by layer 3.
- **Parity** shows a **two-stage** profile: a weak plateau (~0.67–0.72,
  layers 5–12) then a sharp jump to ≥0.94 from **layer 13**. This is consistent
  with parity requiring computation (units-digit interaction) absent from the
  surface form — compatible with the staged-emergence pattern of Tenney et al.
  (2019).

> *P0 reviewer question (sign @ layer 0):* perfect/near-perfect decodability at
> the embedding layer indicates the information is **in the input tokens**, not
> produced by internal computation. The thesis must frame layer-0 sign as a
> property of the *stimulus encoding*, not of model reasoning. **See §4 — this is
> compounded by a measured confound.**

---

## 4. Confound analysis — **the key finding** 🚩

A hardened, **direct** confound test (not just weight-cosine) was run per layer.

**Sign probe is contaminated by operand-1 magnitude (Confound N-01) on all 24
layers.** Two facts coexist:

- The probe **weight vector** is nearly orthogonal to the magnitude direction:
  `cosine(w_sign, w_mag)` max = **0.064** — consistent with the earlier benign
  reading (≈0.078) that previously led us to *dismiss* N-01.
- But the probe's **predictions** track operand-1 strongly:
  `corr(sign_logits, operand1)` = **0.55–0.67** across layers, with operand-1
  itself near-perfectly decodable (R² = 0.97–1.00). **All 24 layers flagged
  significant.**

> **Implication:** the cosine-of-weights test was **insufficient**; the direct
> logit-correlation test shows the sign probe is **partially a magnitude
> shortcut**. The RQ2 "sign is decodable everywhere" claim must be **down-graded
> and explicitly qualified**. This directly answers reviewer P0 #2 (a *direct*
> test existed and was needed).

**Parity** is **far less** confounded: `corr(parity_logits, operand2_parity)`
max = **0.199**, ground-truth `corr(result_parity, operand2_parity)` = 0.12,
operand-1 parity balance = 0.514. The BH flag is significant on all 24 layers,
but this is a **small effect** (statistical significance ≠ effect size, E-M-03) —
parity decodability is much more defensible as a genuine internal property.

---

## 5. RQ3 — Probe trajectory under QLoRA fine-tuning

Frozen RQ2 probes applied to 6 trajectory points (base + 5 checkpoints,
step 0 → 12343):

| step | 0 | 2500 | 5000 | 7500 | 10000 | 12343 |
|---|---|---|---|---|---|---|
| **sign acc** | 0.994 | 0.737 | 0.744 | 0.732 | 0.745 | 0.739 |
| **parity acc** | 0.797 | 0.724 | 0.712 | 0.703 | 0.703 | 0.703 |
| **Frobenius drift (rel., math)** | 0.000 | 0.448 | 0.536 | 0.552 | 0.562 | 0.572 |

- Fine-tuning on MetaMathQA causes a **large, immediate drop** in linear
  decodability (sign 0.99→0.74 by the first checkpoint, then stable), while
  geometric **drift rises monotonically** and saturates (~0.57).
- The probe-accuracy drop is **front-loaded** (most of it by step 2500) whereas
  drift keeps creeping — the two curves are not proportional.

> *Reviewer caveats:* (E-G-04) drift measures **weight change, not capability
> change**; (scope) adapters are **QKV-only**, so RQ3 claims are restricted to
> attention projections, not MLPs. **The NF4 noise floor (T16) is required to
> contextualize the 0.57 drift** and is pending (see §6).

---

## 6. Outstanding (pending GPU reset)

1. **GSM8K 0-shot** across the 6 trajectory points (baseline expected ≈ 0.0 for a
   1.4B base model — Cobbe et al. 2021; declare 0-shot vs 5-shot gap, E-F-01).
2. **NF4/T16 degradation baseline** — quantifies how much of the RQ3 drift is
   pure 4-bit quantization noise vs. fine-tuning (E-F-03). *(The first attempt
   had a numerical bug, now fixed: the unquantized reference is computed in
   bfloat16 to avoid FP16 overflow on deep-layer activations.)*

---

## 7. Questions for the reviewer

1. Given §4, is the **sign** result salvageable as a qualified claim
   ("decodable, but entangled with operand magnitude"), or should the thesis lead
   with **parity** as the cleaner demonstration?
2. Is the **layer-0 sign decodability** best framed purely as a stimulus-encoding
   artifact, with the interesting RQ2 signal being **parity's layer-13 jump**?
3. For RQ1, do you agree the **CKA terminal divergence should be reported as
   within-noise** (null), rather than as late-stage specialization?
4. 500 pairs/category vs the BLiMP 1000 standard — acceptable for a BSc thesis
   with the CIs shown, or a limitation to foreground?
