"""determinization.py — RQ4: behavioral determinization at the "=" token.

E-P-02: the extraction point is the "=" token, so the next-token distribution there
is the model's *expected result* before it is generated, not the computed result.
RQ4 measures how that distribution *sharpens* over fine-tuning — lower next-token
entropy, larger top1-top2 logit margin, higher probability on the answer token —
as inference-only, correlative evidence of the FT trajectory.

This module is layered for testability:
  - pure metric functions on a [n, vocab] logit matrix (no model, no tokenizer);
  - build_targets(tokenizer, stimuli): the only tokenizer-dependent piece;
  - extract_eq_logits(model, stimuli, batch_size): the only model-dependent piece.

All metrics are computed in float32 (logsumexp/softmax stability over the ~50k vocab)
even though the model runs in fp16. Deterministic over the full math set — no RNG.
"""

from typing import TypedDict

import numpy as np


# ── SECTION 1 — ROW CONTRACT (ARCH-03) ────────────────────────────────────────

class RQ4DeterminizationRow(TypedDict):
    """Per-(step, category) row in determinization.csv. Aggregated over examples.

    entropy_mean / margin_mean / p_first_token_mean are over ALL rows in the category;
    p_correct_single (+ Wald CI) is over the single-token-result subset only, where the
    correct answer is exactly one token so P(first token) == P(full answer).
    """
    step: int
    category: str
    n_rows: int
    n_single_token: int
    entropy_mean: float
    margin_mean: float
    p_first_token_mean: float
    p_correct_single: float
    p_correct_single_ci_lo: float
    p_correct_single_ci_hi: float


# ── SECTION 2 — PURE METRIC FUNCTIONS ─────────────────────────────────────────

def next_token_entropy(logits: np.ndarray) -> np.ndarray:
    """Natural-log Shannon entropy of softmax(logits) per row. Returns [n] float32.

    Computed in log space (shift by row max) so the ~50k-wide softmax never overflows;
    the 0*log(0) limit is handled explicitly to stay NaN-safe under fp underflow.
    """
    x = logits.astype(np.float32, copy=False)
    z = x - x.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    log_p = z - logsumexp
    p = np.exp(log_p)
    # p*log_p → 0 where p underflows to 0 (log_p = -inf), avoiding 0 * -inf = nan.
    terms = np.where(p > 0.0, p * log_p, np.float32(0.0))
    return (-terms.sum(axis=-1)).astype(np.float32)


def top1_top2_margin(logits: np.ndarray) -> np.ndarray:
    """Gap between the largest and second-largest logit per row. Returns [n] float32."""
    x = logits.astype(np.float32, copy=False)
    if x.shape[-1] < 2:
        return np.zeros(x.shape[0], dtype=np.float32)
    # Partition puts the two largest in the last two slots (unordered between them).
    top2 = np.partition(x, -2, axis=-1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


def prob_of_target(logits: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """Softmax probability assigned to each row's target token. Returns [n] float32."""
    x = logits.astype(np.float32, copy=False)
    z = x - x.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    log_p = z - logsumexp
    rows = np.arange(x.shape[0])
    p_target = np.exp(log_p[rows, target_ids])
    return np.nan_to_num(p_target, nan=0.0).astype(np.float32)
