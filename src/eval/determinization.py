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
import torch

from src.config.categories import MATH_CATS
from src.extraction.extract_states import _last_token_indices, _resolve_pad_id


# ── SECTION 1 — ROW CONTRACT (ARCH-03) ────────────────────────────────────────

class RQ4DeterminizationRow(TypedDict):
    """Per-(step, category) row in determinization.csv. Aggregated over examples.

    entropy_mean / margin_mean / p_first_token_mean are over ALL rows in the category;
    the *_single fields (entropy/margin/p_correct + Wald CI) are over the single-token-
    result subset only, where the correct answer is exactly one token so P(first token)
    == P(full answer). The subset isolates CAT-SIGN's positive (digit-answer) half from
    its negative half, whose first token is the sign " -", not a digit.
    """
    step: int
    category: str
    n_rows: int
    n_single_token: int
    entropy_mean: float
    margin_mean: float
    entropy_mean_single: float
    margin_mean_single: float
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


# ── SECTION 3 — TARGET-TOKEN BUILDER (tokenizer-dependent) ────────────────────

def math_stimuli(stimuli: list[dict]) -> list[dict]:
    """Math-category rows in source order. The single ordering RQ4 aligns to:
    build_targets and extract_eq_logits must both filter through this helper."""
    return [s for s in stimuli if s["category"] in MATH_CATS]


def build_targets(tokenizer, stimuli: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per math stimulus, the answer's first token id and a single-token mask.

    The answer appears after "=" as a space-led number, so the target is the first
    token of the continuation `text + " " + result`. encode(text) must be a true
    prefix of encode(text + " " + result); otherwise the gather position and the
    target token would refer to different tokenizations (fail fast, C-02).

    single_token_mask marks rows whose result is exactly one token (positives/zeros);
    multi-token rows are the negatives, whose first token is " -" (the answer's sign).
    Returns (target_ids [n], single_token_mask [n]) aligned to math_stimuli order.
    """
    rows = math_stimuli(stimuli)
    target_ids = np.empty(len(rows), dtype=np.int64)
    single = np.empty(len(rows), dtype=bool)

    for i, s in enumerate(rows):
        text = s["text"]
        result = s["labels"]["result"]
        prefix = tokenizer.encode(text)
        full = tokenizer.encode(f"{text} {result}")
        if full[: len(prefix)] != prefix:
            raise ValueError(
                f"encode(text) is not a prefix of encode(text + ' ' + result) for id "
                f"{s.get('id', '?')}: prefix={prefix} full={full}. The '=' gather "
                "position would not align with the answer token."
            )
        continuation = full[len(prefix):]
        target_ids[i] = continuation[0]
        single[i] = (len(continuation) == 1)

    return target_ids, single


# ── SECTION 4 — LOGIT EXTRACTOR (model-dependent) ─────────────────────────────

def extract_eq_logits(model, stimuli: list[dict], batch_size: int) -> np.ndarray:
    """Next-token logits at the "=" position for every math stimulus. Returns [n, vocab] float32.

    Mirrors extract_states.extract_from_model's gather: to_tokens right-pads, so the
    real terminal token ("=") is found per row via _last_token_indices, and the mask
    is derived from non-pad positions with column 0 forced on (BOS shares pad's id 0).
    logits[:, t, :] predicts token t+1, so the row at last_idx is P(next token after "=").
    """
    rows = math_stimuli(stimuli)
    if not rows:
        return np.empty((0, 0), dtype=np.float32)

    pad_id = _resolve_pad_id(model)
    gathered_batches = []

    for i in range(0, len(rows), batch_size):
        texts = [s["text"] for s in rows[i : i + batch_size]]
        tokens = model.to_tokens(texts, prepend_bos=True)

        attention_mask = (tokens != pad_id).long()
        attention_mask[:, 0] = 1

        last_idx = _last_token_indices(tokens, pad_id)
        row_idx = torch.arange(tokens.shape[0])

        with torch.no_grad():
            logits = model(tokens, attention_mask=attention_mask, return_type="logits")

        eq_logits = logits[row_idx, last_idx, :]
        gathered_batches.append(eq_logits.detach().cpu().to(torch.float32).numpy())

    return np.concatenate(gathered_batches, axis=0)
