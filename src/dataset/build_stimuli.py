"""
build_stimuli.py  —  v5
=======================
Property-contrastive arithmetic stimuli for Pythia-1.4B geometric analysis.

Research Questions
------------------
RQ1  Emergence   — Isotropy + CKA across Pythia-1.4B layers (24 layers).
RQ2  Decoding    — Linear probes for sign and parity at the last token.
RQ3  Dynamics    — Same stimuli re-evaluated at every MetaMath/QLoRA checkpoint.

Dataset categories
------------------
CAT-SIGN   (1 000 stimuli = 500 pairs)
    Contrast:   (a − b = [positive])  vs  (b − a = [negative])
    Operator:   subtraction (fixed)
    Controlled: operator, |operands|, |result|, parity(|result|), template
    Varies:     sign of result only
    ⚠ N-01: first operand differs between pair members — unavoidable for any
             sign contrast on a causal/left-to-right model; documented here,
             not correctable without introducing a worse confound.

CAT-PARITY (1 000 stimuli = 500 pairs)
    Contrast:   (a + b = [even])  vs  (a + (b+1) = [odd])
    Operator:   addition (fixed)
    Controlled: operator, operand a, magnitude (|result| differs by 1), template
    Varies:     parity of result only
    ⚠ N-02: second operand b vs b+1 — minimum possible surface difference for
             intra-operator parity contrast; documented here, not correctable.

Operand domain
--------------
Both generators use integers in [10, 50].  Every two-digit integer in this
range is a **single token** on the GPT-NeoX (Pythia) tokenizer, giving a
fixed four-token structure for the minimal template:

    "12 - 7 ="  →  ["12", " -", " 7", " ="]          (minimal)
    "Compute: 12 - 7 ="  →  ["Compute", ":", " 12", " -", " 7", " ="]
    "Calculate 12 - 7 ="  →  ["Calculate", " 12", " -", " 7", " ="]

The "=" token is **always the last token** of the input, so
``equals_sign_index == last_token_index`` for every arithmetic stimulus.

Templates
---------
Three English templates per category, assigned round-robin across pairs,
guaranteeing exact balance (500/3 ≈ 167 per template, remainder on TPL-*-1).

Extraction strategy
-------------------
Both sign and parity are extracted at "last_token" (= the "=" token).
This follows the ROME/MEMIT finding that factual information integrates at
the last token of the subject expression, not at the operator token.

Layer strategy
--------------
probe_layer_strategy = "all_layers" — representations at every layer of
Pythia-1.4B (layers 0–23) are extracted and analysed independently.

USAGE
-----
    python build_stimuli.py --n_pairs 500 \\
                            --tokenizer EleutherAI/pythia-1.4b \\
                            --output data/raw/stimuli_arithmetic_v5.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Version & domain constants
# ─────────────────────────────────────────────────────────────────────────────

DATASET_VERSION = "v5"

# Operand ranges — 2-digit integers, single-token on GPT-NeoX.
SIGN_A_MIN, SIGN_A_MAX = 10, 50   # 'a' in (a − b); must be > b
SIGN_B_MIN, SIGN_B_MAX = 10, 50   # 'b' in (a − b); must be < a

PAR_A_MIN,  PAR_A_MAX  = 10, 50   # 'a' in (a + b)
# Only even b values enter the pool.  Each entry (a, b_even) generates:
#   member A  →  a + b_even        parity = a%2   (even+even or odd+even)
#   member B  →  a + (b_even+1)    parity = 1−a%2
# Restricting to even b values means no two pool entries share a "boundary"
# text, eliminating all cross-pair duplicate stimuli.
PAR_B_EVEN: List[int] = list(range(10, 49, 2))   # [10,12,...,48]  20 values

# Templates — English, always ending with "=" so last_token == equals_sign_index.
# Round-robin assignment: pair k gets template index (k % 3).
SIGN_TEMPLATES: Dict[str, str] = {
    "TPL-SIGN-1": "{a} - {b} =",
    "TPL-SIGN-2": "Compute: {a} - {b} =",
    "TPL-SIGN-3": "Calculate {a} - {b} =",
}

PARITY_TEMPLATES: Dict[str, str] = {
    "TPL-PAR-1": "{a} + {b} =",
    "TPL-PAR-2": "Compute: {a} + {b} =",
    "TPL-PAR-3": "Calculate {a} + {b} =",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Labels:
    """Semantic labels for a single arithmetic stimulus."""
    result:  int    # exact integer result
    sign:    int    # 0 = non-negative, 1 = negative
    parity:  int    # 0 = even, 1 = odd (based on abs(result))


@dataclass(frozen=True)
class Contrast:
    """Contrastive pair metadata."""
    pair_id:         str
    varying_axis:    str         # "sign" | "parity"
    controlled_axes: Tuple[str, ...]


@dataclass(frozen=True)
class TokenFields:
    """
    Tokenizer-dependent fields.  All indices are None until
    populate_token_fields() is called.  Uses None (not -1) as the
    unpopulated sentinel throughout (fixes I-12).
    """
    tokenizer_name:       Optional[str]        = None
    n_tokens:             Optional[int]        = None
    token_ids:            Optional[Tuple[int, ...]]   = None
    token_strs:           Optional[Tuple[str, ...]]   = None
    token_length_strata:  Optional[str]        = None  # "short"|"medium"|"long"
    equals_sign_index:    Optional[int]        = None  # index of "=" token
    last_token_index:     Optional[int]        = None  # always n_tokens - 1


@dataclass(frozen=True)
class Stimulus:
    """One arithmetic or control stimulus."""
    id:          str
    text:        str
    split:       str   # always "geometric_eval"
    category:    str   # "CAT-SIGN" | "CAT-PARITY"
    template_id: str
    macro_format: str  # "symbolic_arithmetic"
    extraction_strategy_by_property: Dict[str, str]
    n_reasoning_steps: int
    labels:      Labels
    contrast:    Contrast
    token_fields: TokenFields
    ood_target:  str
    dataset_version: str
    operand_digit_class: str   # e.g. "2d_2d"
    probe_layer_strategy: str  # "all_layers"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _digit_class(a: int, b: int) -> str:
    """Return a string like '2d_2d' describing the digit counts of |a| and |b|."""
    return f"{len(str(abs(a)))}d_{len(str(abs(b)))}d"


def _sign_label(x: int) -> int:
    return 1 if x < 0 else 0


def _parity_label(x: int) -> int:
    return abs(x) % 2


def _token_length_strata(n: int) -> str:
    if n <= 5:  return "short"
    if n <= 8:  return "medium"
    return "long"


def _make_unpopulated_token_fields() -> TokenFields:
    return TokenFields()


_EXTRACTION_STRATEGY: Dict[str, str] = {
    "sign":   "last_token",
    "parity": "last_token",
}


# ─────────────────────────────────────────────────────────────────────────────
# SignContrastGenerator
# ─────────────────────────────────────────────────────────────────────────────

class SignContrastGenerator:
    """
    Generates CAT-SIGN contrastive pairs.

    Each pair consists of:
        Member A:  "{a} - {b} ="   →   result = +(a−b),  sign = 0
        Member B:  "{b} - {a} ="   →   result = −(a−b),  sign = 1

    with a > b, a ∈ [SIGN_A_MIN, SIGN_A_MAX], b ∈ [SIGN_B_MIN, SIGN_A_MAX-1].

    Pool size: Σ_{a=11}^{50} (a−10) = 820 distinct (a,b) pairs.
    """

    CATEGORY = "CAT-SIGN"

    def _build_pool(self) -> List[Tuple[int, int]]:
        return [
            (a, b)
            for a in range(SIGN_A_MIN, SIGN_A_MAX + 1)
            for b in range(SIGN_B_MIN, a)          # guarantees a > b
        ]

    def build(self, n_pairs: int, seed: int = 42) -> List[Stimulus]:
        """
        Sample n_pairs without replacement and return 2*n_pairs Stimulus objects.

        Raises ValueError if n_pairs exceeds the pool size (820).
        """
        pool = self._build_pool()
        if n_pairs > len(pool):
            raise ValueError(
                f"SignContrastGenerator: requested {n_pairs} pairs but pool "
                f"has only {len(pool)}. Lower n_pairs or widen the operand range."
            )
        rng = random.Random(seed)
        sampled = rng.sample(pool, n_pairs)

        template_keys = list(SIGN_TEMPLATES.keys())
        stimuli: List[Stimulus] = []

        for idx, (a, b) in enumerate(sampled):
            tpl_key = template_keys[idx % len(template_keys)]
            tpl_str = SIGN_TEMPLATES[tpl_key]
            pair_id = f"SIGN-{idx:04d}"
            stimuli.extend(self._make_pair(pair_id, a, b, tpl_key, tpl_str))

        return stimuli

    def _make_pair(
        self,
        pair_id: str,
        a: int, b: int,
        tpl_key: str,
        tpl_str: str,
    ) -> Tuple[Stimulus, Stimulus]:
        result_a = a - b    # positive
        result_b = b - a    # negative

        common = dict(
            split              = "geometric_eval",
            category           = self.CATEGORY,
            template_id        = tpl_key,
            macro_format       = "symbolic_arithmetic",
            extraction_strategy_by_property = dict(_EXTRACTION_STRATEGY),
            n_reasoning_steps  = 1,
            contrast           = Contrast(
                pair_id         = pair_id,
                varying_axis    = "sign",
                controlled_axes = ("operator", "operands_abs", "result_abs",
                                   "result_parity", "template"),
            ),
            token_fields       = _make_unpopulated_token_fields(),
            ood_target         = "in_distribution",
            dataset_version    = DATASET_VERSION,
            operand_digit_class = _digit_class(a, b),
            probe_layer_strategy = "all_layers",
        )

        stim_a = Stimulus(  
            id      = f"{pair_id}-A",
            text    = tpl_str.format(a=a, b=b),
            labels  = Labels(result=result_a, sign=_sign_label(result_a),
                             parity=_parity_label(result_a)),
            **common, # type: ignore[arg-type]
        ) 
        stim_b = Stimulus(
            id      = f"{pair_id}-B",
            text    = tpl_str.format(a=b, b=a),   # operands swapped
            labels  = Labels(result=result_b, sign=_sign_label(result_b),
                             parity=_parity_label(result_b)),
            **common, # type: ignore[arg-type]
        ) 
        return stim_a, stim_b


# ─────────────────────────────────────────────────────────────────────────────
# ParityContrastGenerator
# ─────────────────────────────────────────────────────────────────────────────

class ParityContrastGenerator:
    """
    Generates CAT-PARITY contrastive pairs.

    Pool design
    -----------
    Pool entries are (a, b_even) where b_even ∈ PAR_B_EVEN = [10,12,...,48].
    Each entry produces:
        Member A:  "{a} + {b_A} ="   →  result parity = 0  (even)
        Member B:  "{a} + {b_B} ="   →  result parity = 1  (odd)

    The mapping from (a, b_even) to (b_A, b_B) depends on the parity of a:
        a even:  b_A = b_even,    b_B = b_even + 1   (even+even=even, even+odd=odd)
        a odd:   b_A = b_even+1,  b_B = b_even       (odd+odd=even,  odd+even=odd)

    Member A is always the even result; Member B is always the odd result.

    Why no duplicate texts
    ----------------------
    • All member-A texts use b_A ∈ {10,11,12,...,49} but b_A is even iff a is even
      and odd iff a is odd.  Because a-parity and b-value together uniquely index
      each text, and pool entries use non-adjacent b_even values (step=2), no two
      distinct pool entries yield the same text.

    Exact 50/50 balance
    -------------------
    Achieved via stratified sampling: n_pairs // 2 pairs from even-a pool and
    n_pairs - n_pairs // 2 from odd-a pool.  Since every even-a pair contributes
    (parity=0, parity=1) and every odd-a pair also contributes (parity=0, parity=1),
    total counts are exactly equal.

    Pool sizes
    ----------
        even-a pool : 21 values × 20 b_even = 420 pairs  (need ≤ 500//2 = 250)
        odd-a pool  : 20 values × 20 b_even = 400 pairs  (need ≤ 500//2 = 250)

    Known confound N-02
    -------------------
    b_A and b_B differ by exactly 1 (minimum possible for intra-operator parity
    contrast).  This is unavoidable; documented in methodology.
    """

    CATEGORY = "CAT-PARITY"

    def _build_pools(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        even_a = [(a, b_e)
                  for a in range(PAR_A_MIN, PAR_A_MAX + 1) if a % 2 == 0
                  for b_e in PAR_B_EVEN]
        odd_a  = [(a, b_e)
                  for a in range(PAR_A_MIN, PAR_A_MAX + 1) if a % 2 == 1
                  for b_e in PAR_B_EVEN]
        return even_a, odd_a

    def build(self, n_pairs: int, seed: int = 42) -> List[Stimulus]:
        """
        Sample n_pairs with stratified 50/50 parity balance.
        Raises ValueError if n_pairs exceeds pool capacity.
        """
        even_pool, odd_pool = self._build_pools()
        n_even = n_pairs // 2
        n_odd  = n_pairs - n_even

        if n_even > len(even_pool):
            raise ValueError(
                f"ParityContrastGenerator: even-a pool has {len(even_pool)} entries "
                f"but {n_even} requested. Lower n_pairs."
            )
        if n_odd > len(odd_pool):
            raise ValueError(
                f"ParityContrastGenerator: odd-a pool has {len(odd_pool)} entries "
                f"but {n_odd} requested. Lower n_pairs."
            )

        rng = random.Random(seed)
        sampled = rng.sample(even_pool, n_even) + rng.sample(odd_pool, n_odd)
        rng.shuffle(sampled)

        template_keys = list(PARITY_TEMPLATES.keys())
        stimuli: List[Stimulus] = []

        for idx, (a, b_even) in enumerate(sampled):
            tpl_key = template_keys[idx % len(template_keys)]
            tpl_str = PARITY_TEMPLATES[tpl_key]
            pair_id = f"PAR-{idx:04d}"
            stimuli.extend(self._make_pair(pair_id, a, b_even, tpl_key, tpl_str))

        return stimuli

    def _make_pair(
        self,
        pair_id: str,
        a: int,
        b_even: int,
        tpl_key: str,
        tpl_str: str,
    ) -> Tuple[Stimulus, Stimulus]:
        b_odd = b_even + 1
        # member A is always even result, member B always odd
        if a % 2 == 0:
            b_A, b_B = b_even, b_odd    # even+even=even, even+odd=odd
        else:
            b_A, b_B = b_odd, b_even    # odd+odd=even,  odd+even=odd

        r_A, r_B = a + b_A, a + b_B
        assert r_A % 2 == 0 and r_B % 2 == 1, (
            f"Invariant violated: r_A={r_A}, r_B={r_B} for a={a}, b_even={b_even}"
        )

        common = dict(
            split              = "geometric_eval",
            category           = self.CATEGORY,
            template_id        = tpl_key,
            macro_format       = "symbolic_arithmetic",
            extraction_strategy_by_property = dict(_EXTRACTION_STRATEGY),
            n_reasoning_steps  = 1,
            contrast           = Contrast(
                pair_id         = pair_id,
                varying_axis    = "parity",
                controlled_axes = ("operator", "operand_a", "magnitude_pm1",
                                   "template"),
            ),
            token_fields       = _make_unpopulated_token_fields(),
            ood_target         = "in_distribution",
            dataset_version    = DATASET_VERSION,
            operand_digit_class = _digit_class(a, b_A),
            probe_layer_strategy = "all_layers",
        )

        stim_a = Stimulus( 
            id      = f"{pair_id}-A",
            text    = tpl_str.format(a=a, b=b_A),
            labels  = Labels(result=r_A, sign=_sign_label(r_A), parity=0),
            **common, # type: ignore[arg-type]
        ) 
        stim_b = Stimulus( 
            id      = f"{pair_id}-B",
            text    = tpl_str.format(a=a, b=b_B),
            labels  = Labels(result=r_B, sign=_sign_label(r_B), parity=1),
            **common, # type: ignore[arg-type]
        ) 
        return stim_a, stim_b


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def populate_token_fields(
    stimuli: List[Stimulus],
    tokenizer_name: str,
) -> List[Stimulus]:
    """
    Populate TokenFields for every stimulus using the given HuggingFace tokenizer.

    For CAT-SIGN and CAT-PARITY stimuli the "=" token is always last, so
    equals_sign_index == last_token_index by construction.  The function
    verifies this invariant and raises AssertionError if violated.

    For CTRL stimuli (no "=" in text) equals_sign_index is set to None.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required for tokenisation.\n"
            "Install with: pip install transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    populated: List[Stimulus] = []
    for s in stimuli:
        ids   = tokenizer.encode(s.text, add_special_tokens=False)
        strs  = tokenizer.convert_ids_to_tokens(ids)
        n     = len(ids)
        last  = n - 1

        # Locate "=" — strip surrounding whitespace/Ġ prefix for comparison.
        eq_idx: Optional[int] = None
        for i, tok in enumerate(strs):
            clean = tok.replace("Ġ", "").replace("▁", "").strip()
            if clean == "=":
                eq_idx = i

        # For arithmetic stimuli, "=" must be the last token.
        if s.category.startswith("CAT-"):
            assert eq_idx == last, (
                f"Stimulus {s.id}: expected '=' at index {last} "
                f"(last token), found at {eq_idx}.\n"
                f"Text: {s.text!r}\nTokens: {strs}"
            )

        tf = TokenFields(
            tokenizer_name      = tokenizer_name,
            n_tokens            = n,
            token_ids           = tuple(ids),
            token_strs          = tuple(strs),
            token_length_strata = _token_length_strata(n),
            equals_sign_index   = eq_idx,
            last_token_index    = last,
        )
        populated.append(replace(s, token_fields=tf))

    return populated


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(stimuli: List[Stimulus]) -> None:
    """
    Assert structural invariants on a fully generated (optionally tokenised)
    dataset.  Raises AssertionError with a descriptive message on failure.
    """
    sign_stimuli  = [s for s in stimuli if s.category == "CAT-SIGN"]
    par_stimuli   = [s for s in stimuli if s.category == "CAT-PARITY"]
    arith_stimuli = sign_stimuli + par_stimuli

    # 1. Sign balance: exactly 50 % positive, 50 % negative in CAT-SIGN.
    sign_dist = Counter(s.labels.sign for s in sign_stimuli)
    assert sign_dist[0] == sign_dist[1], (
        f"Sign imbalance in CAT-SIGN: {dict(sign_dist)}. "
        "Expected equal counts of sign=0 and sign=1."
    )

    # 2. Parity balance: exactly 50 % even, 50 % odd in CAT-PARITY.
    par_dist = Counter(s.labels.parity for s in par_stimuli)
    assert par_dist[0] == par_dist[1], (
        f"Parity imbalance in CAT-PARITY: {dict(par_dist)}. "
        "Expected equal counts of parity=0 and parity=1."
    )

    # 3. Pair coherence: every pair_id appears exactly twice within its category.
    for cat_stimuli, cat_name in [(sign_stimuli, "CAT-SIGN"),
                                   (par_stimuli, "CAT-PARITY")]:
        pair_counts = Counter(s.contrast.pair_id for s in cat_stimuli)
        bad = {k: v for k, v in pair_counts.items() if v != 2}
        assert not bad, (
            f"{cat_name}: pair_id(s) with count ≠ 2: {bad}"
        )

    # 4. Correct varying_axis field.
    for s in sign_stimuli:
        assert s.contrast.varying_axis == "sign", (
            f"{s.id}: expected varying_axis='sign', got {s.contrast.varying_axis!r}"
        )
    for s in par_stimuli:
        assert s.contrast.varying_axis == "parity", (
            f"{s.id}: expected varying_axis='parity', got {s.contrast.varying_axis!r}"
        )

    # 5. Digit class uniformity — all arithmetic stimuli should be "2d_2d"
    #    given the [10, 50] operand range.
    bad_dc = [s.id for s in arith_stimuli if s.operand_digit_class != "2d_2d"]
    assert not bad_dc, (
        f"Non-2d_2d digit classes found: {bad_dc[:5]}"
    )

    # 6. No duplicate texts within each arithmetic category.
    for cat_stimuli, cat_name in [(sign_stimuli, "CAT-SIGN"),
                                   (par_stimuli, "CAT-PARITY")]:
        texts = [s.text for s in cat_stimuli]
        n_dup = len(texts) - len(set(texts))
        assert n_dup == 0, (
            f"{cat_name}: {n_dup} duplicate text(s) found."
        )

    # 7. Template balance within each category (each template used ≈ n/3 times).
    for cat_stimuli, cat_name in [(sign_stimuli, "CAT-SIGN"),
                                   (par_stimuli, "CAT-PARITY")]:
        tpl_counts = Counter(s.template_id for s in cat_stimuli)
        counts = sorted(tpl_counts.values())
        # Allow at most 1 stimulus difference between most- and least-used template.
        assert counts[-1] - counts[0] <= 2, (
            f"{cat_name}: template imbalance {dict(tpl_counts)}. "
            "Max allowed difference between template counts: 2."
        )

    # 8. Token-field checks (only if stimuli have been tokenised).
    tokenised = [s for s in arith_stimuli if s.token_fields.n_tokens is not None]
    if tokenised:
        for s in tokenised:
            assert s.token_fields.equals_sign_index is not None, (
                f"{s.id}: tokenised but equals_sign_index is None."
            )
            assert s.token_fields.equals_sign_index == s.token_fields.last_token_index, (
                f"{s.id}: equals_sign_index ({s.token_fields.equals_sign_index}) "
                f"!= last_token_index ({s.token_fields.last_token_index})."
            )
            assert s.token_fields.tokenizer_name is not None, (
                f"{s.id}: tokenizer_name not set in TokenFields."
            )

    # 9. CAT-SIGN invariant: |result_A| == |result_B| within each pair.
    pair_map: Dict[str, List[Stimulus]] = {}
    for s in sign_stimuli:
        pair_map.setdefault(s.contrast.pair_id, []).append(s)
    for pid, pair in pair_map.items():
        assert len(pair) == 2
        assert abs(pair[0].labels.result) == abs(pair[1].labels.result), (
            f"CAT-SIGN pair {pid}: |result| mismatch "
            f"{pair[0].labels.result} vs {pair[1].labels.result}."
        )

    # 10. CAT-PARITY invariant: result_B == result_A ± 1 within each pair.
    pair_map_p: Dict[str, List[Stimulus]] = {}
    for s in par_stimuli:
        pair_map_p.setdefault(s.contrast.pair_id, []).append(s)
    for pid, pair in pair_map_p.items():
        assert len(pair) == 2
        diff = abs(pair[0].labels.result - pair[1].labels.result)
        assert diff == 1, (
            f"CAT-PARITY pair {pid}: result difference = {diff}, expected 1."
        )
        # Confirm member-A is always even (parity=0), member-B always odd (parity=1).
        stim_a = next(s for s in pair if s.id.endswith("-A"))
        stim_b = next(s for s in pair if s.id.endswith("-B"))
        assert stim_a.labels.parity == 0, (
            f"CAT-PARITY {pid}-A: expected parity=0, got {stim_a.labels.parity}."
        )
        assert stim_b.labels.parity == 1, (
            f"CAT-PARITY {pid}-B: expected parity=1, got {stim_b.labels.parity}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_jsonl(path: Path | str, stimuli: List[Stimulus]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(s.to_json() for s in stimuli) + "\n",
        encoding="utf-8",
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sign- and parity-contrastive arithmetic stimuli (v5)."
    )
    parser.add_argument("--n_pairs",   type=int, default=500,
                        help="Pairs per category (default 500 → 1 000 stimuli each).")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HuggingFace tokenizer name (e.g. EleutherAI/pythia-1.4b). "
                             "Skip tokenisation if not provided.")
    parser.add_argument("--output",    type=str,
                        default="data/raw/stimuli_arithmetic_v5.jsonl")
    args = parser.parse_args()

    print(f"Generating {args.n_pairs} pairs per category "
          f"(seed={args.seed}) …")

    sign_gen  = SignContrastGenerator()
    par_gen   = ParityContrastGenerator()

    sign_stimuli = sign_gen.build(args.n_pairs, seed=args.seed)
    par_stimuli  = par_gen.build(args.n_pairs,  seed=args.seed + 1)
    all_stimuli  = sign_stimuli + par_stimuli

    print(f"  CAT-SIGN   : {len(sign_stimuli):>5} stimuli")
    print(f"  CAT-PARITY : {len(par_stimuli):>5} stimuli")

    if args.tokenizer:
        print(f"Tokenising with {args.tokenizer} …")
        all_stimuli = populate_token_fields(all_stimuli, args.tokenizer)

    print("Validating …")
    validate_dataset(all_stimuli)

    out = write_jsonl(args.output, all_stimuli)
    print(f"✓  {len(all_stimuli)} stimuli written to {out}")


if __name__ == "__main__":
    main()