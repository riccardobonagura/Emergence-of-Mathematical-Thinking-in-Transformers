"""
build_control.py  —  v5
=======================
Generators for CTRL-NEU and CTRL-NUM control categories.

Both categories are in English (matching the arithmetic stimulus language)
and use the GPT-NeoX / Pythia tokenizer for length matching.

Length matching
---------------
Arithmetic stimuli (CAT-SIGN, CAT-PARITY) tokenise to 4–6 tokens on
GPT-NeoX.  Control stimuli are sampled so that their token-length
distribution mirrors the arithmetic distribution (fixes N-04).

The matching is performed by ``length_match_to_arithmetic()``, which
takes the tokenised arithmetic stimuli as reference and resamples the
control pool stratum-by-stratum.

CTRL-NEU  —  English prose, no numbers
    Short  (4–5 tokens) :  simple subject-verb(-object) phrases
    Medium (6–7 tokens) :  + adjective or adverb
    Long   (8–9 tokens) :  full clause with circumstantial phrase

CTRL-NUM  —  English numeric context (non-arithmetic)
    Short  (4–5 tokens) :  single-number contexts (platform, bus, floor)
    Medium (6–7 tokens) :  two-number contexts (time, measures)
    Long   (8–9 tokens) :  multi-number contexts (flights, contracts)

Sentinel convention (fixes I-12)
---------------------------------
equals_sign_index is always None for CTRL stimuli (no "=" in text).
last_token_index is set to n_tokens - 1 after tokenisation.
No "-1" sentinels are used anywhere in this file.

USAGE
-----
    python build_control.py --category neu --n_stimuli 500 \\
                            --tokenizer EleutherAI/pythia-1.4b \\
                            --output data/raw/stimuli_ctrl_neu_v5.jsonl

    python build_control.py --category num --n_stimuli 500 \\
                            --tokenizer EleutherAI/pythia-1.4b \\
                            --output data/raw/stimuli_ctrl_num_v5.jsonl
"""
from __future__ import annotations

import argparse
import random
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from build_stimuli import (
        DATASET_VERSION,
        Contrast,
        Labels,
        Stimulus,
        TokenFields,
        _token_length_strata,
        populate_token_fields,
        write_jsonl,
    )
except ImportError:
    print("[ERROR] build_stimuli.py not found in the same directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Shared defaults for control stimuli
# ─────────────────────────────────────────────────────────────────────────────

_CTRL_EXTRACTION_STRATEGY: Dict[str, str] = {
    "sign":   "last_token",
    "parity": "last_token",
}

_CTRL_LABELS = Labels(result=0, sign=-1, parity=-1)   # sentinel: no arithmetic


def _ctrl_contrast(category: str) -> Contrast:
    return Contrast(pair_id="N/A", varying_axis="N/A", controlled_axes=())


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class ControlGenerator(ABC):
    """Shared scaffolding: pool construction, sampling, schema assembly."""

    CATEGORY:   str = ""
    ID_PREFIX:  str = ""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    @abstractmethod
    def _build_pool_by_tier(self) -> Dict[str, List[str]]:
        """Return {"short": [...], "medium": [...], "long": [...]}."""

    def build(self, n: int) -> List[Stimulus]:
        """
        Build n control stimuli sampled uniformly across the three tiers.
        Raises ValueError when the pool of any tier is too small.
        """
        pool  = self._build_pool_by_tier()
        tiers = ("short", "medium", "long")
        base  = n // 3
        rem   = n % 3
        # distribute remainder to the first tiers
        counts = {t: base + (1 if i < rem else 0)
                  for i, t in enumerate(tiers)}

        selected: List[str] = []
        for tier, count in counts.items():
            if len(pool[tier]) < count:
                raise ValueError(
                    f"{self.__class__.__name__}: tier '{tier}' pool size "
                    f"{len(pool[tier])} < requested {count}.  "
                    "Reduce n or extend the word lists."
                )
            selected.extend(self.rng.sample(pool[tier], count))

        self.rng.shuffle(selected)
        return [
            self._make_stimulus(f"{self.ID_PREFIX}-{i:04d}", text)
            for i, text in enumerate(selected)
        ]

    def _make_stimulus(self, stim_id: str, text: str) -> Stimulus:
        return Stimulus(
            id           = stim_id,
            text         = text,
            split        = "geometric_eval",
            category     = self.CATEGORY,
            template_id  = f"{self.CATEGORY}-TPL",
            macro_format = "natural_language",
            extraction_strategy_by_property = dict(_CTRL_EXTRACTION_STRATEGY),
            n_reasoning_steps = 0,
            labels       = _CTRL_LABELS,
            contrast     = _ctrl_contrast(self.CATEGORY),
            token_fields = TokenFields(),   # all None — correct sentinel
            ood_target   = "control",
            dataset_version = DATASET_VERSION,
            operand_digit_class = "N/A",
            probe_layer_strategy = "all_layers",
        )


# ─────────────────────────────────────────────────────────────────────────────
# NeutralGenerator  (CTRL-NEU)
# ─────────────────────────────────────────────────────────────────────────────

class NeutralGenerator(ControlGenerator):
    """
    English natural-language prose without numbers or arithmetic structure.

    Word lists are chosen to produce 4–9 token sequences on GPT-NeoX
    so that the length distribution matches CAT-SIGN / CAT-PARITY.

    Tier construction (product of word lists):
      short  = S + V                →    8 × 8      =    64 texts
      medium = S + V + O            →    8 × 8 × 9  =   576 texts
      long   = S + V + O + ADV      →    8 × 8 × 9 × 8 = 4 608 texts
    """

    CATEGORY  = "CTRL-NEU"
    ID_PREFIX = "CTRL-NEU"

    # Short subjects (1 token each on GPT-NeoX)
    _S = [
    "Rain", "Wind", "Fog", "Snow", "Frost", "Dust", "Smoke", "Steam",  
    "Hail", "Mist", "Ice", "Ash", "Dew", "Cloud",                     
]

    # Verbs: space-prefixed so they join cleanly (1–2 tokens each)
    _V = [
    "falls", "blows", "clears", "drifts",
    "rises", "settles", "spreads", "fades",                           
    "lifts", "swirls", "gathers", "melts", "flows",                   
]

    # Objects: short 1–2 word noun phrases
    _O = [
        "the hills", "the fields", "the rooftops",
        "the streets", "the coastline", "the valleys",
        "the forest", "the harbour", "the skyline",
    ]

    # Adverbial phrases (2–3 tokens each)
    _ADV = [
        "at dawn", "by midday", "each winter",
        "in silence", "quite slowly", "every morning",
        "before noon", "near dusk",
    ]

    def _build_pool_by_tier(self) -> Dict[str, List[str]]:
        short  = [f"{s} {v}."           for s, v
                  in product(self._S, self._V)]
        medium = [f"{s} {v} over {o}."  for s, v, o
                  in product(self._S, self._V, self._O)]
        long   = [f"{s} {v} over {o} {adv}." for s, v, o, adv
                  in product(self._S, self._V, self._O, self._ADV)]
        return {"short": short, "medium": medium, "long": long}


# ─────────────────────────────────────────────────────────────────────────────
# NumericGenerator  (CTRL-NUM)
# ─────────────────────────────────────────────────────────────────────────────

class NumericGenerator(ControlGenerator):
    """
    English sentences containing numbers in non-arithmetic contexts
    (platform numbers, times, identifiers, measurements).

    Numbers are drawn from the same [10, 50] range as arithmetic stimuli
    so that numeric magnitude is not a confounding factor in CKA comparisons.

    Tier construction:
      short  : 4 templates × 41 values          =   164 texts
      medium : 4 templates × 41 × 41 combos     ≈  6 724 texts (sampled)
      long   : 4 templates × 41 × 41 × 10       ≈ many  (sampled)
    """

    CATEGORY  = "CTRL-NUM"
    ID_PREFIX = "CTRL-NUM"

    _N  = list(range(10, 51))          # 41 values — matches arithmetic domain
    _N2 = list(range(10, 51))
    _CITY = [
        "Paris", "Vienna", "Berlin", "Madrid", "Zurich",
        "London", "Prague", "Lisbon", "Warsaw", "Dublin",
    ]

    def _build_pool_by_tier(self) -> Dict[str, List[str]]:
        # Short: single-number sentences (4–5 tokens on GPT-NeoX)
        short_templates = [
            "Platform {a} is closed.",
            "Floor {a} is restricted.",
            "Gate {a} has boarded.",
            "Ward {a} is full.",
            "Room {a} is vacant.",
        ]
        short = [t.format(a=a)
                 for t in short_templates
                 for a in self._N]

        # Medium: two-number sentences (6–7 tokens)
        medium_templates = [
            "The train departs at {a}:{b:02d}.",
            "Room {a} fits {b} guests.",
            "Section {a} holds {b} seats.",
            "Line {a} stops at junction {b}.",
        ]
        medium = [t.format(a=a, b=b)
                  for t in medium_templates
                  for a in self._N
                  for b in self._N2]

        # Long: multi-number sentences (8–9 tokens)
        long_templates = [
            "Flight {a} departs from gate {b} to {city}.",
            "Contract {a} covers {b} months and expires in {city}.",
            "Route {a} has {b} stops before reaching {city}.",
            "Unit {a} shares {b} resources with the {city} hub.",
        ]
        long = [t.format(a=a, b=b, city=c)
                for t in long_templates
                for a in self._N
                for b in self._N2
                for c in self._CITY]

        return {"short": short, "medium": medium, "long": long}


# ─────────────────────────────────────────────────────────────────────────────
# Length matching
# ─────────────────────────────────────────────────────────────────────────────

def length_match_to_arithmetic(
    ctrl_stimuli: List[Stimulus],
    arith_stimuli: List[Stimulus],
    seed: int = 0,
) -> List[Stimulus]:
    """
    Resample ``ctrl_stimuli`` so that the distribution of
    ``token_length_strata`` mirrors that of ``arith_stimuli``.

    Both collections must already be tokenised (token_fields.n_tokens ≠ None).
    Returns a new list of the same length as ``ctrl_stimuli``.

    Raises RuntimeError if any stratum in the arithmetic distribution cannot
    be matched from the control pool.
    """
    # Compute arithmetic stratum distribution (proportions).
    arith_strata = Counter(
        s.token_fields.token_length_strata for s in arith_stimuli
        if s.token_fields.token_length_strata is not None
    )
    total_arith = sum(arith_strata.values())
    if total_arith == 0:
        raise RuntimeError("arith_stimuli are not tokenised; run populate_token_fields first.")

    n_ctrl = len(ctrl_stimuli)
    # Build a stratum → [stimuli] map for the control pool.
    ctrl_by_strata: Dict[str, List[Stimulus]] = {}
    for s in ctrl_stimuli:
        k = s.token_fields.token_length_strata or "unknown"
        ctrl_by_strata.setdefault(k, []).append(s)

    # Compute target counts for each stratum.
    target: Dict[str, int] = {}
    allocated = 0
    strata_sorted = sorted(arith_strata.keys())
    for i, stratum in enumerate(strata_sorted):
        if i < len(strata_sorted) - 1:
            count = round(n_ctrl * arith_strata[stratum] / total_arith)
        else:
            count = n_ctrl - allocated   # absorb rounding residual
        target[stratum] = count
        allocated += count

    rng = random.Random(seed)
    matched: List[Stimulus] = []
    for stratum, count in target.items():
        pool = ctrl_by_strata.get(stratum, [])
        if len(pool) < count:
            raise RuntimeError(
                f"length_match: stratum '{stratum}' needs {count} control stimuli "
                f"but pool only has {len(pool)}.  "
                "Increase n_stimuli or widen the word lists."
            )
        matched.extend(rng.sample(pool, count))

    rng.shuffle(matched)
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible wrappers
# ─────────────────────────────────────────────────────────────────────────────

def generate_neutral_stimuli(n: int = 500, seed: int = 84) -> List[Stimulus]:
    return NeutralGenerator(seed=seed).build(n)


def generate_numeric_stimuli(n: int = 500, seed: int = 42) -> List[Stimulus]:
    return NumericGenerator(seed=seed).build(n)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CTRL-NEU or CTRL-NUM control stimuli (v5)."
    )
    parser.add_argument("--category",   choices=["neu", "num"], required=True)
    parser.add_argument("--n_stimuli",  type=int, default=500)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--tokenizer",  type=str, default=None,
                        help="HuggingFace tokenizer for length matching and "
                             "token-field population.")
    parser.add_argument("--arith_jsonl", type=str, default=None,
                        help="Path to tokenised arithmetic JSONL for length "
                             "matching.  Required when --tokenizer is given.")
    parser.add_argument("--output",     type=str, default=None)
    args = parser.parse_args()

    gen = (NeutralGenerator(seed=args.seed)
           if args.category == "neu"
           else NumericGenerator(seed=args.seed))
    default_out = f"data/raw/stimuli_ctrl_{args.category}_v5.jsonl"

    print(f"Generating {args.n_stimuli} {gen.CATEGORY} stimuli …")
    stimuli = gen.build(args.n_stimuli)

    if args.tokenizer:
        print(f"Tokenising with {args.tokenizer} …")
        stimuli = populate_token_fields(stimuli, args.tokenizer)

        if args.arith_jsonl:
            import json as _json
            print(f"Length-matching to {args.arith_jsonl} …")
            raw = Path(args.arith_jsonl).read_text(encoding="utf-8").strip().split("\n")

            # Reconstruct minimal arith stubs for length_match (only needs strata).
            from dataclasses import fields as dc_fields
            arith_stubs: List[Stimulus] = []
            for line in raw:
                obj = _json.loads(line)
                tf_raw = obj.get("token_fields", {})
                stub_tf = TokenFields(
                    tokenizer_name      = tf_raw.get("tokenizer_name"),
                    n_tokens            = tf_raw.get("n_tokens"),
                    token_length_strata = tf_raw.get("token_length_strata"),
                )
                # We only need token_length_strata, so a dummy Stimulus works.
                arith_stubs.append(replace(stimuli[0], token_fields=stub_tf))

            stimuli = length_match_to_arithmetic(stimuli, arith_stubs, seed=args.seed)
            print(f"  After length match: {len(stimuli)} stimuli retained.")

    out = write_jsonl(args.output or default_out, stimuli)
    print(f"✓  {len(stimuli)} stimuli written to {out}")


if __name__ == "__main__":
    main()