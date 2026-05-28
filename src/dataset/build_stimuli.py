"""
build_stimuli.py — v5
Property-contrastive arithmetic stimuli for Pythia-1.4B geometric analysis.

Provides:
  - SignContrastGenerator   (CAT-SIGN, sign-of-result contrast)
  - ParityContrastGenerator (CAT-PARITY, parity-of-result contrast)
  - populate_token_fields() (HF tokenisation pass — required before merge_stimuli)
  - validate_dataset()      (post-generation structural invariants)
  - write_jsonl()           (atomic-style serialiser)

Sign labels:   1 = positive, 0 = negative, -1 = sentinel (CAT-PARITY, CTRL-*)
Parity labels: 0 = even, 1 = odd, -1 = sentinel (CTRL-*)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

from src.probing.seeds import get_seed

DATASET_VERSION = "v5"


class Labels(TypedDict):
    result: int
    sign: int     # 1 = positive, 0 = negative, -1 = sentinel
    parity: int   # 0 = even, 1 = odd, -1 = sentinel
    operand1: int
    operand2: int


class Contrast(TypedDict):
    pair_id: str
    varying_axis: str
    controlled_axes: Tuple[str, ...]


class TokenFields(TypedDict, total=False):
    tokenizer_name: str
    n_tokens: int
    token_ids: Tuple[int, ...]
    token_strs: Tuple[str, ...]
    token_length_strata: str
    equals_sign_index: int
    last_token_index: int


class Stimulus(TypedDict):
    id: str
    text: str
    split: str
    category: str
    template_id: str
    macro_format: str
    n_reasoning_steps: int
    labels: Labels
    contrast: Contrast
    token_fields: TokenFields
    ood_target: str
    dataset_version: str
    probe_layer_strategy: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def _token_length_strata(n: int) -> str:
    if n <= 5:
        return "short"
    if n <= 8:
        return "medium"
    return "long"


# ── Generators ───────────────────────────────────────────────────────────────

class SignContrastGenerator:
    CATEGORY = "CAT-SIGN"
    TEMPLATES = [
        "{a} - {b} =",
        "Compute: {a} - {b} =",
        "Calculate {a} - {b} =",
    ]

    def build(self, n_pairs: int, seed: int) -> List[Stimulus]:
        # Deterministic RNG seeded via project-wide get_seed discipline.
        import random
        rng = random.Random(seed)

        stimuli: List[Stimulus] = []
        pairs_generated = 0
        seen = set()

        while pairs_generated < n_pairs:
            a = rng.randint(10, 50)
            b = rng.randint(10, 50)

            key = (max(a, b), min(a, b))
            if a == b or key in seen:
                continue
            seen.add(key)

            if a < b:
                a, b = b, a

            tpl_idx = pairs_generated % len(self.TEMPLATES)
            tpl = self.TEMPLATES[tpl_idx]
            pair_id = f"SIGN-{pairs_generated:04d}"

            # A: a − b → positive (sign=1); B: b − a → negative (sign=0).
            stim_a = self._make_stim(
                f"{pair_id}-A", tpl.format(a=a, b=b), pair_id,
                res=a - b, sign=1, parity=(a - b) % 2, op1=a, op2=b, tpl_idx=tpl_idx,
            )
            stim_b = self._make_stim(
                f"{pair_id}-B", tpl.format(a=b, b=a), pair_id,
                res=b - a, sign=0, parity=abs(b - a) % 2, op1=b, op2=a, tpl_idx=tpl_idx,
            )

            stimuli.extend([stim_a, stim_b])
            pairs_generated += 1

        return stimuli

    def _make_stim(
        self, sid: str, text: str, pair_id: str,
        res: int, sign: int, parity: int, op1: int, op2: int, tpl_idx: int,
    ) -> Stimulus:
        return {
            "id": sid,
            "text": text,
            "split": "geometric_eval",
            "category": self.CATEGORY,
            "template_id": f"TPL-SIGN-{tpl_idx + 1}",
            "macro_format": "symbolic_arithmetic",
            "n_reasoning_steps": 1,
            "labels": {
                "result": res,
                "sign": sign,
                "parity": parity,
                "operand1": op1,
                "operand2": op2,
            },
            "contrast": {
                "pair_id": pair_id,
                "varying_axis": "sign",
                "controlled_axes": ("operator", "operands_abs", "result_abs", "template"),
            },
            "token_fields": {},
            "ood_target": "in_distribution",
            "dataset_version": DATASET_VERSION,
            "probe_layer_strategy": "all_layers",
        }


class ParityContrastGenerator:
    CATEGORY = "CAT-PARITY"
    ADD_TEMPLATES = ["{a} + {b} =", "Compute: {a} + {b} =", "Calculate {a} + {b} ="]
    SUB_TEMPLATES = ["{a} - {b} =", "Compute: {a} - {b} =", "Calculate {a} - {b} ="]

    def build(self, n_pairs: int, seed: int) -> List[Stimulus]:
        import random
        rng = random.Random(seed)

        stimuli: List[Stimulus] = []
        seen = set()

        n_add = n_pairs // 2
        n_sub = n_pairs - n_add

        # Addition pool — b vs b+1 flips parity at fixed first operand.
        add_generated = 0
        while add_generated < n_add:
            a = rng.randint(10, 50)
            b = rng.randint(10, 49)  # b+1 ∈ [11,50] stays within single-token range
            key = ("add", a, b)
            if key in seen:
                continue
            seen.add(key)

            tpl_idx = add_generated % len(self.ADD_TEMPLATES)
            tpl = self.ADD_TEMPLATES[tpl_idx]
            pair_id = f"PAR-{add_generated:04d}"

            stimuli.extend(self._make_parity_pair(
                pair_id, tpl, a, b, op="add", tpl_idx=tpl_idx,
            ))
            add_generated += 1

        # Subtraction pool — same mechanism, but a > b strictly so result ≥ 1.
        # Fix #8: b ∈ [10, a-1] includes the tight a=b+1 case (res=1, single token).
        sub_generated = 0
        while sub_generated < n_sub:
            a = rng.randint(11, 50)
            b = rng.randint(10, a - 1)
            key = ("sub", a, b)
            if key in seen:
                continue
            seen.add(key)

            tpl_idx = sub_generated % len(self.SUB_TEMPLATES)
            tpl = self.SUB_TEMPLATES[tpl_idx]
            pair_id = f"PAR-{(n_add + sub_generated):04d}"

            stimuli.extend(self._make_parity_pair(
                pair_id, tpl, a, b, op="sub", tpl_idx=tpl_idx,
            ))
            sub_generated += 1

        return stimuli

    def _make_parity_pair(
        self, pair_id: str, tpl: str, a: int, b: int, op: str, tpl_idx: int,
    ) -> List[Stimulus]:
        """
        Build (-A, -B) deterministically so that -A is always the even-parity
        member and -B the odd-parity member. Fix #9: removes the silent post-hoc
        swap that previously broke pair_id audit semantics.
        """
        if op == "add":
            res_b, res_bp1 = a + b, a + (b + 1)
            text_b, text_bp1 = tpl.format(a=a, b=b), tpl.format(a=a, b=b + 1)
            op2_b, op2_bp1 = b, b + 1
        else:  # sub
            res_b, res_bp1 = a - b, a - (b + 1)
            text_b, text_bp1 = tpl.format(a=a, b=b), tpl.format(a=a, b=b + 1)
            op2_b, op2_bp1 = b, b + 1

        # Pick the even-parity result as A and the odd-parity result as B.
        if res_b % 2 == 0:
            even_res, even_text, even_op2 = res_b, text_b, op2_b
            odd_res, odd_text, odd_op2 = res_bp1, text_bp1, op2_bp1
        else:
            even_res, even_text, even_op2 = res_bp1, text_bp1, op2_bp1
            odd_res, odd_text, odd_op2 = res_b, text_b, op2_b

        prefix = "TPL-PAR-ADD" if op == "add" else "TPL-PAR-SUB"
        stim_a = self._make_stim(
            f"{pair_id}-A", even_text, pair_id,
            res=even_res, parity=0, op1=a, op2=even_op2, tpl_idx=tpl_idx, tpl_prefix=prefix,
        )
        stim_b = self._make_stim(
            f"{pair_id}-B", odd_text, pair_id,
            res=odd_res, parity=1, op1=a, op2=odd_op2, tpl_idx=tpl_idx, tpl_prefix=prefix,
        )
        return [stim_a, stim_b]

    def _make_stim(
        self, sid: str, text: str, pair_id: str,
        res: int, parity: int, op1: int, op2: int, tpl_idx: int, tpl_prefix: str,
    ) -> Stimulus:
        # CAT-PARITY: sign is sentinel — no ground-truth sign contrast here.
        return {
            "id": sid,
            "text": text,
            "split": "geometric_eval",
            "category": self.CATEGORY,
            "template_id": f"{tpl_prefix}-{tpl_idx + 1}",
            "macro_format": "symbolic_arithmetic",
            "n_reasoning_steps": 1,
            "labels": {
                "result": res,
                "sign": -1,
                "parity": parity,
                "operand1": op1,
                "operand2": op2,
            },
            "contrast": {
                "pair_id": pair_id,
                "varying_axis": "parity",
                "controlled_axes": ("operator", "operand_a", "template"),
            },
            "token_fields": {},
            "ood_target": "in_distribution",
            "dataset_version": DATASET_VERSION,
            "probe_layer_strategy": "all_layers",
        }


# ── Tokenisation ─────────────────────────────────────────────────────────────

def populate_token_fields(
    stimuli: List[Stimulus],
    tokenizer_name: str,
) -> List[Stimulus]:
    """
    Populate TokenFields for every stimulus using a HuggingFace tokenizer.

    For CAT-SIGN / CAT-PARITY stimuli the "=" token must be the last token
    (enforced by assertion — the extraction strategy depends on it).
    For CTRL-* stimuli (no "=") equals_sign_index is set to None.

    Mutates input dicts in place and returns the same list — callers may
    discard the return value.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for tokenisation. "
            "Install with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    for s in stimuli:
        ids = tokenizer.encode(s["text"], add_special_tokens=False)
        strs = tokenizer.convert_ids_to_tokens(ids)
        n = len(ids)
        last = n - 1

        # Locate "=" — strip GPT-NeoX leading-space marker "Ġ" before comparing.
        eq_idx = None
        for i, tok in enumerate(strs):
            if tok.replace("Ġ", "").replace("▁", "").strip() == "=":
                eq_idx = i

        if s["category"].startswith("CAT-"):
            assert eq_idx == last, (
                f"Stimulus {s['id']}: expected '=' at index {last} (last token), "
                f"found at {eq_idx}.\nText: {s['text']!r}\nTokens: {strs}"
            )

        s["token_fields"] = {
            "tokenizer_name": tokenizer_name,
            "n_tokens": n,
            "token_ids": tuple(ids),
            "token_strs": tuple(strs),
            "token_length_strata": _token_length_strata(n),
            "equals_sign_index": eq_idx,
            "last_token_index": last,
        }

    return stimuli


# ── Validation ───────────────────────────────────────────────────────────────

def validate_dataset(stimuli: List[Stimulus]) -> None:
    """
    Assert structural invariants on a fully generated (optionally tokenised)
    dataset. Raises AssertionError with a descriptive message on failure.
    """
    from collections import Counter

    sign_stimuli = [s for s in stimuli if s["category"] == "CAT-SIGN"]
    par_stimuli = [s for s in stimuli if s["category"] == "CAT-PARITY"]

    # 50/50 sign balance in CAT-SIGN.
    if sign_stimuli:
        sign_dist = Counter(s["labels"]["sign"] for s in sign_stimuli)
        assert sign_dist[0] == sign_dist[1], (
            f"CAT-SIGN imbalance: {dict(sign_dist)}"
        )

    # 50/50 parity balance in CAT-PARITY.
    if par_stimuli:
        par_dist = Counter(s["labels"]["parity"] for s in par_stimuli)
        assert par_dist[0] == par_dist[1], (
            f"CAT-PARITY imbalance: {dict(par_dist)}"
        )

    # Each pair_id appears exactly twice (one -A, one -B).
    for cat_stims, cat in [(sign_stimuli, "CAT-SIGN"), (par_stimuli, "CAT-PARITY")]:
        pair_counts = Counter(s["contrast"]["pair_id"] for s in cat_stims)
        odd = [k for k, v in pair_counts.items() if v != 2]
        assert not odd, f"{cat}: pair_ids without exactly 2 members: {odd[:3]}"

    # No duplicate stimulus IDs.
    all_ids = [s["id"] for s in stimuli]
    assert len(all_ids) == len(set(all_ids)), "Duplicate stimulus IDs detected."

    # CAT-PARITY uses sentinel sign (-1) — no real sign label.
    for s in par_stimuli:
        assert s["labels"]["sign"] == -1, (
            f"{s['id']}: CAT-PARITY must use sign sentinel -1, got {s['labels']['sign']}"
        )

    # CAT-SIGN: parity within each pair must be identical (a-b and b-a have same |result|).
    pair_map: Dict[str, List[Stimulus]] = {}
    for s in sign_stimuli:
        pair_map.setdefault(s["contrast"]["pair_id"], []).append(s)
    for pid, members in pair_map.items():
        if len(members) == 2:
            assert members[0]["labels"]["parity"] == members[1]["labels"]["parity"], (
                f"CAT-SIGN pair {pid}: parities differ — expected identical |result| parity"
            )

    # CAT-PARITY: -A is even, -B is odd (Fix #9 deterministic ordering).
    for s in par_stimuli:
        if s["id"].endswith("-A"):
            assert s["labels"]["parity"] == 0, f"{s['id']}: -A should be even"
        elif s["id"].endswith("-B"):
            assert s["labels"]["parity"] == 1, f"{s['id']}: -B should be odd"


# ── Serialisation ────────────────────────────────────────────────────────────

def write_jsonl(out_path: Path, stimuli: List[Stimulus]) -> None:
    """Atomic-style JSONL writer: temp file + rename to avoid partial reads."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for s in stimuli:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build sign and parity arithmetic stimuli (v5).")
    parser.add_argument("--n_pairs", type=int, default=500, help="Pairs per category (default 500 → 1000 stimuli each).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-1.4b",
                        help="HF tokenizer for populate_token_fields. Empty string skips tokenisation.")
    parser.add_argument("--output", type=str, default="data/raw/stimuli_arithmetic_v5.jsonl")
    args = parser.parse_args()

    # Seed routing through the project's get_seed discipline (S-03 SSOT, CLAUDE.md invariant).
    sign_seed = get_seed(args.seed, "build_stimuli_sign", 0)
    par_seed = get_seed(args.seed, "build_stimuli_parity", 0)

    print(f"Generating {args.n_pairs} pairs per category (base_seed={args.seed}) ...")

    sign_stimuli = SignContrastGenerator().build(args.n_pairs, seed=sign_seed)
    par_stimuli = ParityContrastGenerator().build(args.n_pairs, seed=par_seed)
    all_stimuli: List[Stimulus] = sign_stimuli + par_stimuli

    print(f"  CAT-SIGN   : {len(sign_stimuli):>5} stimuli")
    print(f"  CAT-PARITY : {len(par_stimuli):>5} stimuli")

    if args.tokenizer:
        print(f"Tokenising with {args.tokenizer} ...")
        populate_token_fields(all_stimuli, args.tokenizer)

    validate_dataset(all_stimuli)

    write_jsonl(Path(args.output), all_stimuli)
    print(f"Dataset committed to {args.output}")


if __name__ == "__main__":
    main()
