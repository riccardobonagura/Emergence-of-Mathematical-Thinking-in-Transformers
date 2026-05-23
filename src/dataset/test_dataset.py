"""
test_dataset.py  —  v5
=======================
Test suite for the v5 dataset (sign + parity probing, Pythia-1.4B target).

Sections
--------
  1. UNIT        — arithmetic computation, label assignment
  2. INTEGRATION — generation + tokenisation + validate_dataset
  3. STATISTICAL — balance, pair coherence, template distribution
  4. ROUND-TRIP  — JSONL serialisation / deserialisation
  5. CONTROL     — schema, length distribution, no duplicates, merge compat

Run
---
    python test_dataset.py --tokenizer EleutherAI/pythia-1.4b
    python test_dataset.py --tokenizer EleutherAI/pythia-1.4b --n_pairs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

try:
    from build_stimuli import (
        DATASET_VERSION,
        ParityContrastGenerator,
        SignContrastGenerator,
        Stimulus,
        populate_token_fields,
        validate_dataset,
        write_jsonl,
    )
except ImportError as e:
    print(f"[ERROR] Cannot import build_stimuli: {e}")
    sys.exit(1)

try:
    from build_control import (
        NeutralGenerator,
        NumericGenerator,
        generate_neutral_stimuli,
        generate_numeric_stimuli,
        length_match_to_arithmetic,
    )
except ImportError as e:
    print(f"[ERROR] Cannot import build_control: {e}")
    sys.exit(1)

try:
    from merge_stimuli import REQUIRED_ROOT_KEYS, validate_schema
except ImportError as e:
    print(f"[ERROR] Cannot import merge_stimuli: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    print(f"  {tag} {name}")
    if not condition and detail:
        print(f"      → {detail}")
    results["passed" if condition else "failed"] += 1
    return condition


def section(title: str) -> None:
    print(f"\n{'─' * 62}\n  {title}\n{'─' * 62}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIT
# ─────────────────────────────────────────────────────────────────────────────

def test_unit() -> None:
    section("1. UNIT — Arithmetic computation and label assignment")

    sg = SignContrastGenerator()
    pg = ParityContrastGenerator()

    # sign: 0 = non-negative, 1 = negative
    for a, b, expected_sign in [(20, 7, 0), (7, 20, 1), (15, 15, 0)]:
        result = a - b
        got = 1 if result < 0 else 0
        check(f"sign({a} - {b} = {result}) == {expected_sign}", got == expected_sign,
              f"got {got}")

    # parity: 0 = even, 1 = odd
    for x, expected_par in [(20, 0), (21, 1), (0, 0), (99, 1)]:
        got = abs(x) % 2
        check(f"parity({x}) == {expected_par}", got == expected_par, f"got {got}")

    # CAT-SIGN pair: |result_A| == |result_B|
    pair = sg._make_pair("test-0000", 30, 12, "TPL-SIGN-1", "{a} - {b} =")
    check("CAT-SIGN pair: |result| identical",
          abs(pair[0].labels.result) == abs(pair[1].labels.result),
          f"{pair[0].labels.result} vs {pair[1].labels.result}")
    check("CAT-SIGN pair: signs are opposite",
          pair[0].labels.sign != pair[1].labels.sign)
    check("CAT-SIGN pair: parities are identical",
          pair[0].labels.parity == pair[1].labels.parity)

    # CAT-PARITY pair: results differ by exactly 1
    ppair = pg._make_pair("test-0000", 20, 14, "TPL-PAR-1", "{a} + {b} =")
    diff = abs(ppair[0].labels.result - ppair[1].labels.result)
    check("CAT-PARITY pair: results differ by 1", diff == 1, f"diff = {diff}")
    check("CAT-PARITY pair: parities are opposite",
          ppair[0].labels.parity != ppair[1].labels.parity)
    check("CAT-PARITY pair: signs are both non-negative",
          ppair[0].labels.sign == 0 and ppair[1].labels.sign == 0)

    # operand_digit_class
    check("digit_class [10,50] → '2d_2d'",
          pair[0].operand_digit_class == "2d_2d",
          f"got {pair[0].operand_digit_class!r}")

    # dataset_version
    check(f"dataset_version == '{DATASET_VERSION}'",
          pair[0].dataset_version == DATASET_VERSION)

    # sentinel: no -1 in token_fields
    check("Unpopulated TokenFields: no -1 sentinel anywhere",
          pair[0].token_fields.equals_sign_index is None and
          pair[0].token_fields.last_token_index is None)


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def test_integration(n_pairs: int, tokenizer_name: str) -> List[Stimulus]:
    section("2. INTEGRATION — Generation + tokenisation + validate_dataset")

    sg = SignContrastGenerator()
    pg = ParityContrastGenerator()
    sign_stims = sg.build(n_pairs, seed=42)
    par_stims  = pg.build(n_pairs, seed=43)
    all_stims  = sign_stims + par_stims

    check(f"Generated {2 * n_pairs} stimuli per category",
          len(sign_stims) == 2 * n_pairs and len(par_stims) == 2 * n_pairs)
    check("All splits are 'geometric_eval'",
          all(s.split == "geometric_eval" for s in all_stims))
    check("All operand_digit_class are '2d_2d'",
          all(s.operand_digit_class == "2d_2d" for s in all_stims))
    check("probe_layer_strategy == 'all_layers'",
          all(s.probe_layer_strategy == "all_layers" for s in all_stims))

    print(f"\n  Loading tokenizer: {tokenizer_name} …")
    try:
        tokenised = populate_token_fields(all_stims, tokenizer_name)
    except Exception as exc:
        check("Tokenisation succeeded", False, str(exc))
        return all_stims

    check("All n_tokens populated",
          all(s.token_fields.n_tokens is not None for s in tokenised))
    check("tokenizer_name stored in every TokenFields",
          all(s.token_fields.tokenizer_name == tokenizer_name for s in tokenised))
    check("equals_sign_index == last_token_index for all CAT stimuli",
          all(
              s.token_fields.equals_sign_index == s.token_fields.last_token_index
              for s in tokenised if s.category.startswith("CAT-")
          ))
    check("No -1 sentinel in any token_fields",
          all(s.token_fields.equals_sign_index != -1 for s in tokenised))

    try:
        validate_dataset(tokenised)
        check("validate_dataset() passes all internal checks", True)
    except AssertionError as exc:
        check("validate_dataset() passes all internal checks", False, str(exc))

    return tokenised


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL
# ─────────────────────────────────────────────────────────────────────────────

def test_statistical(dataset: List[Stimulus]) -> None:
    section("3. STATISTICAL — Balance and pair coherence")

    sign_stims = [s for s in dataset if s.category == "CAT-SIGN"]
    par_stims  = [s for s in dataset if s.category == "CAT-PARITY"]

    # Sign balance
    sign_dist = Counter(s.labels.sign for s in sign_stims)
    check("CAT-SIGN: exact 50/50 sign balance",
          sign_dist[0] == sign_dist[1],
          f"sign distribution: {dict(sign_dist)}")

    # Parity balance
    par_dist = Counter(s.labels.parity for s in par_stims)
    check("CAT-PARITY: exact 50/50 parity balance",
          par_dist[0] == par_dist[1],
          f"parity distribution: {dict(par_dist)}")

    # Pair coherence
    for cat_stims, cat in [(sign_stims, "CAT-SIGN"), (par_stims, "CAT-PARITY")]:
        pair_counts = Counter(s.contrast.pair_id for s in cat_stims)
        check(f"{cat}: every pair_id appears exactly twice",
              all(v == 2 for v in pair_counts.values()),
              f"odd counts: {[k for k, v in pair_counts.items() if v != 2][:3]}")

    # Template balance (round-robin: difference ≤ 2)
    for cat_stims, cat in [(sign_stims, "CAT-SIGN"), (par_stims, "CAT-PARITY")]:
        tpl_counts = Counter(s.template_id for s in cat_stims)
        vals = sorted(tpl_counts.values())
        check(f"{cat}: template counts balanced (max spread ≤ 2)",
              vals[-1] - vals[0] <= 2,
              f"template distribution: {dict(tpl_counts)}")

    # No cross-category ID collisions
    all_ids = [s.id for s in dataset]
    check("No duplicate stimulus IDs across categories",
          len(all_ids) == len(set(all_ids)))

    # extraction_strategy_by_property keys
    check("All stimuli: extraction_strategy has 'sign' and 'parity' keys",
          all(
              set(s.extraction_strategy_by_property.keys()) >= {"sign", "parity"}
              for s in dataset
          ))

    # CAT-SIGN: parity of |result| is same within each pair
    pair_map: dict = {}
    for s in sign_stims:
        pair_map.setdefault(s.contrast.pair_id, []).append(s)
    parity_ok = all(
        p[0].labels.parity == p[1].labels.parity
        for p in pair_map.values() if len(p) == 2
    )
    check("CAT-SIGN: parity(|result|) identical within each pair", parity_ok)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROUND-TRIP
# ─────────────────────────────────────────────────────────────────────────────

def test_roundtrip(dataset: List[Stimulus], tmp_path: Path) -> None:
    section("4. ROUND-TRIP — JSONL serialisation")

    out_file = tmp_path / "roundtrip.jsonl"
    write_jsonl(out_file, dataset)

    required_root = REQUIRED_ROOT_KEYS
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    check(f"JSONL line count matches ({len(dataset)})",
          len(lines) == len(dataset))

    errors = []
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {i}: {exc}")
            continue
        missing = required_root - obj.keys()
        if missing:
            errors.append(f"line {i} ({obj.get('id','?')}): missing {missing}")

    check("All lines valid JSON with required root keys", not errors,
          "; ".join(errors[:3]))

    # Type preservation: sign and parity must remain int after JSON round-trip.
    type_errors = []
    for line in lines:
        obj = json.loads(line)
        for key in ("sign", "parity"):
            val = obj["labels"][key]
            if not isinstance(val, int):
                type_errors.append(f"{obj['id']}.labels.{key} = {val!r} ({type(val).__name__})")
    check("labels.sign and labels.parity are int after round-trip",
          not type_errors, "; ".join(type_errors[:3]))

    # Sentinel: no -1 anywhere in token_fields
    sentinel_errors = [
        obj["id"]
        for line in lines
        for obj in [json.loads(line)]
        if obj["token_fields"].get("equals_sign_index") == -1
    ]
    check("No -1 sentinel in token_fields after round-trip",
          not sentinel_errors, str(sentinel_errors[:3]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def test_control(tokenizer_name: str, arith_dataset: List[Stimulus]) -> None:
    section("5. CONTROL — Schema, length distribution, no duplicates")

    neu_stimuli = generate_neutral_stimuli(n=90, seed=84)
    num_stimuli = generate_numeric_stimuli(n=90, seed=42)

    for label, ctrl in [("CTRL-NEU", neu_stimuli), ("CTRL-NUM", num_stimuli)]:

        # 5a. Schema: required root keys present
        missing = [
            (s.id, REQUIRED_ROOT_KEYS - s.to_dict().keys())
            for s in ctrl if REQUIRED_ROOT_KEYS - s.to_dict().keys()
        ]
        check(f"{label} 5a: all required root keys present",
              not missing, str(missing[:2]))

        # 5b. Sentinel: equals_sign_index is None (not -1) for CTRL
        bad_sentinel = [s.id for s in ctrl
                        if s.token_fields.equals_sign_index == -1]
        check(f"{label} 5b: equals_sign_index is None (not -1)",
              not bad_sentinel, str(bad_sentinel[:3]))

        # 5c. extraction_strategy keys
        check(f"{label} 5c: extraction_strategy has sign + parity",
              all(
                  {"sign", "parity"} <= set(s.extraction_strategy_by_property.keys())
                  for s in ctrl
              ))

        # 5d. Same root schema as arithmetic stimuli
        arith_keys = set(arith_dataset[0].to_dict().keys())
        ctrl_keys  = set(ctrl[0].to_dict().keys())
        sym_diff   = arith_keys.symmetric_difference(ctrl_keys)
        check(f"{label} 5d: identical root keys to arithmetic stimuli",
              not sym_diff, f"symmetric difference: {sym_diff}")

        # 5e. JSON round-trip preserves int types
        type_errors = [
            f"{s.id}.labels.{k}"
            for s in ctrl
            for k in ("sign", "parity")
            if not isinstance(json.loads(s.to_json())["labels"][k], int)
        ]
        check(f"{label} 5e: sign/parity are int after JSON round-trip",
              not type_errors, str(type_errors[:3]))

        # 5f. No duplicate texts
        texts = [s.text for s in ctrl]
        n_dup = len(texts) - len(set(texts))
        check(f"{label} 5f: zero duplicate texts ({len(texts)} stimuli)",
              n_dup == 0, f"{n_dup} duplicates found")

        # 5g. validate_schema from merge_stimuli passes (pre-tokenisation)
        schema_errors = [
            (s.id, msg)
            for s in ctrl
            for ok, msg in [validate_schema(s.to_dict(), require_tokenized=False)]
            if not ok
        ]
        check(f"{label} 5g: validate_schema passes (untokenised)",
              not schema_errors, str(schema_errors[:2]))

    # 5h. Tokenisation + length-matching
    print(f"\n  Tokenising control stimuli with {tokenizer_name} …")
    try:
        tok_neu = populate_token_fields(neu_stimuli, tokenizer_name)
        tok_num = populate_token_fields(num_stimuli, tokenizer_name)
        check("5h: tokenisation of control stimuli succeeded", True)
    except Exception as exc:
        check("5h: tokenisation of control stimuli succeeded", False, str(exc))
        return

    # Length-matching test: after matching, strata distribution should agree.
    tok_arith = [s for s in arith_dataset
                 if s.token_fields.token_length_strata is not None]
    if tok_arith:
        try:
            matched_neu = length_match_to_arithmetic(tok_neu, tok_arith, seed=0)
            arith_strata = Counter(
                s.token_fields.token_length_strata for s in tok_arith
            )
            match_strata = Counter(
                s.token_fields.token_length_strata for s in matched_neu
            )
            # After matching, strata keys should agree.
            same_keys = set(arith_strata.keys()) == set(match_strata.keys())
            check("5h: length-matched CTRL-NEU has same strata keys as arithmetic",
                  same_keys,
                  f"arith={set(arith_strata)}, matched={set(match_strata)}")
        except RuntimeError as exc:
            check("5h: length_match_to_arithmetic", False, str(exc))
    else:
        print("  (skipping length-match test: arithmetic stimuli not tokenised)")

    # 5i. validate_schema passes post-tokenisation
    for label, tok_ctrl in [("CTRL-NEU", tok_neu), ("CTRL-NUM", tok_num)]:
        schema_errors = [
            (s.id, msg)
            for s in tok_ctrl
            for ok, msg in [validate_schema(s.to_dict(), require_tokenized=True)]
            if not ok
        ]
        check(f"{label} 5i: validate_schema passes post-tokenisation",
              not schema_errors, str(schema_errors[:2]))

        # Token-length strata all present (short/medium/long)
        strata = Counter(s.token_fields.token_length_strata for s in tok_ctrl)
        missing_tiers = [t for t in ("short", "medium", "long")
                         if strata.get(t, 0) == 0]
        check(f"{label} 5i+: all three length strata present",
              not missing_tiers,
              f"missing: {missing_tiers}, distribution: {dict(strata)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. METADATA
# ─────────────────────────────────────────────────────────────────────────────

def test_metadata_completeness() -> None:
    section("6. METADATA — completeness contract for downstream consumers")
    # These fields are required by run_rq2, run_rq3, MetadataHandler, and cka.py.
    REQUIRED_METADATA_KEYS = {
        "stimuli_ids", "categories", "probe_strategy", "dataset_version",
        "n_layers", "d_model", "n_stimuli",
        "labels",    # dict with "sign" and "parity" sub-keys
    }
    REQUIRED_LABEL_FIELDS = {"sign", "parity"}

    import json
    import pathlib
    import tempfile

    from build_stimuli import SignContrastGenerator
    from src.extraction.extract_states import save_extraction_metadata

    stims = SignContrastGenerator().build(n_pairs=2, seed=0)

    class _Cfg:
        n_layers = 4
        d_model  = 8
    class _Model:
        cfg = _Cfg()

    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td)
        raw = [s.to_dict() for s in stims]
        save_extraction_metadata(raw, out, _Model())
        meta = json.loads((out / "metadata.json").read_text())

    missing_keys = REQUIRED_METADATA_KEYS - meta.keys()
    check("6a: all required root keys present in metadata",
          not missing_keys, f"Missing: {missing_keys}")

    if "labels" in meta:
        missing_label_fields = REQUIRED_LABEL_FIELDS - meta["labels"].keys()
        check("6b: labels block has sign and parity fields",
              not missing_label_fields, f"Missing: {missing_label_fields}")

        check("6c: labels arrays parallel to stimuli_ids",
              len(meta["labels"]["sign"]) == len(meta["stimuli_ids"]) and
              len(meta["labels"]["parity"]) == len(meta["stimuli_ids"]))

    check("6d: n_stimuli == len(stimuli_ids)",
          meta.get("n_stimuli") == len(meta.get("stimuli_ids", [])))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test suite for the v5 dataset."
    )
    parser.add_argument("--n_pairs",   type=int, default=50,
                        help="Pairs per category to generate in tests (default 50).")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-1.4b",
                        help="HuggingFace tokenizer to use (default: Pythia-1.4B).")
    parser.add_argument("--tmp_dir",   type=str, default="/tmp/test_dataset_v5")
    args = parser.parse_args()

    tmp_path = Path(args.tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

    test_unit()
    dataset = test_integration(args.n_pairs, args.tokenizer)
    test_statistical(dataset)
    test_roundtrip(dataset, tmp_path)
    test_control(args.tokenizer, dataset)
    test_metadata_completeness()

    total = results["passed"] + results["failed"]
    colour = "\033[92m" if results["failed"] == 0 else "\033[91m"
    print(f"\n{colour}  Results: {results['passed']}/{total} tests passed\033[0m")
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()