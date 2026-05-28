"""
test_dataset.py  —  v5
=======================
Manual test script for the v5 dataset (sign + parity probing, Pythia-1.4B target).

Stimuli are plain dicts (TypedDict contracts in build_stimuli.py); all access is
by key. Run as a module from the project root so package imports resolve:

    python -m src.dataset.test_dataset --tokenizer EleutherAI/pythia-1.4b
    python -m src.dataset.test_dataset --tokenizer EleutherAI/pythia-1.4b --n_pairs 50

Sections
--------
  1. UNIT        — arithmetic computation, label assignment, pair invariants
  2. INTEGRATION — generation + tokenisation + validate_dataset
  3. STATISTICAL — balance, pair coherence, template distribution
  4. ROUND-TRIP  — JSONL serialisation / deserialisation
  5. CONTROL     — schema, length distribution, no duplicates, merge compat
  6. METADATA    — dataset → ExtractionMetadata handoff contract
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

try:
    from src.dataset.build_stimuli import (
        DATASET_VERSION,
        ParityContrastGenerator,
        SignContrastGenerator,
        Stimulus,
        populate_token_fields,
        validate_dataset,
        write_jsonl,
    )
    from src.dataset.build_control import (
        generate_neutral_stimuli,
        generate_numeric_stimuli,
        length_match_to_arithmetic,
    )
    from src.dataset.merge_stimuli import REQUIRED_ROOT_KEYS, validate_schema
    from src.extraction.extract_states import ExtractionMetadata
except ImportError as e:
    print(f"[ERROR] Import failed (run with `python -m src.dataset.test_dataset`): {e}")
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


def _by_pair(stimuli: List[Stimulus]) -> Dict[str, List[Stimulus]]:
    """Group stimuli by contrast.pair_id."""
    out: Dict[str, List[Stimulus]] = {}
    for s in stimuli:
        out.setdefault(s["contrast"]["pair_id"], []).append(s)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIT
# ─────────────────────────────────────────────────────────────────────────────

def test_unit() -> None:
    section("1. UNIT — Arithmetic computation and label assignment")

    # sign: 1 = positive result, 0 = negative result (CAT-SIGN convention)
    for a, b, expected_sign in [(20, 7, 1), (7, 20, 0), (50, 10, 1)]:
        result = a - b
        got = 1 if result > 0 else 0
        check(f"sign({a} - {b} = {result}) == {expected_sign}", got == expected_sign,
              f"got {got}")

    # parity: 0 = even, 1 = odd
    for x, expected_par in [(20, 0), (21, 1), (0, 0), (99, 1)]:
        got = abs(x) % 2
        check(f"parity({x}) == {expected_par}", got == expected_par, f"got {got}")

    # CAT-SIGN pair invariants (generated via the public build() API).
    sign_pairs = _by_pair(SignContrastGenerator().build(n_pairs=5, seed=0))
    pid, members = next(iter(sign_pairs.items()))
    a_stim = next(s for s in members if s["id"].endswith("-A"))
    b_stim = next(s for s in members if s["id"].endswith("-B"))
    check("CAT-SIGN pair: |result| identical",
          abs(a_stim["labels"]["result"]) == abs(b_stim["labels"]["result"]),
          f"{a_stim['labels']['result']} vs {b_stim['labels']['result']}")
    check("CAT-SIGN pair: signs are opposite (1 vs 0)",
          {a_stim["labels"]["sign"], b_stim["labels"]["sign"]} == {0, 1})
    check("CAT-SIGN pair: parities are identical",
          a_stim["labels"]["parity"] == b_stim["labels"]["parity"])

    # CAT-PARITY pair invariants.
    par_pairs = _by_pair(ParityContrastGenerator().build(n_pairs=6, seed=0))
    pid, members = next(iter(par_pairs.items()))
    a_stim = next(s for s in members if s["id"].endswith("-A"))
    b_stim = next(s for s in members if s["id"].endswith("-B"))
    diff = abs(a_stim["labels"]["result"] - b_stim["labels"]["result"])
    check("CAT-PARITY pair: results differ by 1", diff == 1, f"diff = {diff}")
    check("CAT-PARITY pair: parities are opposite",
          a_stim["labels"]["parity"] != b_stim["labels"]["parity"])
    check("CAT-PARITY pair: sign is sentinel (-1) on both members",
          a_stim["labels"]["sign"] == -1 and b_stim["labels"]["sign"] == -1)

    # dataset_version
    check(f"dataset_version == '{DATASET_VERSION}'",
          a_stim["dataset_version"] == DATASET_VERSION)

    # Unpopulated token_fields: empty dict, no -1 sentinel anywhere.
    check("Unpopulated token_fields: empty dict (no -1 sentinel)",
          a_stim["token_fields"] == {})


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def test_integration(n_pairs: int, tokenizer_name: str) -> List[Stimulus]:
    section("2. INTEGRATION — Generation + tokenisation + validate_dataset")

    sign_stims = SignContrastGenerator().build(n_pairs, seed=42)
    par_stims  = ParityContrastGenerator().build(n_pairs, seed=43)
    all_stims  = sign_stims + par_stims

    check(f"Generated {2 * n_pairs} stimuli per category",
          len(sign_stims) == 2 * n_pairs and len(par_stims) == 2 * n_pairs)
    check("All splits are 'geometric_eval'",
          all(s["split"] == "geometric_eval" for s in all_stims))
    check("probe_layer_strategy == 'all_layers'",
          all(s["probe_layer_strategy"] == "all_layers" for s in all_stims))

    print(f"\n  Loading tokenizer: {tokenizer_name} …")
    try:
        tokenised = populate_token_fields(all_stims, tokenizer_name)
    except Exception as exc:
        check("Tokenisation succeeded", False, str(exc))
        return all_stims

    check("All n_tokens populated",
          all(s["token_fields"]["n_tokens"] is not None for s in tokenised))
    check("tokenizer_name stored in every token_fields",
          all(s["token_fields"]["tokenizer_name"] == tokenizer_name for s in tokenised))
    check("equals_sign_index == last_token_index for all CAT stimuli",
          all(
              s["token_fields"]["equals_sign_index"] == s["token_fields"]["last_token_index"]
              for s in tokenised if s["category"].startswith("CAT-")
          ))
    check("No -1 sentinel in any token_fields",
          all(s["token_fields"]["equals_sign_index"] != -1 for s in tokenised))

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

    sign_stims = [s for s in dataset if s["category"] == "CAT-SIGN"]
    par_stims  = [s for s in dataset if s["category"] == "CAT-PARITY"]

    # Sign balance
    sign_dist = Counter(s["labels"]["sign"] for s in sign_stims)
    check("CAT-SIGN: exact 50/50 sign balance",
          sign_dist[0] == sign_dist[1],
          f"sign distribution: {dict(sign_dist)}")

    # Parity balance
    par_dist = Counter(s["labels"]["parity"] for s in par_stims)
    check("CAT-PARITY: exact 50/50 parity balance",
          par_dist[0] == par_dist[1],
          f"parity distribution: {dict(par_dist)}")

    # Pair coherence
    for cat_stims, cat in [(sign_stims, "CAT-SIGN"), (par_stims, "CAT-PARITY")]:
        pair_counts = Counter(s["contrast"]["pair_id"] for s in cat_stims)
        check(f"{cat}: every pair_id appears exactly twice",
              all(v == 2 for v in pair_counts.values()),
              f"odd counts: {[k for k, v in pair_counts.items() if v != 2][:3]}")

    # Template balance (round-robin: difference ≤ 2)
    for cat_stims, cat in [(sign_stims, "CAT-SIGN"), (par_stims, "CAT-PARITY")]:
        tpl_counts = Counter(s["template_id"] for s in cat_stims)
        vals = sorted(tpl_counts.values())
        check(f"{cat}: template counts balanced (max spread ≤ 2)",
              vals[-1] - vals[0] <= 2,
              f"template distribution: {dict(tpl_counts)}")

    # No cross-category ID collisions
    all_ids = [s["id"] for s in dataset]
    check("No duplicate stimulus IDs across categories",
          len(all_ids) == len(set(all_ids)))

    # CAT-PARITY deterministic ordering: -A even, -B odd.
    order_ok = all(
        (s["labels"]["parity"] == 0) if s["id"].endswith("-A")
        else (s["labels"]["parity"] == 1)
        for s in par_stims
    )
    check("CAT-PARITY: -A is even, -B is odd", order_ok)

    # CAT-SIGN: parity of |result| identical within each pair.
    parity_ok = all(
        p[0]["labels"]["parity"] == p[1]["labels"]["parity"]
        for p in _by_pair(sign_stims).values() if len(p) == 2
    )
    check("CAT-SIGN: parity(|result|) identical within each pair", parity_ok)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROUND-TRIP
# ─────────────────────────────────────────────────────────────────────────────

def test_roundtrip(dataset: List[Stimulus], tmp_path: Path) -> None:
    section("4. ROUND-TRIP — JSONL serialisation")

    out_file = tmp_path / "roundtrip.jsonl"
    write_jsonl(out_file, dataset)

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
        missing = REQUIRED_ROOT_KEYS - obj.keys()
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

    # Sentinel: no -1 anywhere in token_fields.
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
            (s["id"], REQUIRED_ROOT_KEYS - s.keys())
            for s in ctrl if REQUIRED_ROOT_KEYS - s.keys()
        ]
        check(f"{label} 5a: all required root keys present",
              not missing, str(missing[:2]))

        # 5b. Sentinel: equals_sign_index is not -1 for CTRL (token_fields empty)
        bad_sentinel = [s["id"] for s in ctrl
                        if s["token_fields"].get("equals_sign_index") == -1]
        check(f"{label} 5b: equals_sign_index never -1",
              not bad_sentinel, str(bad_sentinel[:3]))

        # 5c. Control label sentinels: sign and parity are both -1
        bad_labels = [s["id"] for s in ctrl
                      if s["labels"]["sign"] != -1 or s["labels"]["parity"] != -1]
        check(f"{label} 5c: sign and parity are sentinel (-1)",
              not bad_labels, str(bad_labels[:3]))

        # 5d. Same root schema as arithmetic stimuli
        arith_keys = set(arith_dataset[0].keys())
        ctrl_keys  = set(ctrl[0].keys())
        sym_diff   = arith_keys.symmetric_difference(ctrl_keys)
        check(f"{label} 5d: identical root keys to arithmetic stimuli",
              not sym_diff, f"symmetric difference: {sym_diff}")

        # 5e. JSON round-trip preserves int types
        type_errors = [
            f"{s['id']}.labels.{k}"
            for s in ctrl
            for k in ("sign", "parity")
            if not isinstance(json.loads(json.dumps(s))["labels"][k], int)
        ]
        check(f"{label} 5e: sign/parity are int after JSON round-trip",
              not type_errors, str(type_errors[:3]))

        # 5f. No duplicate texts
        texts = [s["text"] for s in ctrl]
        n_dup = len(texts) - len(set(texts))
        check(f"{label} 5f: zero duplicate texts ({len(texts)} stimuli)",
              n_dup == 0, f"{n_dup} duplicates found")

        # 5g. validate_schema accepts untokenised control under --allow-untokenized
        schema_errors = [
            (s["id"], msg)
            for s in ctrl
            for ok, msg in [validate_schema(s, require_tokenized=False)]
            if not ok
        ]
        check(f"{label} 5g: validate_schema passes (untokenised, allow flag)",
              not schema_errors, str(schema_errors[:2]))

        # 5g+. validate_schema rejects untokenised control when tokenisation required
        rejected = all(
            not validate_schema(s, require_tokenized=True)[0] for s in ctrl
        )
        check(f"{label} 5g+: validate_schema rejects untokenised (strict mode)",
              rejected)

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
                 if s["token_fields"].get("token_length_strata") is not None]
    if tok_arith:
        try:
            matched_neu = length_match_to_arithmetic(tok_neu, tok_arith, seed=0)
            arith_strata = set(
                s["token_fields"]["token_length_strata"] for s in tok_arith
            )
            match_strata = set(
                s["token_fields"]["token_length_strata"] for s in matched_neu
            )
            check("5h: length-matched CTRL-NEU has same strata keys as arithmetic",
                  arith_strata == match_strata,
                  f"arith={arith_strata}, matched={match_strata}")
        except RuntimeError as exc:
            check("5h: length_match_to_arithmetic", False, str(exc))
    else:
        print("  (skipping length-match test: arithmetic stimuli not tokenised)")

    # 5i. validate_schema passes post-tokenisation
    for label, tok_ctrl in [("CTRL-NEU", tok_neu), ("CTRL-NUM", tok_num)]:
        schema_errors = [
            (s["id"], msg)
            for s in tok_ctrl
            for ok, msg in [validate_schema(s, require_tokenized=True)]
            if not ok
        ]
        check(f"{label} 5i: validate_schema passes post-tokenisation",
              not schema_errors, str(schema_errors[:2]))


# ─────────────────────────────────────────────────────────────────────────────
# 6. METADATA — dataset → ExtractionMetadata handoff
# ─────────────────────────────────────────────────────────────────────────────

def test_metadata_completeness() -> None:
    section("6. METADATA — dataset → ExtractionMetadata handoff contract")
    # extract_from_model() builds an ExtractionMetadata from these stimulus
    # fields. Verify the dataset supplies every field that projection reads,
    # without needing a model on the GPU.
    stims: List[Stimulus] = (
        SignContrastGenerator().build(n_pairs=2, seed=0)
        + ParityContrastGenerator().build(n_pairs=2, seed=0)
    )

    # Root-level fields read per stimulus.
    root_ok = all({"id", "category", "dataset_version"} <= s.keys() for s in stims)
    check("6a: every stimulus exposes id / category / dataset_version", root_ok)

    # Label fields read into ExtractionMetadataLabels.
    label_fields = {"sign", "parity", "operand1", "operand2"}
    labels_ok = all(label_fields <= s["labels"].keys() for s in stims)
    check("6b: labels expose sign / parity / operand1 / operand2", labels_ok)

    # Replicate the projection extract_from_model performs and type-check it
    # against the ExtractionMetadata contract.
    meta: ExtractionMetadata = {
        "model_name": "test-model",
        "n_layers": 4,
        "d_model": 8,
        "n_stimuli": len(stims),
        "stimuli_ids": [s["id"] for s in stims],
        "categories": [s["category"] for s in stims],
        "labels": {
            "sign": [s["labels"]["sign"] for s in stims],
            "parity": [s["labels"]["parity"] for s in stims],
            "operand1": [s["labels"]["operand1"] for s in stims],
            "operand2": [s["labels"]["operand2"] for s in stims],
        },
        "probe_strategy": "last_token",
        "dataset_version": stims[0]["dataset_version"],
    }

    check("6c: label arrays parallel to stimuli_ids",
          all(len(meta["labels"][k]) == len(meta["stimuli_ids"])
              for k in ("sign", "parity", "operand1", "operand2")))
    check("6d: n_stimuli == len(stimuli_ids)",
          meta["n_stimuli"] == len(meta["stimuli_ids"]))
    check("6e: metadata JSON-serialisable",
          isinstance(json.dumps(meta), str))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual test suite for the v5 dataset."
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
