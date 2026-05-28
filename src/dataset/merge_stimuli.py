"""
merge_stimuli.py  —  v5
=======================
Central validator and merger for the v5 dataset (sign + parity focus).

Merges multiple JSONL files (arithmetic + control) into a single master
dataset, enforcing schema consistency required by the analysis pipeline.

Schema changes from v4 → v5
-----------------------------
  Removed:  operator (label), magnitude_log10, finetuning_train split,
            extraction_strategy_by_property, operand_digit_class
  Added:    probe_layer_strategy; labels.operand1 / labels.operand2
  Changed:  equals_sign_index sentinel is now None (never -1)
  Changed:  all stimuli carry split = "geometric_eval"
  Changed:  tokenizer_name is a required sub-field of token_fields

USAGE
-----
    python merge_stimuli.py \\
        --inputs data/raw/stimuli_arithmetic_v5.jsonl \\
                 data/raw/stimuli_ctrl_neu_v5.jsonl \\
                 data/raw/stimuli_ctrl_num_v5.jsonl \\
        --output data/processed/dataset_master_v5.jsonl

    Default mode blocks on untokenised stimuli.
    Use --allow-untokenized to downgrade to warning (development only).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Tuple

from src.config.categories import ALL_CATS

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_ROOT_KEYS = {
    "id", "text", "split", "template_id", "macro_format", "category",
    "n_reasoning_steps", "labels",
    "contrast", "token_fields", "ood_target", "dataset_version",
    "probe_layer_strategy",
}

REQUIRED_LABELS_KEYS   = {"result", "sign", "parity"}
REQUIRED_TOKEN_KEYS    = {"n_tokens", "token_ids", "token_strs",
                          "token_length_strata", "equals_sign_index",
                          "last_token_index", "tokenizer_name"}
REQUIRED_CONTRAST_KEYS = {"pair_id", "varying_axis", "controlled_axes"}

VALID_CATEGORIES: frozenset[str] = frozenset(ALL_CATS)
VALID_SPLITS     = {"geometric_eval"}


def validate_schema(
    stimulus: dict,
    require_tokenized: bool = True,
) -> Tuple[bool, str]:
    """
    Verify that a deserialised stimulus dict satisfies the v5 interface contract.

    Args:
        stimulus:          Dictionary from a single JSONL line.
        require_tokenized: If True (default), return an error when
                           token_fields.n_tokens is None.

    Returns:
        (True, "")          — stimulus is valid.
        (False, error_msg)  — stimulus is invalid; error_msg describes the issue.
    """

    # ── Root keys ──────────────────────────────────────────────────────────
    missing_root = REQUIRED_ROOT_KEYS - stimulus.keys()
    if missing_root:
        return False, f"Missing root keys: {missing_root}"

    # ── split ──────────────────────────────────────────────────────────────
    if stimulus["split"] not in VALID_SPLITS:
        return False, (
            f"Invalid split: {stimulus['split']!r}. "
            f"Allowed values: {VALID_SPLITS}"
        )

    # ── category ───────────────────────────────────────────────────────────
    if stimulus["category"] not in VALID_CATEGORIES:
        return False, (
            f"Invalid category: {stimulus['category']!r}. "
            f"Allowed values: {VALID_CATEGORIES}"
        )

    # ── labels ─────────────────────────────────────────────────────────────
    labels = stimulus["labels"]
    if not isinstance(labels, dict):
        return False, "'labels' must be a dict."
    missing_lab = REQUIRED_LABELS_KEYS - labels.keys()
    if missing_lab:
        return False, f"Missing labels keys: {missing_lab}"
    for key in ("sign", "parity"):
        if not isinstance(labels[key], int):
            return False, (
                f"labels.{key} must be an int, "
                f"got {type(labels[key]).__name__}."
            )

    # ── contrast ───────────────────────────────────────────────────────────
    contrast = stimulus["contrast"]
    if not isinstance(contrast, dict):
        return False, "'contrast' must be a dict."
    missing_con = REQUIRED_CONTRAST_KEYS - contrast.keys()
    if missing_con:
        return False, f"Missing contrast keys: {missing_con}"

    # ── token_fields ───────────────────────────────────────────────────────
    tf = stimulus["token_fields"]
    if not isinstance(tf, dict):
        return False, "'token_fields' must be a dict."

    # Freshly built stimuli carry an empty token_fields ({}); it is populated
    # only by populate_token_fields(). Enforce the full key contract whenever
    # tokenisation is required or any token field is already present — but let
    # an untokenised stimulus through when --allow-untokenized is active.
    if require_tokenized or tf:
        missing_tf = REQUIRED_TOKEN_KEYS - tf.keys()
        if missing_tf:
            return False, f"Missing token_fields keys: {missing_tf}"

    if require_tokenized and tf.get("n_tokens") is None:
        return False, (
            "token_fields.n_tokens is null — stimulus not yet tokenised.  "
            "Run populate_token_fields() before merging, or pass "
            "--allow-untokenized to downgrade this to a warning."
        )

    # ── sentinel check: -1 must never appear as equals_sign_index ──────────
    if tf.get("equals_sign_index") == -1:
        return False, (
            "token_fields.equals_sign_index = -1 detected.  "
            "The v5 convention uses None as the unpopulated sentinel.  "
            "This stimulus was likely generated by a v4 builder."
        )

    # ── tokenizer_name must be set when tokenised ──────────────────────────
    if tf.get("n_tokens") is not None and tf.get("tokenizer_name") is None:
        return False, (
            "token_fields.n_tokens is set but tokenizer_name is None.  "
            "Always populate tokenizer_name when tokenising (fixes I-10)."
        )

    # ── arithmetic-only checks ─────────────────────────────────────────────
    cat = stimulus["category"]
    if cat in ("CAT-SIGN", "CAT-PARITY"):
        # equals_sign_index == last_token_index (when tokenised)
        eq  = tf.get("equals_sign_index")
        lst = tf.get("last_token_index")
        if eq is not None and lst is not None and eq != lst:
            return False, (
                f"For {cat}, equals_sign_index ({eq}) must equal "
                f"last_token_index ({lst}).  The '=' token must be last."
            )

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and validate v5 JSONL datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs",  nargs="+", required=True,
                        help="One or more tokenised JSONL files to merge.")
    parser.add_argument("--output",  required=True,
                        help="Output JSONL path (master dataset).")
    parser.add_argument("--allow-untokenized", action="store_true",
                        help="Downgrade missing token_fields from error to warning. "
                             "Do not use for final experiments.")
    args = parser.parse_args()

    require_tokenized    = not args.allow_untokenized
    out_path             = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    master_dataset       = []
    seen_ids             = set()
    category_counts      = Counter() # type: ignore
    split_counts         = Counter() # type: ignore
    strata_counts        = Counter() # type: ignore
    untokenized_warnings = 0

    print(f"\nMerging {len(args.inputs)} file(s) …")
    if not require_tokenized:
        print("  [WARNING] --allow-untokenized active: "
              "missing token_fields will not block merge.")

    for in_file in args.inputs:
        fp = Path(in_file)
        if not fp.exists():
            print(f"[ERROR] File not found: {fp}")
            sys.exit(1)

        print(f"  Processing {fp.name} …")
        with fp.open("r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    stimulus = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[ERROR] Malformed JSON in {fp.name} line {line_idx}.")
                    sys.exit(1)

                ok, err = validate_schema(stimulus, require_tokenized)
                if not ok:
                    print(f"[ERROR] Schema violation in {fp.name} line {line_idx}: {err}")
                    sys.exit(1)

                if not require_tokenized and stimulus["token_fields"].get("n_tokens") is None:
                    untokenized_warnings += 1

                sid = stimulus["id"]
                if sid in seen_ids:
                    print(f"[ERROR] Duplicate ID: '{sid}' in {fp.name} line {line_idx}.")
                    sys.exit(1)

                seen_ids.add(sid)
                master_dataset.append(stimulus)
                category_counts[stimulus["category"]] += 1
                split_counts[stimulus.get("split", "unknown")] += 1
                strata_counts[
                    (stimulus.get("token_fields") or {}).get(
                        "token_length_strata", "unpopulated"
                    )
                ] += 1

    if untokenized_warnings:
        print(f"\n  [WARNING] {untokenized_warnings} untokenised stimuli present.")

    # ── Write master JSONL ─────────────────────────────────────────────────
    print(f"\nWriting master dataset to {out_path} …")
    with out_path.open("w", encoding="utf-8") as fh:
        for s in master_dataset:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ── Write metadata sidecar ─────────────────────────────────────────────
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "dataset_version":       "v5",
        "total_stimuli":         len(master_dataset),
        "source_files":          [Path(p).name for p in args.inputs],
        "allow_untokenized":     not require_tokenized,
        "schema_verified":       True,
        "category_distribution": dict(category_counts),
        "split_distribution":    dict(split_counts),
        "token_length_strata":   dict(strata_counts),
        "probe_properties":      ["sign", "parity"],
        "probe_layer_strategy":  "all_layers",
        "target_model":          "EleutherAI/pythia-1.4b",
        "notes": {
            "N-01": (
                "CAT-SIGN: first operand differs between pair members.  "
                "Unavoidable for sign contrast on causal LM.  "
                "Document as a known confound in methodology."
            ),
            "N-02": (
                "CAT-PARITY: second operand b vs b+1.  "
                "Minimum surface difference for intra-operator parity contrast.  "
                "Document as a known confound in methodology."
            ),
        },
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n✓  MERGE COMPLETE")
    print(f"   Total stimuli validated : {len(master_dataset)}")
    for cat, n in sorted(category_counts.items()):
        print(f"   {cat:<15}: {n}")
    print(f"\n   Metadata written to {meta_path}")


if __name__ == "__main__":
    main()