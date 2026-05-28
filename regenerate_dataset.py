#!/usr/bin/env python
"""
regenerate_dataset.py — one-shot pipeline runner to bring the refactored v5 schema live.

The deployed dataset/tensors/probes predate the dataset refactor: their labels lack
operand1/operand2 and CAT-PARITY carries sign=0 instead of the -1 sentinel. The
operand-based confound checks (run_confound_checks.py, run_parity_confound_checks.py)
therefore cannot run against the deployed artifacts. This runner rebuilds the dataset
with the current builders and, optionally, re-extracts states and re-runs RQ2 + the
confound audits so everything is mutually consistent.

Stages
------
  always       build CAT-SIGN/CAT-PARITY → tokenise → build CTRL-NEU/CTRL-NUM →
               tokenise → merge (schema validation + balance diagnostic)
  --with-extraction   load Pythia via HookedTransformer, re-extract layer_XX.pt   [GPU]
  --with-rq2          run run_rq2.py on the new tensors                            [GPU-ish]
  --with-confounds    run sign (N-01) and parity (N-02) confound checks

Safety
------
By default the master is written to a *_regenerated.jsonl path so the deployed v5
dataset and its (aligned) tensors are left intact. Pass --commit to write the
canonical path. The heavy stages re-read the canonical master and overwrite the
extracted tensors / RQ2 results, so they REQUIRE --commit (a fresh dataset and stale
tensors must never coexist at the canonical path).

Examples
--------
    # Non-destructive: regenerate + inspect the new dataset only (CPU).
    python regenerate_dataset.py

    # Commit the new dataset and rebuild everything downstream (GPU).
    python regenerate_dataset.py --commit --with-extraction --with-rq2 --with-confounds \\
        --config configs/config_rq2.yaml
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from src.dataset.build_stimuli import (
    SignContrastGenerator,
    ParityContrastGenerator,
    populate_token_fields,
    validate_dataset,
    write_jsonl,
)
from src.dataset.build_control import (
    generate_neutral_stimuli,
    generate_numeric_stimuli,
)
from src.probing.seeds import get_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regenerate")

CANONICAL_MASTER = Path("data/processed/dataset_master_v5.jsonl")


def build_dataset(args: argparse.Namespace) -> Path:
    """Build arithmetic + control stimuli, tokenise, write raw shards, and merge."""
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Arithmetic (CAT-SIGN, CAT-PARITY) ──
    sign_seed = get_seed(args.seed, "build_stimuli_sign", 0)
    par_seed = get_seed(args.seed, "build_stimuli_parity", 0)
    logger.info(f"Building {args.n_pairs} pairs/category (seed={args.seed}) ...")
    sign_stims = SignContrastGenerator().build(args.n_pairs, seed=sign_seed)
    par_stims = ParityContrastGenerator().build(args.n_pairs, seed=par_seed)
    arith = sign_stims + par_stims

    # ── Load tokenizer object once (length-filters controls; names token_fields) ──
    tokenizer_obj = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer {args.tokenizer} ...")
        tokenizer_obj = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        populate_token_fields(arith, args.tokenizer)

    # ── Controls (CTRL-NEU, CTRL-NUM), length-filtered when a tokenizer is present ──
    logger.info(f"Building {args.n_control} control stimuli/category ...")
    neu = generate_neutral_stimuli(args.n_control, seed=args.seed, tokenizer=tokenizer_obj)
    num = generate_numeric_stimuli(args.n_control, seed=args.seed, tokenizer=tokenizer_obj)
    if args.tokenizer:
        populate_token_fields(neu, args.tokenizer)
        populate_token_fields(num, args.tokenizer)

    # Internal structural invariants on the arithmetic half.
    validate_dataset(arith)

    # ── Write raw shards (names mirror the deployed source_files) ──
    arith_path = raw_dir / "stimuli_arithmetic_v5.jsonl"
    neu_path = raw_dir / "stimuli_ctrl_neu_v5.jsonl"
    num_path = raw_dir / "stimuli_ctrl_num_v5.jsonl"
    write_jsonl(arith_path, arith)
    write_jsonl(neu_path, neu)
    write_jsonl(num_path, num)
    logger.info(f"Raw shards written to {raw_dir}/")

    # ── Merge (schema validation + category-balance diagnostic) ──
    master_path = CANONICAL_MASTER if args.commit else Path(args.out)
    merge_cmd = [
        sys.executable, "-m", "src.dataset.merge_stimuli",
        "--inputs", str(arith_path), str(neu_path), str(num_path),
        "--output", str(master_path),
    ]
    if not args.tokenizer:
        merge_cmd.append("--allow-untokenized")
    logger.info("Merging shards → %s", master_path)
    result = subprocess.run(merge_cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError("merge_stimuli failed; aborting regeneration.")
    return master_path


def run_extraction(args: argparse.Namespace) -> None:
    """Re-extract base hidden states for the committed dataset. Requires GPU."""
    import torch
    from transformer_lens import HookedTransformer
    from src.extraction.extract_states import extract_from_model, load_stimuli

    model_id = args.hf_path
    out_dir = Path("data/processed") / args.model_name
    logger.info(f"Loading {model_id} into HookedTransformer (cuda, fp16) ...")
    model = HookedTransformer.from_pretrained(
        model_id, device="cuda", dtype=torch.float16, fold_ln=True
    )
    stimuli = load_stimuli(CANONICAL_MASTER)
    logger.info(f"Extracting {len(stimuli)} stimuli → {out_dir} ...")
    extract_from_model(model, stimuli, out_dir, batch_size=args.batch_size)


def run_subprocess_stage(label: str, cmd: list[str]) -> None:
    logger.info(f"[{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the v5 dataset and (optionally) the downstream pipeline.")
    parser.add_argument("--n_pairs", type=int, default=500, help="Pairs per arithmetic category (default 500 → 1000 stimuli each).")
    parser.add_argument("--n_control", type=int, default=500, help="Stimuli per control category (default 500).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-1.4b",
                        help="HF tokenizer for token_fields; empty string skips tokenisation.")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/processed/dataset_master_v5_regenerated.jsonl",
                        help="Master output path when --commit is NOT set (non-destructive default).")
    parser.add_argument("--commit", action="store_true",
                        help="Write the canonical master path. Required for any heavy downstream stage.")
    parser.add_argument("--with-extraction", action="store_true", help="Re-extract hidden states (GPU).")
    parser.add_argument("--with-rq2", action="store_true", help="Run run_rq2.py after extraction.")
    parser.add_argument("--with-confounds", action="store_true", help="Run sign + parity confound checks.")
    parser.add_argument("--config", type=str, default="configs/config_rq2.yaml", help="Config for rq2/confound stages.")
    parser.add_argument("--model_name", type=str, default="pythia-1.4b", help="Tensor dir name under data/processed.")
    parser.add_argument("--hf_path", type=str, default="EleutherAI/pythia-1.4b", help="HF model id for extraction.")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    heavy = args.with_extraction or args.with_rq2 or args.with_confounds
    if heavy and not args.commit:
        parser.error(
            "Heavy stages (--with-extraction/--with-rq2/--with-confounds) overwrite the "
            "canonical tensors/results and require --commit so the dataset and tensors stay aligned."
        )
    if (args.with_rq2 or args.with_confounds) and not args.with_extraction:
        logger.warning(
            "Running rq2/confounds WITHOUT --with-extraction: existing tensors must already "
            "align with the regenerated dataset, otherwise results will be silently wrong."
        )

    master_path = build_dataset(args)
    logger.info(f"✓ Dataset regenerated → {master_path}")
    if not args.commit:
        logger.info("Non-destructive run complete. Deployed v5 + tensors untouched. "
                    "Re-run with --commit (+ heavy stages) to bring the new schema live.")
        return

    if args.with_extraction:
        run_extraction(args)
    if args.with_rq2:
        run_subprocess_stage("RQ2", [sys.executable, "run_rq2.py", "--config", args.config])
    if args.with_confounds:
        run_subprocess_stage("confound-sign",
                             [sys.executable, "-m", "src.probing.run_confound_checks", "--config", args.config])
        run_subprocess_stage("confound-parity",
                             [sys.executable, "-m", "src.probing.run_parity_confound_checks", "--config", args.config])

    logger.info("✓ Regeneration pipeline complete.")


if __name__ == "__main__":
    main()
