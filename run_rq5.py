#!/usr/bin/env python
"""run_rq5.py — RQ5 driver: behavioral determinization at the "=" token.

E-P-02: the "=" next-token distribution is the model's *expected result* before it
is generated. RQ5 tracks how that distribution sharpens across the QLoRA checkpoints
— next-token entropy ↓, top1-top2 logit margin ↑, P(answer) ↑ — as inference-only,
correlative evidence of the fine-tuning trajectory (no causal claim, E-O-01).

Steps mirror RQ4/GSM8K: {0 base, 2500, 5000, 7500, 10000, total_training_steps}.
Step 0 is the un-merged base model; adapters are merged in memory. Deterministic over
the full math set — no RNG. New config keys are read via config.get(key, default).
"""

import argparse
import copy
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import yaml

# Guard against the GPT-NeoX vmap/SDPA bug in newer transformers.
assert transformers.__version__ < "4.49", (
    f"transformers {transformers.__version__} has a vmap/SDPA bug with GPT-NeoX. "
    "Pin to <4.49: pip install 'transformers>=4.46,<4.49'"
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer

from src.config.models import get_model_profile
from src.eval.determinization import (RQ5DeterminizationRow, build_targets,
                                       extract_eq_logits, math_stimuli,
                                       next_token_entropy, prob_of_target,
                                       top1_top2_margin)
from src.eval.eval_gsm8k import calculate_binomial_ci
from src.extraction.extract_states import load_stimuli
from src.probing.io_utils import _atomic_write_csv, _atomic_write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_rq5")

CHECKPOINTS_BASE = Path("data/processed/checkpoints")
STIMULI_PATH = Path("data/processed/dataset_master_v5.jsonl")


# ── MODEL LOADERS (patched out in the e2e test) ───────────────────────────────

def load_base_model(base_model_id: str):
    """Preload the base HF model to CPU once; cloned per step before merging."""
    return AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, device_map="cpu")


def load_tokenizer(base_model_id: str):
    return AutoTokenizer.from_pretrained(base_model_id)


def build_hooked_model(base_hf, base_model_id: str, ckpt_dir: Path | None):
    """Wrap base (step 0) or merged adapter in TransformerLens. Same flags as RQ4."""
    hf = copy.deepcopy(base_hf)
    if ckpt_dir is not None:
        peft_model = PeftModel.from_pretrained(hf, str(ckpt_dir))
        hf = peft_model.merge_and_unload()
    return HookedTransformer.from_pretrained(
        base_model_id, hf_model=hf, device="cuda", dtype=torch.float16, fold_ln=True)


# ── STEP ENUMERATION ──────────────────────────────────────────────────────────

def enumerate_steps(config: dict) -> list[tuple[int, Path | None]]:
    """(step, ckpt_dir) pairs: base (0, None) + sorted checkpoint-* + terminal adapter.

    Reuses the checkpoint_loop idiom; the terminal adapter maps to
    config.get('total_training_steps', 2000) so RQ4's axis ends where GSM8K's does.
    """
    steps: list[tuple[int, Path | None]] = [(0, None)]  # base, pre-FT

    if not CHECKPOINTS_BASE.exists():
        logger.warning(f"No checkpoints under {CHECKPOINTS_BASE}; evaluating base only.")
        return steps

    sequential = sorted(
        [d for d in CHECKPOINTS_BASE.iterdir() if d.is_dir() and "checkpoint" in d.name],
        key=lambda x: int(x.name.split("-")[-1]) if "-" in x.name else 0,
    )
    for d in sequential:
        steps.append((int(d.name.split("-")[-1]), d))

    terminal_step = int(config.get("total_training_steps", 2000))
    for terminal_name in ("final_adapter", "final_checkpoint"):
        terminal_path = CHECKPOINTS_BASE / terminal_name
        if terminal_path.exists():
            steps.append((terminal_step, terminal_path))
            break

    return steps


# ── PER-STEP METRIC AGGREGATION ───────────────────────────────────────────────

def aggregate_step(
    step: int,
    logits: np.ndarray,
    categories: np.ndarray,
    target_ids: np.ndarray,
    single_mask: np.ndarray,
) -> list[RQ5DeterminizationRow]:
    """One row per math category for this step. Continuous metrics in float32."""
    entropy = next_token_entropy(logits)
    margin = top1_top2_margin(logits)
    p_first = prob_of_target(logits, target_ids)  # P(answer's first token)

    rows: list[RQ5DeterminizationRow] = []
    for cat in ("CAT-SIGN", "CAT-PARITY"):
        cat_mask = categories == cat
        n_rows = int(cat_mask.sum())
        if n_rows == 0:
            continue

        single_in_cat = cat_mask & single_mask
        n_single = int(single_in_cat.sum())
        # On single-token rows P(first token) == P(full answer): a true P(correct).
        p_correct = float(p_first[single_in_cat].mean()) if n_single > 0 else 0.0
        ci_lo, ci_hi = calculate_binomial_ci(p_correct, n_single)

        # Entropy/margin over the single-token subset only — for CAT-SIGN this drops the
        # negative half (whose first token is the sign " -") and reads the digit-answer
        # sharpening apart from it.
        entropy_single = float(entropy[single_in_cat].mean()) if n_single > 0 else 0.0
        margin_single = float(margin[single_in_cat].mean()) if n_single > 0 else 0.0

        rows.append({
            "step": step,
            "category": cat,
            "n_rows": n_rows,
            "n_single_token": n_single,
            "entropy_mean": round(float(entropy[cat_mask].mean()), 6),
            "margin_mean": round(float(margin[cat_mask].mean()), 6),
            "entropy_mean_single": round(entropy_single, 6),
            "margin_mean_single": round(margin_single, 6),
            "p_first_token_mean": round(float(p_first[cat_mask].mean()), 6),
            "p_correct_single": round(p_correct, 6),
            "p_correct_single_ci_lo": round(float(ci_lo), 6),
            "p_correct_single_ci_hi": round(float(ci_hi), 6),
        })
    return rows


# ── DRIVER ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RQ5 determinization at '='")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "pythia-1.4b")
    profile = get_model_profile(model_name)
    base_model_id = profile["hf_path"]
    batch_size = int(config.get("rq5_batch_size", profile["extract_batch_size"]))
    out_dir = Path(config.get("rq5_output_dir", "results/rq5_determinization"))

    stimuli = load_stimuli(STIMULI_PATH)
    categories = np.array([s["category"] for s in math_stimuli(stimuli)])

    tokenizer = load_tokenizer(base_model_id)
    target_ids, single_mask = build_targets(tokenizer, stimuli)
    logger.info("Math rows: %d | single-token: %d | multi-token: %d",
                len(categories), int(single_mask.sum()), int((~single_mask).sum()))

    base_hf = load_base_model(base_model_id)
    steps = enumerate_steps(config)
    logger.info("Evaluating %d steps: %s", len(steps), [s for s, _ in steps])

    all_rows: list[RQ5DeterminizationRow] = []
    for step, ckpt_dir in steps:
        logger.info("Step %d (%s)", step, "base" if ckpt_dir is None else ckpt_dir.name)
        model = build_hooked_model(base_hf, base_model_id, ckpt_dir)
        logits = extract_eq_logits(model, stimuli, batch_size)

        step_rows = aggregate_step(step, logits, categories, target_ids, single_mask)
        all_rows.extend(step_rows)
        _atomic_write_json(out_dir / f"determinization_step_{step}.json", {"rows": step_rows})

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    traj = out_dir / "determinization.csv"
    if traj.exists():
        old = pd.read_csv(traj)
        df = pd.concat([old[~old["step"].isin(df["step"].unique())], df], ignore_index=True)

    _atomic_write_csv(traj, df.to_dict("records"), df.columns.tolist())
    logger.info("RQ5 determinization written to %s (%d rows)", traj, len(df))


if __name__ == "__main__":
    main()
