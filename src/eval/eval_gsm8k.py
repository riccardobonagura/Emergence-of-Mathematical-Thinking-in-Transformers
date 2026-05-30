#!/usr/bin/env python
"""
eval_gsm8k.py — Dynamic Evaluation Layer (RQ3).
Executes strictly controlled 0-shot evaluation on GSM8K.
Enforces Binomial Confidence Intervals and NaN-safe trajectory updates.
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
import lm_eval
from lm_eval.utils import make_table

from src.probing.io_utils import (_atomic_write_csv, _atomic_write_json,
                                   setup_logging)
from src.probing.seeds import get_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_gsm8k")


def calculate_binomial_ci(accuracy: float, n_samples: int, z_score: float = 1.96) -> Tuple[float, float]:
    """
    Calculates the Wald confidence interval for a binomial proportion.
    Essential to prove that accuracy=0.0 is statistically bounded.
    """
    if n_samples == 0:
        return 0.0, 0.0
    margin = z_score * math.sqrt((accuracy * (1.0 - accuracy)) / n_samples)
    return max(0.0, round(accuracy - margin, 4)), min(1.0, round(accuracy + margin, 4))


def parse_step_from_tag(tag: str, config: dict) -> int:
    """Extracts step number from tags like 'ckpt_500', 'baseline', or 'final_adapter'.

    - 'baseline' / 'base' → 0
    - 'final' / 'final_adapter' / 'final_checkpoint' → config['total_training_steps']
    - 'ckpt_NNNN' or any tag with digits → that integer
    """
    t = tag.lower()
    if t in ("baseline", "base"):
        return 0
    if t in ("final", "final_adapter", "final_checkpoint") or t.startswith("final"):
        step = config.get("total_training_steps")
        if step is None:
            raise ValueError(
                f"Tag '{tag}' designates the terminal adapter, but config has no 'total_training_steps'."
            )
        return int(step)
    digits = ''.join(filter(str.isdigit, tag))
    if not digits:
        raise ValueError(f"Cannot parse training step from tag '{tag}'")
    return int(digits)


def append_to_trajectory(step: int, acc: float, ci_lower: float, ci_upper: float, csv_path: Path) -> None:
    """
    Safely merges the external GSM8K benchmark with the internal geometric drift.
    Prevents NaN injection on unaligned rows.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        # If trajectories doesn't exist yet, we can't append external metrics safely.
        # Wait for run_rq3.py to initialize it with geometric data.
        logging.warning(f"Trajectory file {csv_path} not found. GSM8K metrics will be saved in JSON only.")
        return

    df = pd.read_csv(csv_path)

    # Ensure columns exist to prevent KeyError
    for col in ["gsm8k_acc", "gsm8k_ci_lower", "gsm8k_ci_upper"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Update only the rows matching the current step
    mask = df["step"] == step
    if not mask.any():
        logging.warning(f"Step {step} not found in {csv_path}. Run run_rq3.py for this step first.")
        return

    df.loc[mask, "gsm8k_acc"] = acc
    df.loc[mask, "gsm8k_ci_lower"] = ci_lower
    df.loc[mask, "gsm8k_ci_upper"] = ci_upper

    _atomic_write_csv(csv_path, df.to_dict("records"), df.columns.tolist())


def check_adapter_consistency(model_path: str, strategy: str) -> None:
    path = Path(model_path)
    if path.is_dir():
        has_adapter = (path / "adapter_config.json").exists()
        has_config = (path / "config.json").exists()

        if strategy == "peft" and not has_adapter:
            raise ValueError(
                f"FATAL: '--loading_strategy peft' expects an unmerged LoRA adapter, "
                f"but {model_path} lacks adapter_config.json."
            )
        elif strategy in ["merged_cpu", "merged_direct"] and has_adapter and not has_config:
            raise ValueError(
                f"FATAL: '--loading_strategy {strategy}' expects a merged model, "
                f"but {model_path} points to an unmerged adapter."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict GSM8K Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="HF Hub ID, local merged path, or adapter path.")
    parser.add_argument("--tag", type=str, required=True, help="Evaluation tag (e.g., 'baseline', 'ckpt_500').")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml to extract global seed architecture.")
    parser.add_argument("--loading_strategy", type=str, default="peft", choices=["peft", "merged_cpu", "merged_direct"])
    args = parser.parse_args()

    # Enforce strict configuration seed fetching over manual default fallbacks
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Derive a dedicated, isolated evaluation seed hash for lm_eval
    eval_seed = get_seed(config["seed"], "gsm8k_evaluation", 0)

    json_out_dir = Path("results/gsm8k")
    logger = setup_logging(json_out_dir)

    try:
        check_adapter_consistency(args.model_path, args.loading_strategy)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Starting rigorous 0-shot evaluation on GSM8K for {args.tag}")

    # Base model string formulation for lm_eval
    if args.loading_strategy == "peft":
        model_args = f"pretrained=EleutherAI/pythia-1.4b,peft={args.model_path}"
    else:
        model_args = f"pretrained={args.model_path}"

    # Simple_evaluate executing 0-shot regime with explicit seed encapsulation
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["gsm8k"],
        num_fewshot=0,             # Documented 0-shot boundary
        batch_size="auto",
        device="cuda",
        limit=None,                # Full evaluation
        random_seed=eval_seed,
        numpy_random_seed=eval_seed,
        torch_random_seed=eval_seed,
    )

    if results is None or "results" not in results:
        logger.error("lm_eval returned empty results. Check CUDA OOM or network issues.")
        sys.exit(1)

    # Parse results
    gsm8k_res = results["results"].get("gsm8k", {})
    accuracy = float(gsm8k_res.get("exact_match,strict-match", 0.0))

    # Binomial Confidence Interval (assuming standard 1319 test samples for GSM8K)
    n_samples = 1319
    ci_lo, ci_hi = calculate_binomial_ci(accuracy, n_samples)

    payload = {
        "tag": args.tag,
        "model_path": args.model_path,
        "strategy": args.loading_strategy,
        "regime": "0-shot",
        "seed": eval_seed,
        "accuracy": accuracy,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat()
    }

    json_path = json_out_dir / f"gsm8k_{args.tag}.json"
    _atomic_write_json(json_path, payload)

    logger.info(f"\n{make_table(results)}")
    logger.info(f"Accuracy [{args.tag}]: {accuracy:.4f} (95% CI: {ci_lo:.4f} - {ci_hi:.4f})")

    # Update trajectory orchestrator
    step = parse_step_from_tag(args.tag, config)
    csv_path = Path("results/rq2_probing/dynamic/trajectories_probing.csv")
    append_to_trajectory(step, accuracy, ci_lo, ci_hi, csv_path)
    logger.info(f"Trajectory alignment completed for step {step}")


if __name__ == "__main__":
    main()
