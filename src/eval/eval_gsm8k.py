"""
eval_gsm8k.py — Dynamic Evaluation Layer (RQ3).
Executes 0-shot evaluation on GSM8K using EleutherAI's lm-evaluation-harness.
Strictly decoupled from model merging: requires base HF models or already merged weights.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import lm_eval

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("eval_gsm8k")

def check_unmerged_adapter(model_path: str) -> None:
    """
    Prevents execution on raw QLoRA adapters.
    Validates that if the path is local, it contains a full model, not just adapter_config.json.
    """
    path = Path(model_path)
    if path.is_dir():
        has_adapter = (path / "adapter_config.json").exists()
        has_config = (path / "config.json").exists()
        
        if has_adapter and not has_config:
            raise ValueError(
                f"FATAL: '--model_path' points to an unmerged LoRA adapter ({model_path}).\n"
                "Merging is the responsibility of checkpoint_loop.py. Please provide a fused model path."
            )

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Writes a DataFrame to CSV atomically to prevent I/O corruption."""
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)

def append_to_trajectory(step: int, gsm8k_acc: float, csv_path: Path) -> None:
    """
    Updates the dynamic trajectory CSV mapping checkpoints to their GSM8K accuracy.
    If the file exists, it matches on the 'step' column. If not, it creates it.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # Ensure step column exists
        if "step" not in df.columns:
            raise KeyError(f"Corrupted trajectory CSV: missing 'step' column in {csv_path}")
            
        # Ensure gsm8k_acc column exists
        if "gsm8k_acc" not in df.columns:
            df["gsm8k_acc"] = pd.NA
            
        if step in df["step"].values:
            df.loc[df["step"] == step, "gsm8k_acc"] = gsm8k_acc
        else:
            # Create a new row filled with NaNs for the geometric columns, then update
            new_row = {col: pd.NA for col in df.columns}
            new_row["step"] = step
            new_row["gsm8k_acc"] = gsm8k_acc
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Create from scratch if RQ3 hasn't logged geometric drift yet
        df = pd.DataFrame([{"step": step, "gsm8k_acc": gsm8k_acc}])
        
    _atomic_write_csv(df, csv_path)

def parse_step_from_tag(tag: str) -> int:
    """Extracts integer step from tags like 'baseline' or 'step500'."""
    if tag.lower() == "baseline":
        return 0
    if tag.lower().startswith("step"):
        return int(tag[4:])
    try:
        return int(tag)
    except ValueError:
        raise ValueError(f"Cannot parse integer step from tag '{tag}'. Expected 'baseline' or 'step<N>'.")

def evaluate_model(model_path: str, tag: str, logger: logging.Logger) -> None:
    """Runs lm-evaluation-harness on GSM8K and stores results."""
    check_unmerged_adapter(model_path)
    
    logger.info(f"Initiating GSM8K evaluation for model: {model_path} (Tag: {tag})")
    
    # 0-shot GSM8K evaluation using LM Eval Harness
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=["gsm8k"],
        num_fewshot=0,
        batch_size="auto"
    )
    
    # Safely extract accuracy. GSM8K typically outputs 'exact_match,strict', but we fallback to 'acc,none'
    task_results = results["results"]["gsm8k"]
    if "acc,none" in task_results:
        accuracy = task_results["acc,none"]
    elif "exact_match,strict" in task_results:
        accuracy = task_results["exact_match,strict"]
    else:
        # Fallback to the first float value found if API changes
        accuracy = next(v for v in task_results.values() if isinstance(v, float))
    
    # Get total samples (if available in n-shot config or inferred)
    # n_samples might be stored in 'n-shot' context or can be hardcoded to 1319 (GSM8K test set size)
    n_samples = 1319 
    
    payload = {
        "accuracy": accuracy,
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
        "tag": tag
    }
    
    # Save JSON Report
    json_out_dir = Path("results/gsm8k")
    json_out_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_out_dir / f"gsm8k_{tag}.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        
    logger.info(f"JSON result saved to {json_path}")
    logger.info(f"Accuracy [{tag}]: {accuracy:.4f}")
    
    # Append to Trajectory CSV
    step = parse_step_from_tag(tag)
    csv_path = Path("results/rq2_probing/dynamic/trajectories.csv")
    
    append_to_trajectory(step, accuracy, csv_path)
    logger.info(f"Trajectory updated in {csv_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K and log to trajectory.")
    parser.add_argument("--model_path", type=str, required=True, help="HF Hub ID or local merged path.")
    parser.add_argument("--tag", type=str, required=True, help="Evaluation tag (e.g., 'baseline', 'step500').")
    
    args = parser.parse_args()
    logger = setup_logger()
    
    try:
        evaluate_model(args.model_path, args.tag, logger)
    except Exception as e:
        logger.error(f"GSM8K Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()