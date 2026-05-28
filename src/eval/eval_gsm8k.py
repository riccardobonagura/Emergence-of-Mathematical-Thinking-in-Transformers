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

def check_adapter_consistency(model_path: str, strategy: str) -> None:
    path = Path(model_path)
    if path.is_dir():
        has_adapter = (path / "adapter_config.json").exists()
        has_config = (path / "config.json").exists()
        
        if strategy == "peft":
            if not has_adapter:
                raise ValueError(
                    f"FATAL: '--loading_strategy peft' expects an unmerged LoRA adapter, "
                    f"but {model_path} lacks adapter_config.json."
                )
        elif strategy in ["merged_cpu", "merged_direct"]:
            if has_adapter and not has_config:
                raise ValueError(
                    f"FATAL: '--loading_strategy {strategy}' expects a merged model, "
                    f"but {model_path} points to an unmerged adapter."
                )

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)

def append_to_trajectory(step: int, gsm8k_acc: float, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        if "step" not in df.columns:
            raise KeyError(f"Corrupted trajectory CSV: missing 'step' column in {csv_path}")
            
        if "gsm8k_acc" not in df.columns:
            df["gsm8k_acc"] = pd.NA
            
        if step in df["step"].values:
            df.loc[df["step"] == step, "gsm8k_acc"] = gsm8k_acc
        else:
            new_row = {col: pd.NA for col in df.columns}
            new_row["step"] = step
            new_row["gsm8k_acc"] = gsm8k_acc
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([{"step": step, "gsm8k_acc": gsm8k_acc}])
        
    _atomic_write_csv(df, csv_path)

def parse_step_from_tag(tag: str) -> int:
    if tag.lower() in ("baseline", "step0"):
        return 0
    if tag.lower() == "final":
        return 12343  # last training step
    if tag.lower().startswith("step"):
        return int(tag[4:])
    try:
        return int(tag)
    except ValueError:
        raise ValueError(f"Cannot parse integer step from tag '{tag}'.")
        
def evaluate_model(model_path: str, tag: str, strategy: str, logger: logging.Logger) -> None:
    check_adapter_consistency(model_path, strategy)
    
    logger.info(f"Initiating GSM8K evaluation for model: {model_path} (Tag: {tag}, Strategy: {strategy})")
    
    if strategy == "peft":
        model_args = f"pretrained=EleutherAI/pythia-1.4b,peft={model_path},dtype=float16"
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=["gsm8k"],
            num_fewshot=0,
            batch_size=4,
        )
    elif strategy == "merged_direct":
        model_args = f"pretrained={model_path},dtype=float16"
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=["gsm8k"],
            num_fewshot=0,
            batch_size=4,
        )
    elif strategy == "merged_cpu":
        # Lazy HF imports for bypass
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lm_eval.models.huggingface import HFLM
        
        logger.info("Bypassing accelerate meta device: loading directly to CPU, then moving to CUDA...")
        model_obj = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        tokenizer_obj = AutoTokenizer.from_pretrained(model_path)
        model_obj = model_obj.cuda()
        
        hf_model = HFLM(pretrained=model_obj, tokenizer=tokenizer_obj, batch_size=4)
        results = lm_eval.simple_evaluate(
            model=hf_model,
            tasks=["gsm8k"],
            num_fewshot=0,
            batch_size=4,
        )
    else:
        raise ValueError(f"Unknown loading strategy: {strategy}")
    
    task_results = results["results"]["gsm8k"]
    if "acc,none" in task_results:
        accuracy = task_results["acc,none"]
    elif "exact_match,strict" in task_results:
        accuracy = task_results["exact_match,strict"]
    else:
        accuracy = next(v for v in task_results.values() if isinstance(v, float))
    
    n_samples = 1319 
    
    payload = {
        "accuracy": accuracy,
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
        "tag": tag,
        "loading_strategy": strategy
    }
    
    json_out_dir = Path("results/gsm8k")
    json_out_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_out_dir / f"gsm8k_{tag}.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        
    logger.info(f"JSON result saved to {json_path}")
    logger.info(f"Accuracy [{tag}]: {accuracy:.4f}")
    
    step = parse_step_from_tag(tag)
    csv_path = Path("results/rq2_probing/dynamic/trajectories.csv")
    
    append_to_trajectory(step, accuracy, csv_path)
    logger.info(f"Trajectory updated in {csv_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K and log to trajectory.")
    parser.add_argument("--model_path", type=str, required=True, help="HF Hub ID, local merged path, or adapter path.")
    parser.add_argument("--tag", type=str, required=True, help="Evaluation tag (e.g., 'baseline', 'step500').")
    parser.add_argument(
        "--loading_strategy", 
        type=str, 
        default="peft", 
        choices=["peft", "merged_cpu", "merged_direct"],
        help="Strategy to load the model (fixes WSL2 mmap deadlocks)."
    )
    
    args = parser.parse_args()
    logger = setup_logger()
    
    try:
        evaluate_model(args.model_path, args.tag, args.loading_strategy, logger)
    except Exception as e:
        logger.error(f"GSM8K Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()