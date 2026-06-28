"""
checkpoint_loop.py — driver for RQ4 dynamic evaluation.
Iterates over saved LoRA checkpoints, merges weights, wraps in TransformerLens,
extracts representations, and triggers the static probing validation.

Processes the terminal adapter, bounds the per-checkpoint subprocess with a
timeout, reuses the base model via deepcopy, and reads paths from the config.
"""

import argparse
import copy
import gc
import logging
import subprocess
import sys
import yaml
from pathlib import Path

import torch
import transformers

# Guard against the GPT-NeoX vmap/SDPA bug in newer transformers.
assert transformers.__version__ < "4.49", (
    f"transformers {transformers.__version__} has a vmap/SDPA bug with GPT-NeoX. "
    "Pin to <4.49: pip install 'transformers>=4.46,<4.49'"
)

from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformer_lens import HookedTransformer

from src.extraction.extract_states import extract_from_model, load_stimuli
from src.config.models import get_model_profile


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("checkpoint_loop")


def process_checkpoint(
    base_hf: AutoModelForCausalLM,
    ckpt_dir: Path,
    base_model_id: str,
    extract_batch_size: int,
    stimuli: list[dict],
    config_path: Path,
    checkpoints_extracted_dir: Path,
    logger: logging.Logger
) -> None:
    """Merge the LoRA adapter, extract states, and trigger RQ4 metric computation."""
    logger.info(f"Processing checkpoint: {ckpt_dir.name}")

    # Clone the preloaded base model to avoid reloading from disk each time.
    logger.info("Cloning in-memory base model...")
    base_hf_copy = copy.deepcopy(base_hf)

    # 2. Merge LoRA weights.
    logger.info("Merging LoRA weights (in memory)...")
    peft_model = PeftModel.from_pretrained(base_hf_copy, str(ckpt_dir))
    merged_hf = peft_model.merge_and_unload()

    # 3. Wrap in HookedTransformer.
    logger.info("Wrapping merged model in HookedTransformer...")
    hooked_model = HookedTransformer.from_pretrained(
        base_model_id,
        hf_model=merged_hf,
        device="cuda",
        dtype=torch.float16,
        fold_ln=True
    )

    # 4. Extract hidden states.
    out_dir = checkpoints_extracted_dir / ckpt_dir.name
    logger.info(f"Extracting activations to {out_dir} (batch size {extract_batch_size})...")

    # The attention mask is computed internally by this call.
    extract_from_model(hooked_model, stimuli, out_dir, batch_size=extract_batch_size)

    # Free VRAM before the next checkpoint.
    del hooked_model
    del merged_hf
    del peft_model
    del base_hf_copy
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Run RQ4 in a subprocess.
    logger.info("Triggering run_rq4.py evaluation...")
    # Use sys.executable to stay inside the active conda/venv.
    cmd = [
        sys.executable,
        "run_rq4.py",
        "--config", str(config_path),
        "--checkpoint_dir", str(out_dir)
    ]

    try:
        # Bound the subprocess to avoid an indefinite hang.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"run_rq4.py failed for {ckpt_dir.name}:\n{result.stderr}")
        else:
            logger.info(f"Evaluation complete for {ckpt_dir.name}.")
    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timed out for {ckpt_dir.name}. Skipping.")


def main() -> None:
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Checkpoint evaluation loop (RQ4)")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config missing: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model_name", "pythia-1.4b")
    profile = get_model_profile(model_name)
    base_model_id = profile["hf_path"]
    extract_batch_size = profile["extract_batch_size"]

    # Read extraction output path from the config.
    checkpoints_extracted_base = Path(cfg.get("checkpoints_extracted_dir", "data/processed/checkpoints_extracted"))

    stimuli_path = Path("data/processed/dataset_master_v5.jsonl")
    if not stimuli_path.exists():
        raise FileNotFoundError(f"Dataset missing: {stimuli_path}")
    stimuli = load_stimuli(stimuli_path)

    checkpoints_base = Path("data/processed/checkpoints")

    # Standard sequential checkpoints, ordered by step.
    checkpoints = sorted(
        [d for d in checkpoints_base.iterdir() if d.is_dir() and "checkpoint" in d.name],
        key=lambda x: int(x.name.split("-")[-1]) if "-" in x.name else 0
    )

    # Detect the terminal LoRA adapter under either canonical name.
    # Older runs used "final_checkpoint/"; current train_qlora.py emits "final_adapter/".
    for terminal_name in ("final_adapter", "final_checkpoint"):
        terminal_path = checkpoints_base / terminal_name
        if terminal_path.exists():
            logger.info(f"Located terminal adapter at '{terminal_name}'. Appending to targets.")
            checkpoints.append(terminal_path)
            break  # only the first match

    if not checkpoints:
        logger.warning(f"No adapters found under: {checkpoints_base}")
        return

    logger.info(f"Found {len(checkpoints)} target checkpoints.")

    # Preload the base model once to avoid per-checkpoint disk reloads.
    logger.info(f"Preloading base model ({base_model_id}) to RAM...")
    base_hf = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    for ckpt in checkpoints:
        process_checkpoint(
            base_hf=base_hf,
            ckpt_dir=ckpt,
            base_model_id=base_model_id,
            extract_batch_size=extract_batch_size,
            stimuli=stimuli,
            config_path=config_path,
            checkpoints_extracted_dir=checkpoints_extracted_base,
            logger=logger
        )

    logger.info("Dynamic checkpoint extraction sequence completed.")


if __name__ == "__main__":
    main()
