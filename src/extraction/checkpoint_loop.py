"""
checkpoint_loop.py — Hardware orchestration for RQ3 dynamic evaluation.
Iterates over saved LoRA checkpoints, merges weights, wraps in TransformerLens,
extracts representations, and triggers the static probing validation.

Enforces fixes CL-01 to CL-07: process final_adapter, handle execution timeouts,
reuse base model allocations via deepcopy, and eliminate hardcoded environments.
"""

import argparse
import copy
import gc
import logging
import os
import subprocess
import sys
import yaml
from pathlib import Path

import torch
import transformers

# ENV-02: Protect against GPT-NeoX vmap/SDPA bug in newer transformers
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
    """Merges LoRA adapter, extracts states, and triggers RQ3 metric computation."""
    logger.info(f"Processing checkpoint execution target: {ckpt_dir.name}")

    # CL-03: Clone pre-loaded baseline to evade continuous slow disk I/O reloads
    logger.info("Cloning in-memory baseline model structures...")
    base_hf_copy = copy.deepcopy(base_hf)

    # 2. Merge LoRA Weights
    logger.info("Merging LoRA weights (in-memory alignment)...")
    peft_model = PeftModel.from_pretrained(base_hf_copy, str(ckpt_dir))
    merged_hf = peft_model.merge_and_unload()

    # 3. Wrap in HookedTransformer
    logger.info("Wrapping unified checkpoints inside HookedTransformer...")
    hooked_model = HookedTransformer.from_pretrained(
        base_model_id,
        hf_model=merged_hf,
        device="cuda",
        dtype=torch.float16,
        fold_ln=True
    )

    # 4. Extract Hidden States (CL-04 Configurable target routing)
    out_dir = checkpoints_extracted_dir / ckpt_dir.name
    logger.info(f"Extracting activation vectors to {out_dir} (Batch Footprint: {extract_batch_size})...")

    # CL-06 / E-01: Attention mask is automatically computed and packaged internally by this call
    extract_from_model(hooked_model, stimuli, out_dir, batch_size=extract_batch_size)

    # Flush registers to prevent cross-checkpoint VRAM spikes
    del hooked_model
    del merged_hf
    del peft_model
    del base_hf_copy
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Execute RQ3 Orchestrator via Subprocess
    logger.info("Triggering run_rq3.py evaluation loop...")
    # CL-07: Use sys.executable to maintain environment context isolation inside conda/venvs
    cmd = [
        sys.executable,
        "run_rq3.py",
        "--config", str(config_path),
        "--checkpoint_dir", str(out_dir)
    ]

    try:
        # CL-02: Bound process allocations to prevent indefinite system locks
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"run_rq3.py failed for {ckpt_dir.name}:\n{result.stderr}")
        else:
            logger.info(f"Evaluation complete for {ckpt_dir.name}.")
    except subprocess.TimeoutExpired:
        logger.error(f"Fatal: Evaluation execution timeout reached for {ckpt_dir.name}. Skipping block.")


def main() -> None:
    logger = setup_logger()

    # CL-05: Enforce strict config-driven CLI ingestion arguments parsing
    parser = argparse.ArgumentParser(description="Hardware Checkpoints Evaluation Engine")
    parser.add_argument("--config", required=True, type=str, help="Path to production configuration YAML file.")
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

    # CL-04: Read extraction target routes from active configuration payload
    checkpoints_extracted_base = Path(cfg.get("checkpoints_extracted_dir", "data/processed/checkpoints_extracted"))

    stimuli_path = Path("data/processed/dataset_master_v5.jsonl")
    if not stimuli_path.exists():
        raise FileNotFoundError(f"Dataset missing: {stimuli_path}")
    stimuli = load_stimuli(stimuli_path)

    checkpoints_base = Path("data/processed/checkpoints")

    # Extract standard sequential checkpoint iterations
    checkpoints = sorted(
        [d for d in checkpoints_base.iterdir() if d.is_dir() and "checkpoint" in d.name],
        key=lambda x: int(x.name.split("-")[-1]) if "-" in x.name else 0
    )

    # CL-01: Detect terminal LoRA adapter under either canonical name.
    # Older runs used "final_checkpoint/"; current train_qlora.py emits "final_adapter/".
    for terminal_name in ("final_adapter", "final_checkpoint"):
        terminal_path = checkpoints_base / terminal_name
        if terminal_path.exists():
            logger.info(f"Located unmerged terminal adapter at '{terminal_name}'. Appending to pipeline targets.")
            checkpoints.append(terminal_path)
            break  # only the first match

    if not checkpoints:
        logger.warning(f"No valid adapter modules detected inside target base route: {checkpoints_base}")
        return

    logger.info(f"Allocated {len(checkpoints)} target checkpoints for system execution mapping.")

    # CL-03: Pre-load foundational model footprint once to isolate RAM footprints
    logger.info(f"Pre-loading core uncompressed base model model ({base_model_id}) to RAM storage...")
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

    logger.info("Dynamic checkpoints extraction sequence executed successfully.")


if __name__ == "__main__":
    main()
