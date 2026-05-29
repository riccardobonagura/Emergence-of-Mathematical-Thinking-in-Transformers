#!/usr/bin/env python
"""
train_qlora.py — Config-driven QLoRA fine-tuning orchestrator on MetaMathQA.
Enforces statistical guards (T-01 to T-08), strict validation splitting (E-F-02),
and numerical optimization guidelines for modern workstation GPUs (RTX 5080).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from src.config.models import get_model_profile
from src.probing.seeds import get_seed

# Initialize clean logging infrastructure
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_qlora")


def main() -> None:
    # ── T-01: ENV-02 CRITICAL COMPATIBILITY GUARD ─────────────────────────────
    # Ensures the current transformers suite does not trigger the fatal GPT-NeoX
    # vmap/SDPA attention tracing core bug during either caching or backward passes.
    assert transformers.__version__ < "4.49", (
        f"Fatal: Environment conflict. Transformers version is {transformers.__version__}. "
        "The GPT-NeoX vmap/SDPA runtime bug affects QLoRA backward passes. "
        "Downgrade to transformers < 4.49 to ensure training stability."
    )

    # ── T-03: CLI ENTRY POINT CONFIGURATION PATH ROUTING ──────────────────────
    parser = argparse.ArgumentParser(description="Hardened QLoRA Fine-Tuning Suite")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to main operational config (e.g., configs/config_rq2.yaml)"
    )
    parser.add_argument(
        "--lora_config",
        required=True,
        type=str,
        help="Path to lora hyperparameters config (e.g., configs/lora_config.yaml)"
    )
    args = parser.parse_args()

    # Load configurations cleanly with explicit contract mappings
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(args.lora_config, "r", encoding="utf-8") as f:
        lora_hyperparams = yaml.safe_load(f)

    # Centralize seed allocation and setup determinism anchors
    global_seed = int(config["seed"])
    set_seed(global_seed)

    model_name = config["model_name"]
    # Delegate target modules extraction to the architectural SSOT Model profile
    model_profile = get_model_profile(model_name)

    # ── T-04: CONFIG-DRIVEN OUTPUT DIRECTORY RESOLUTION ───────────────────────
    output_dir = Path(config.get("output_dir", "data/processed/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing QLoRA fine-tuning pipeline for model profile: {model_name}")

    # ── T-08: NUMERICAL QUANTIZATION & COMPUTE CONFIGURATION ──────────────────
    # RTX 5080 leverages Ada Lovelace architecture; bfloat16 provides strict numerical
    # stability over traditional fp16 under deep gradients manipulation loops.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    logger.info("Loading quantized base model profile onto hardware device...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_profile["hf_path"],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=model_profile.get("attn_implementation", "eager")
    )

    # ── T-02: CORRECT PEFT K-BIT INITIALIZATION SEQUENCE ──────────────────────
    # Enforces the mandatory sequence constraint: gradient checkpointing must be injected
    # inside the kbit preparation contract wrap itself to prevent broken hooks in PEFT.
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # Instantiate the structural LoRA adapters configuration layout
    peft_config = LoraConfig(
        r=int(lora_hyperparams["r"]),
        lora_alpha=int(lora_hyperparams["lora_alpha"]),
        target_modules=model_profile["target_modules"],
        lora_dropout=float(lora_hyperparams.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Setup tokenizer layout matching left-padding invariants
    tokenizer = AutoTokenizer.from_pretrained(model_profile["hf_path"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── T-05: VALIDATION SPLITTING & OVERFITTING DETECTION (E-F-02) ───────────
    # Resolves Principle E-F-02 (Epoch Sufficiency tracking). Splitting the massive
    # MetaMathQA layout into a clear 95/5 partition to track validation loss trends.
    logger.info("Loading MetaMathQA reasoning dataset from HuggingFace Hub...")
    raw_dataset = load_dataset("meta-math/MetaMathQA", split="train")

    split_seed = get_seed(global_seed, "dataset_splitting", 0)
    dataset_splits = raw_dataset.train_test_split(test_size=0.05, seed=split_seed)
    train_data = dataset_splits["train"]
    val_data = dataset_splits["test"]

    # ── T-06: SEQUENTIAL CHAIN-OF-THOUGHT TRUNCATION MITIGATION ───────────────
    # LIMITATION STATEMENT: The absolute boundary is expanded to 1024 tokens.
    # Original configurations limited at 512 tokens truncated longer GSM8K
    # Multi-Step Chain-of-Thought solutions, training the model on incomplete syntax.
    max_seq_length = 1024

    def formatting_prompts_func(examples):
        texts = []
        for q, r in zip(examples["query"], examples["response"]):
            # Standard structural prompt wrapper for MetaMath instruction sets
            text = f"Instruction: {q}\nResponse: {r}{tokenizer.eos_token}"
            texts.append(text)

        inputs = tokenizer(
            texts,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length"
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    # ── T-07: PARALLEL MULTI-THREADED TOKENIZATION MAPPING ────────────────────
    # Spreads tokenization overhead evenly across CPU workers (Defaults to 8 or local max).
    num_workers = min(8, os.cpu_count() or 1)
    logger.info(f"Commencing parallel text compilation mapping across {num_workers} CPU cores...")

    tokenized_train = train_data.map(
        formatting_prompts_func,
        batched=True,
        num_proc=num_workers,
        remove_columns=train_data.column_names,
        desc="Compiling training dataset token blocks"
    )
    tokenized_val = val_data.map(
        formatting_prompts_func,
        batched=True,
        num_proc=num_workers,
        remove_columns=val_data.column_names,
        desc="Compiling validation dataset token blocks"
    )

    # ── T-08: ACCELERATED PRODUCTION TRAINING ARGUMENTS ───────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        # ── MODIFICA 1: Sostituzione batch size da registro di estrazione a parametro dedicato train_batch_size
        per_device_train_batch_size=int(config.get("train_batch_size", 8)),
        gradient_accumulation_steps=4,
        learning_rate=float(lora_hyperparams["learning_rate"]),
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,                             # HARDWARE STABILITY FIX FOR RTX 5080
        tf32=True,                             # Ampere/Ada Lovelace TensorCore acceleration
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none"                       # Toggle to "wandb" if required
    )

    logger.info("Initializing HuggingFace Trainer context loop structure...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
    )

    # ── MODIFICA 2: Correzione del typo linguistico nei log di avvio training loop
    logger.info("Launching fine-tuning loop...")
    trainer.train()

    # Final model checkpoint persistence commit
    final_path = output_dir / "final_checkpoint"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"QLoRA training successfully completed. Hyperplanes stored at {final_path}")


if __name__ == "__main__":
    main()
