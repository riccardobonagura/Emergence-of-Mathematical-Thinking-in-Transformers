#!/usr/bin/env python
"""
train_qlora.py — config-driven QLoRA fine-tuning on MetaMathQA.
Uses a held-out validation split (E-F-02) and settings tuned for a 16GB GPU.
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
    # Guard against the GPT-NeoX vmap/SDPA bug in transformers >= 4.49.
    assert transformers.__version__ < "4.49", (
        f"Fatal: Environment conflict. Transformers version is {transformers.__version__}. "
        "The GPT-NeoX vmap/SDPA runtime bug affects QLoRA backward passes. "
        "Downgrade to transformers < 4.49 to ensure training stability."
    )

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning on MetaMathQA")
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

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(args.lora_config, "r", encoding="utf-8") as f:
        lora_hyperparams = yaml.safe_load(f)

    global_seed = int(config["seed"])
    set_seed(global_seed)

    model_name = config["model_name"]
    # Target modules come from the model profile.
    model_profile = get_model_profile(model_name)

    # Output directory from config.
    output_dir = Path(config.get("checkpoints_dir", "data/processed/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing QLoRA fine-tuning pipeline for model profile: {model_name}")

    # NF4 quantization; bf16 compute (more stable than fp16 on Ada).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    logger.info("Loading quantized base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_profile["hf_path"],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=model_profile.get("attn_implementation", "eager")
    )

    # Gradient checkpointing must be enabled inside kbit preparation, before
    # wrapping with PEFT, or the hooks break.
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # LoRA adapter config.
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

    # Tokenizer with left padding.
    tokenizer = AutoTokenizer.from_pretrained(model_profile["hf_path"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 95/5 train/val split to track validation loss (E-F-02).
    logger.info("Loading MetaMathQA dataset from HuggingFace Hub...")
    raw_dataset = load_dataset("meta-math/MetaMathQA", split="train")

    split_seed = get_seed(global_seed, "dataset_splitting", 0)
    dataset_splits = raw_dataset.train_test_split(test_size=0.05, seed=split_seed)
    train_data = dataset_splits["train"]
    val_data = dataset_splits["test"]

    # 1024 tokens: 512 truncated longer chain-of-thought solutions.
    max_seq_length = 1024

    def formatting_prompts_func(examples):
        texts = []
        for q, r in zip(examples["query"], examples["response"]):
            # MetaMath prompt format.
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

    # Spread tokenization across CPU workers (up to 8).
    num_workers = min(8, os.cpu_count() or 1)
    logger.info(f"Tokenizing across {num_workers} CPU workers...")

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

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        bf16=True,                             # bf16 for RTX 5080 stability
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

    logger.info("Launching fine-tuning loop...")
    trainer.train()

    # Final model checkpoint persistence commit
    final_path = output_dir / "final_checkpoint"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"QLoRA training successfully completed. Hyperplanes stored at {final_path}")


if __name__ == "__main__":
    main()
