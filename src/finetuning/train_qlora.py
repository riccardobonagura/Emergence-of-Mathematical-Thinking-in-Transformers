"""
train_qlora.py — QLoRA fine-tuning engine on MetaMathQA.
Handles 4-bit quantization, LoRA adapter injection, and checkpointing.
Model architecture parameters are injected dynamically from the Registry.
"""

import logging
import torch
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.config.models import get_model_profile

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("qlora_trainer")

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def tokenize_metamath(examples: dict, tokenizer, max_len: int) -> dict:
    """Format MetaMathQA as prompt-completion pairs and tokenize in batch."""
    texts = [
        f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}"
        for q, a in zip(examples["query"], examples["response"])
    ]
    return tokenizer(texts, truncation=True, max_length=max_len)
    
def main():
    logger = setup_logger()
    cfg_path = Path("configs/lora_config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    
    cfg = load_config(cfg_path)
    model_name = cfg.get("model_name", "pythia-1.4b")
    profile = get_model_profile(model_name)
    
    model_id = profile["hf_path"]
    output_dir = Path("data/processed/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if profile.get("needs_pad_token_fix") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Initializing 4-bit quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=cfg.get("double_quant", True),
        bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16
    )

    logger.info("Loading base model...")
    load_kwargs: dict = {"quantization_config": bnb_config, "device_map": "auto"}
    if attn_impl := profile.get("attn_implementation"):
        load_kwargs["attn_implementation"] = attn_impl
    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    logger.info(f"Injecting LoRA adapters (Target Modules: {profile['target_modules']})...")
    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=profile["target_modules"],
        lora_dropout=cfg.get("lora_dropout", 0.1),
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    logger.info("Preparing MetaMathQA dataset...")
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    
    max_seq_length = cfg.get("max_seq_length", 512)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_metamath(x, tokenizer, max_seq_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.get("batch_size", 8),
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg["num_epochs"],
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        save_steps=cfg["save_steps"],
        logging_steps=50,
        fp16=True,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    
    logger.info("Saving final adapter...")
    trainer.save_model(str(output_dir / "final_adapter"))
    logger.info("Fine-tuning complete.")

if __name__ == "__main__":
    main()