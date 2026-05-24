"""
validate_configs.py — Rigorous validator for YAML configuration files.
Ensures type safety and prevents malformed execution runs.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Literal, TypedDict, cast

# Import constants from the single source of truth
from src.config.categories import ALL_CATS

class PropConfig(TypedDict):
    label_field: str
    category: str | None
    type: Literal["binary", "multiclass"]

class ProbingConfig(TypedDict):
    model_name: str
    output_dir: str
    figures_dir: str
    train_split: float
    seed: int
    n_jobs: int
    properties: Dict[str, PropConfig]
    max_iter: int
    C: float
    solver: str
    multiclass_strategy: str
    bootstrap_n_samples: int
    bootstrap_ci: float
    eval_subset_size: int

def load_and_validate_probing_config(config_path: Path | str) -> ProbingConfig:
    """Load and validate config_rq2/config.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    _validate_probing_schema(raw_config)
    return cast(ProbingConfig, raw_config)

def _validate_probing_schema(config: Dict[str, Any]) -> None:
    """Execute typed assertions on key fields and v5 verification constraints."""
    
    # 1. Verify basic required keys
    required_keys = {
        "model_name", "output_dir", "figures_dir", "train_split", 
        "seed", "n_jobs", "properties"
    }
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Malformed configuration, missing keys: {missing}")

    # 2. Rigorous validation of 'properties'
    props = config.get("properties", {})
    if not isinstance(props, dict) or not props:
        raise ValueError("The 'properties' field must be a populated dictionary.")

    for prop_name, prop_cfg in props.items():
        if "label_field" not in prop_cfg:
            raise KeyError(f"Property '{prop_name}' is missing 'label_field'.")
        
        if "type" not in prop_cfg or prop_cfg["type"] not in ("binary", "multiclass"):
            raise ValueError(f"Invalid type for '{prop_name}'. Expected 'binary' or 'multiclass'.")

        if "category" not in prop_cfg:
            raise KeyError(
                f"FATAL: The 'category' key is missing for property '{prop_name}'. "
                "In v5, omitting it causes cross-category label contamination. If filtering is not desired, set 'category: null'."
            )
        
        cat = prop_cfg["category"]
        if cat is not None and cat not in ALL_CATS:
            raise ValueError(f"Unknown category '{cat}'. Allowed values: {ALL_CATS}")

    # 3. Numeric types and bounds validation (Hardened against -O flag)
    if not (isinstance(config["train_split"], float) and 0.0 < config["train_split"] < 1.0):
        raise ValueError("train_split must be a float between 0 and 1")
    if not isinstance(config["bootstrap_ci"], float):
        raise TypeError("bootstrap_ci must be a float (e.g., 0.95)")
    if not (isinstance(config["bootstrap_n_samples"], int) and config["bootstrap_n_samples"] > 0):
        raise ValueError("bootstrap_n_samples must be an integer > 0")

def load_and_validate_lora_config(config_path: Path | str) -> Dict[str, Any]:
    """Load and validate lora_config.yaml for hardware orchestration (RQ3)."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing LoRA config file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = {"r", "lora_alpha", "bits", "learning_rate"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Malformed LoRA config, missing keys: {missing}")

    # INVERTED GUARD: Explicitly punish manual injection of target_modules
    if "target_modules" in config:
        raise ValueError("target_modules must not be set in lora_config.yaml — delegate to ModelRegistry (src/config/models.py)")

    # Explicit enforcement (Replaced assert)
    if config["bits"] not in (4, 8):
        raise ValueError("Quantization bits must be 4 or 8")
    
    return config

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Strict YAML configuration validator.")
    parser.add_argument("--probing", type=str, help="Path to config.yaml (RQ2/RQ3)")
    parser.add_argument("--lora", type=str, help="Path to lora_config.yaml")
    args = parser.parse_args()

    if not args.probing and not args.lora:
        parser.print_help()
        sys.exit(0)

    errors = 0
    
    print("--- Configuration Validation Report ---")

    if args.probing:
        try:
            load_and_validate_probing_config(args.probing)
            print(f"[OK] Probing config passed: {args.probing}")
        except Exception as e:
            print(f"[FAIL] Probing config error ({args.probing}):\n       -> {e}")
            errors += 1

    if args.lora:
        try:
            load_and_validate_lora_config(args.lora)
            print(f"[OK] LoRA config passed: {args.lora}")
        except Exception as e:
            print(f"[FAIL] LoRA config error ({args.lora}):\n       -> {e}")
            errors += 1

    print("-" * 39)
    if errors > 0:
        print(f"Validation FAILED with {errors} error(s). Aborting execution.")
        sys.exit(1)
    else:
        print("All configurations are valid and strongly typed.")
        sys.exit(0)