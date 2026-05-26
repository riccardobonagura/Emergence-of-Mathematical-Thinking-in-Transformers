"""
validate_configs.py — Rigorous validator for YAML configuration files.
Ensures type safety and prevents malformed execution runs.
"""

import yaml
import json
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


def validate_extraction(extraction_dir: str) -> list[str]:
    """Validates the integrity of the extraction output directory and metadata."""
    errors = []
    path = Path(extraction_dir)
    
    # 1. Directory exists
    if not path.exists():
        errors.append(f"Extraction dir not found: {path}")
        return errors  # early exit, rest is meaningless
    
    # 2. metadata.json exists and has required keys
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        errors.append("metadata.json missing")
    else:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            
        for key in ["n_layers", "d_model", "n_stimuli", "labels", "stimuli_ids"]:
            if key not in meta:
                errors.append(f"metadata.json missing key: {key}")
                
        if "labels" in meta:
            for field in ["sign", "parity"]:
                if field not in meta["labels"]:
                    errors.append(f"metadata.json missing labels.{field}")
        
        # 3. All layer tensors present
        if "n_layers" in meta:
            for l in range(meta["n_layers"]):
                pt = path / f"layer_{l:02d}.pt"
                if not pt.exists():
                    errors.append(f"Missing tensor: layer_{l:02d}.pt")
                    
    return errors    

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Strict YAML configuration validator.")
    parser.add_argument("--probing", type=str, help="Path to config.yaml (RQ2/RQ3)")
    parser.add_argument("--lora", type=str, help="Path to lora_config.yaml")

    parser.add_argument("--extraction", type=str, default=None,
        metavar="EXTRACTION_DIR",
        help="Validate extraction output dir (e.g. data/processed/pythia-1.4b).")
    
    args = parser.parse_args()

    if not args.probing and not args.lora and not args.extraction:
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

    if args.extraction:
        extr_errors = validate_extraction(args.extraction)
        if extr_errors:
            for err in extr_errors:
                print(f"[FAIL] extraction: {err}")
            errors += 1  # usa errors, non all_errors
        else:
            meta_path = Path(args.extraction) / "metadata.json"
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            print(f"[OK] extraction: {args.extraction} "
              f"({meta.get('n_layers')} layers, {meta.get('n_stimuli')} stimuli)")

        # unico punto di uscita
    if errors > 0:
        print(f"Validation FAILED with {errors} error(s).")
        sys.exit(1)
    else:
        print("All configurations are valid.")
        sys.exit(0)