"""
validate_configs.py — Rigorous validator for YAML configuration files.
Ensures type safety and prevents malformed execution runs.
"""

import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Literal, TypedDict, cast

# Centralized single source of truth for categories validation
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
    n_permutation_tests: int


def load_and_validate_probing_config(config_path: Path | str) -> ProbingConfig:
    """
    Loads and validates config_rq2/config.yaml.
    Raises FileNotFoundError if the path does not exist, or ValueError on schema failures.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    _validate_probing_schema(raw_config)
    return cast(ProbingConfig, raw_config)


def _validate_probing_schema(config: Dict[str, Any]) -> None:
    """
    Executes typed assertions on key fields and v5 verification constraints.
    Prevents silent runtime KeyErrors by pre-gating bootstrapping and training structures.
    """
    # HARDENED: Added bootstrap_ci and bootstrap_n_samples to required validation keys
    required_keys = {
        "model_name", "output_dir", "figures_dir", "train_split",
        "seed", "n_jobs", "properties", "bootstrap_ci", "bootstrap_n_samples",
        "n_permutation_tests"
    }
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Malformed configuration, missing keys: {missing}")

    # Anti-overwrite guardrail to prevent catastrophic historic weights erasures
    out_dir = Path(config.get("output_dir", ""))
    if out_dir.exists() and any(out_dir.iterdir()):
        logging.warning(
            f" [!] WARNING: Output directory '{out_dir}' is not empty. "
            "Existing outputs may be overwritten."
        )

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
                "In v5, omitting it causes cross-category label contamination. "
                "If filtering is not desired, set 'category: null' explicitly."
            )

        cat = prop_cfg["category"]
        if cat is not None and cat not in ALL_CATS:
            raise ValueError(f"Unknown category '{cat}'. Allowed values from registry: {ALL_CATS}")

    if not (isinstance(config["train_split"], float) and 0.0 < config["train_split"] < 1.0):
        raise ValueError("train_split must be a float between 0.0 and 1.0 exclusive.")
    if not isinstance(config["bootstrap_ci"], float) or not (0.0 < config["bootstrap_ci"] < 1.0):
        raise TypeError("bootstrap_ci must be a float tracking bounds between 0.0 and 1.0 (e.g., 0.95).")
    if not (isinstance(config["bootstrap_n_samples"], int) and config["bootstrap_n_samples"] > 0):
        raise ValueError("bootstrap_n_samples must be a strictly positive integer.")
    if not (isinstance(config["n_permutation_tests"], int) and config["n_permutation_tests"] > 0):
        raise ValueError("n_permutation_tests must be a strictly positive integer.")


def load_and_validate_lora_config(config_path: Path | str) -> Dict[str, Any]:
    """
    Loads and validates lora_config.yaml for hardware orchestration fine-tuning (RQ3).
    Ensures quantization targets are clamped to 4 or 8 bits.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing LoRA config file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = {"r", "lora_alpha", "bits", "learning_rate"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Malformed LoRA config, missing keys: {missing}")

    if "target_modules" in config:
        raise ValueError(
            "target_modules must not be manually set in lora_config.yaml. "
            "Delegate layer selection explicitly to ModelRegistry orchestration wrappers."
        )

    if config["bits"] not in (4, 8):
        raise ValueError("Quantization bits must be exclusively 4 or 8.")

    return config


def validate_extraction(extraction_dir: str) -> list[str]:
    """
    Validates the structural integrity of the extraction output directory and metadata.
    Checks that every expected layer tensor exists on disk.
    """
    errors = []
    path = Path(extraction_dir)

    if not path.exists():
        errors.append(f"Extraction dir not found: {path}")
        return errors

    meta_path = path / "metadata.json"
    if not meta_path.exists():
        errors.append("metadata.json missing from processing footprint.")
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        for key in ["n_layers", "d_model", "n_stimuli", "labels", "stimuli_ids"]:
            if key not in meta:
                errors.append(f"metadata.json missing mandatory registration key: '{key}'")

        if "labels" in meta:
            for field in ["sign", "parity"]:
                if field not in meta["labels"]:
                    errors.append(f"metadata.json missing nested labels track: '{field}'")

        if "n_layers" in meta:
            for l in range(meta["n_layers"]):
                pt = path / f"layer_{l:02d}.pt"
                if not pt.exists():
                    errors.append(f"Missing persistent activation tensor: layer_{l:02d}.pt")

    return errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strict production-grade YAML configuration validator.")
    parser.add_argument("--probing", type=str, help="Path to config.yaml to validate probing targets.")
    parser.add_argument("--lora", type=str, help="Path to lora_config.yaml to validate training targets.")
    parser.add_argument("--extraction", type=str, default=None, metavar="EXTRACTION_DIR",
                        help="Validate persistent weights directory (e.g. data/processed/pythia-1.4b).")

    args = parser.parse_args()

    if not args.probing and not args.lora and not args.extraction:
        parser.print_help()
        sys.exit(0)

    error_count = 0
    print("\n--- Configuration Validation Report ---")

    if args.probing:
        try:
            load_and_validate_probing_config(args.probing)
            print(f"[OK] Probing config schema is structurally sound: {args.probing}")
        except Exception as e:
            print(f"[FAIL] Probing config schema failure ({args.probing}):\n       -> {e}")
            error_count += 1

    if args.lora:
        try:
            load_and_validate_lora_config(args.lora)
            print(f"[OK] LoRA training config is structurally sound: {args.lora}")
        except Exception as e:
            print(f"[FAIL] LoRA training config failure ({args.lora}):\n       -> {e}")
            error_count += 1

    print("-" * 39)

    if args.extraction:
        extr_errors = validate_extraction(args.extraction)
        if extr_errors:
            for err in extr_errors:
                print(f"[FAIL] Extraction directory structural validation error: {err}")
            error_count += 1
        else:
            meta_path = Path(args.extraction) / "metadata.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"[OK] Extraction baseline validation verified: {args.extraction} "
                  f"({meta.get('n_layers')} layers, {meta.get('n_stimuli')} stimuli loaded).")

    if error_count > 0:
        print(f"\n[!] VALIDATION FAILED: {error_count} structural anomaly detected. Exiting loudly.")
        sys.exit(1)
    else:
        print("\n[✔] SUCCESS: All targeted configurations are valid and locked for processing.")
        sys.exit(0)
