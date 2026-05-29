"""
models.py — Model profile registry and structural schemas.
Splits Required and Optional fields to ensure type checkers block missing core fields.
"""

from typing import List, TypedDict

class ModelProfileOptional(TypedDict, total=False):
    """Fields that can be omitted to let TransformerLens resolve defaults dynamically."""
    attn_implementation: str

class ModelProfile(ModelProfileOptional):
    """Absolute invariants required to drive the extraction and fine-tuning pipelines."""
    hf_path: str
    target_modules: List[str]
    extract_batch_size: int
    needs_pad_token_fix: bool


def get_model_profile(model_name: str) -> ModelProfile:
    """Single Source of Truth (SSOT) profile factory registry."""
    profiles = {
        "pythia-1.4b": ModelProfile(
            hf_path="EleutherAI/pythia-1.4b",
            target_modules=["query_key_value"],
            extract_batch_size=32,
            needs_pad_token_fix=True,
            attn_implementation="eager"
        )
    }
    if model_name not in profiles:
        raise ValueError(f"Unknown architectural model name profile: {model_name}")
    return profiles[model_name]
