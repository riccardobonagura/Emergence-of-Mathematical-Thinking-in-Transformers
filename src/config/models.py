"""
models.py — The Model Registry (Dockstation).
Single source of truth for architectural quirks, LoRA targets, and hardware limits.
"""
from typing import TypedDict, List

class ModelProfile(TypedDict):
    hf_path: str
    target_modules: List[str]
    extract_batch_size: int
    needs_pad_token_fix: bool

MODEL_REGISTRY: dict[str, ModelProfile] = {
    "pythia-1.4b": {
        "hf_path": "EleutherAI/pythia-1.4b",
        "target_modules": ["query_key_value"], # GPT-NeoX fused attention module
        "extract_batch_size": 32,              # Safe for RTX 5080 (16GB) in forward pass
        "needs_pad_token_fix": True,            # NeoX has no default pad_token
        "attn_implementation": "eager",  # GPT-NeoX + new transformers SDPA vmap bug
    },
    
    # --- FUTURE DOCKSTATIONS (Uncomment and test when needed) ---
    # "mistral-7b": {
    #     "hf_path": "mistralai/Mistral-7B-v0.1",
    #     "target_modules": ["q_proj", "v_proj"],
    #     "extract_batch_size": 4,             # OOM risk on 16GB VRAM
    #     "needs_pad_token_fix": True
    # },
}

def get_model_profile(model_name: str) -> ModelProfile:
    """Retrieve model specs or fail explicitly if unsupported."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{model_name}' not supported. "
            "Add its Dockstation profile to src/config/models.py."
        )
    return MODEL_REGISTRY[model_name]