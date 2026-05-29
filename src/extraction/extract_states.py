"""
extract_states.py — Activation extraction engine layers.
Intercepts residual stream points at the final token position without hoarding memory.

FIX E-01: Uses model.run_with_hooks with explicit attention_mask forwarding.
HARDENED: Restores ExtractionMetadata TypedDict (ARCH-03) and restricts operator verification.

Last-token gather: to_tokens right-pads and pad_id == BOS == eos == 0 for Pythia,
so the terminal token is located by scanning non-pad positions from the right
rather than reading value[:, -1, :] (which would hit a pad for short sequences).
"""

import json
import logging
import torch
from pathlib import Path
from typing import List, TypedDict
from tqdm import tqdm

from src.probing.io_utils import _atomic_write_json

logger = logging.getLogger("extract_states")


# ── ARCH-03: EXPLICIT METADATA CONTRACT PAYLOAD STRUCTURES (Ripristino v1) ───
class ExtractionMetadataLabels(TypedDict):
    sign: List[int]
    parity: List[int]
    operand1: List[int]
    operand2: List[int]


class ExtractionMetadata(TypedDict):
    model_name: str
    n_layers: int
    d_model: int
    n_stimuli: int
    stimuli_ids: List[str]
    categories: List[str]
    labels: ExtractionMetadataLabels
    probe_strategy: str
    dataset_version: str


def _resolve_pad_id(model) -> int:
    """Pythia's tokenizer has pad_token_id=None; TL aliases pad→eos→0 (== BOS).
    Fall back to eos_token_id so the value is always defined and never None."""
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    return pad_id


def _last_token_indices(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Index of the last non-pad token per row, robust to BOS sharing pad's id.

    Right padding puts pads at the tail, so scanning from the right for the first
    non-pad lands on the real terminal token ("=" for math rows) regardless of the
    prepended BOS also being id 0.
    """
    non_pad = (tokens != pad_id)
    return (tokens.shape[1] - 1) - non_pad.int().flip(1).argmax(dim=1)


def validate_extraction_tokens(model, stimuli: list[dict]) -> None:
    """
    Filters for math categories and verifies the *gathered* terminal token decodes
    to '=' under the real batched/padded path used by extract_from_model. Tokenizing
    a mixed-length batch (not single strings) is what surfaces right-padding /
    last-token misalignment.
    """
    math_stimuli = [s for s in stimuli if s["category"] in ("CAT-SIGN", "CAT-PARITY")][:10]
    if not math_stimuli:
        logger.warning("No arithmetic categories located for token pre-flight verification.")
        return

    pad_id = _resolve_pad_id(model)
    tokens = model.to_tokens([s["text"] for s in math_stimuli], prepend_bos=True)
    last_idx = _last_token_indices(tokens, pad_id)

    for row, s in enumerate(math_stimuli):
        last_token_str = model.tokenizer.decode(tokens[row, last_idx[row]])
        if "=" not in last_token_str:
            raise ValueError(
                f"Pre-flight token alignment validation failed for sequence: '{s['text']}'. "
                f"Gathered terminal token decoded as '{last_token_str}' instead of an assignment '=' operator."
            )


def extract_from_model(model, stimuli: list[dict], out_dir: Path, batch_size: int = 32) -> None:
    """Runs data sequence passes, captures residual targets, and commits files atomically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_extraction_tokens(model, stimuli)

    n_layers = model.cfg.n_layers
    n_stimuli = len(stimuli)
    d_model = model.cfg.d_model

    # Initialize CPU-bound accumulators to offload active GPU memory pressure
    layer_tensors = {l: torch.zeros((n_stimuli, d_model), dtype=torch.float16) for l in range(n_layers)}

    pad_id = _resolve_pad_id(model)

    for i in tqdm(range(0, n_stimuli, batch_size), desc="Extracting residual activations"):
        batch = stimuli[i : i + batch_size]
        texts = [s["text"] for s in batch]

        tokens = model.to_tokens(texts, prepend_bos=True)
        # pad_id collides with BOS (both id 0 for Pythia), so derive the mask from
        # non-pad positions but keep the real prepended BOS attended (column 0).
        attention_mask = (tokens != pad_id).long()
        attention_mask[:, 0] = 1

        # Gather each row's true terminal token instead of a blind value[:, -1, :]:
        # to_tokens right-pads, so the last position is a pad for shorter sequences.
        last_idx = _last_token_indices(tokens, pad_id)
        row_idx = torch.arange(tokens.shape[0])

        cache_store = {}
        all_hooks = []

        # EXTRACTION-POSITION ASYMMETRY (RQ1 limitation, document in thesis):
        # We always read the last token (value[:, -1, :]). For CAT-SIGN / CAT-PARITY
        # that token is "=", i.e. the representation of the *expected result*. For
        # CTRL-NEU / CTRL-NUM there is no "=", so the last token is the sentence-final
        # token (a word or "."). At upper layers — which are maximally contextualised —
        # "operator awaiting a result" and "end of a declarative sentence" are different
        # computational states. Part of any math-vs-ctrl CKA/isotropy divergence in RQ1
        # may therefore reflect this positional asymmetry rather than mathematical
        # content. The asymmetry is inherent (controls cannot naturally end in "=") and
        # must be reported as an RQ1 caveat; it does not affect within-math RQ2 probing.
        def make_hook(layer_idx: int):
            def hook_fn(value, hook):
                gathered = value[row_idx, last_idx, :]
                cache_store[layer_idx] = gathered.detach().cpu().to(torch.float16)
                return value
            return hook_fn

        for l in range(n_layers):
            all_hooks.append((f"blocks.{l}.hook_resid_post", make_hook(l)))

        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                attention_mask=attention_mask,
                fwd_hooks=all_hooks
            )

        for l in range(n_layers):
            layer_tensors[l][i : i + len(batch)] = cache_store[l]

    for l in range(n_layers):
        torch.save(layer_tensors[l], out_dir / f"layer_{l:02d}.pt")

    # Modifica 2: Estrazione controllata dei metadati di versione con fallback sicuro
    default_version = stimuli[0].get("dataset_version", "v5") if stimuli else "v5"

    meta: ExtractionMetadata = {
        "model_name": model.cfg.model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_stimuli": n_stimuli,
        "stimuli_ids": [s["id"] for s in stimuli],
        "categories": [s["category"] for s in stimuli],
        "labels": {
            "sign": [s.get("labels", {}).get("sign", -1) for s in stimuli],
            "parity": [s.get("labels", {}).get("parity", -1) for s in stimuli],
            "operand1": [s.get("labels", {}).get("operand1", 0) for s in stimuli],
            "operand2": [s.get("labels", {}).get("operand2", 0) for s in stimuli]
        },
        "probe_strategy": "gathered_terminal",
        "dataset_version": default_version
    }

    # Enforce transaction isolation safety via verified static types contract
    _atomic_write_json(out_dir / "metadata.json", meta)
    logger.info(f"Extraction execution metrics stored atomically inside {out_dir}")


def load_stimuli(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    """Base (pre-fine-tuning) extraction runner feeding RQ1/RQ2.

    Loads Pythia with the *same* HookedTransformer config as the RQ3 checkpoint
    loop (fold_ln=True, fp16) so base-vs-checkpoint CKA/Frobenius drift stays
    comparable.
    """
    import argparse

    import transformers
    import yaml
    from transformer_lens import HookedTransformer

    from src.config.models import get_model_profile

    # ENV-02: GPT-NeoX vmap/SDPA bug in transformers >= 4.49
    assert transformers.__version__ < "4.49", (
        f"transformers {transformers.__version__} has a vmap/SDPA bug with GPT-NeoX. "
        "Pin to <4.49: pip install 'transformers>=4.46,<4.49'"
    )

    parser = argparse.ArgumentParser(description="Base hidden-state extraction (RQ1/RQ2)")
    parser.add_argument("--config", required=True, type=str, help="Path to the master configuration YAML file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model_name", "pythia-1.4b")
    profile = get_model_profile(model_name)

    stimuli_path = Path("data/processed/dataset_master_v5.jsonl")
    if not stimuli_path.exists():
        raise FileNotFoundError(f"Dataset missing: {stimuli_path}")
    stimuli = load_stimuli(stimuli_path)

    logger.info(f"Loading {profile['hf_path']} into HookedTransformer (fp16, fold_ln=True)...")
    model = HookedTransformer.from_pretrained(
        profile["hf_path"],
        device="cuda",
        dtype=torch.float16,
        fold_ln=True,
    )

    out_dir = Path("data/processed") / model_name
    extract_from_model(model, stimuli, out_dir, batch_size=profile["extract_batch_size"])
    logger.info(f"Base extraction complete for {model_name}.")


if __name__ == "__main__":
    main()
