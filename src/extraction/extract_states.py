"""
extract_states.py — Activation extraction engine layers.
Intercepts residual stream points at the final token position without hoarding memory.

FIX E-01: Uses model.run_with_hooks with explicit attention_mask forwarding.
HARDENED: Restores ExtractionMetadata TypedDict (ARCH-03) and restricts operator verification.
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


def validate_extraction_tokens(model, stimuli: list[dict]) -> None:
    """
    Filters for math categories and strictly enforces '=' terminal validation.
    Removes the speculative ':' character check to prevent false negatives on valid prompts.
    """
    math_stimuli = [s for s in stimuli if s["category"] in ("CAT-SIGN", "CAT-PARITY")][:10]
    if not math_stimuli:
        logger.warning("No arithmetic categories located for token pre-flight verification.")
        return

    for s in math_stimuli:
        tokens = model.to_tokens(s["text"], prepend_bos=True)
        last_token_str = model.tokenizer.decode(tokens[0, -1])

        # Modifica 3: Rimosso il check su ":" — l'unico token terminale valido è "="
        if "=" not in last_token_str:
            raise ValueError(
                f"Pre-flight token alignment validation failed for sequence: '{s['text']}'. "
                f"Terminal token decoded as '{last_token_str}' instead of an assignment '=' operator."
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

    for i in tqdm(range(0, n_stimuli, batch_size), desc="Extracting residual activations"):
        batch = stimuli[i : i + batch_size]
        texts = [s["text"] for s in batch]

        tokens = model.to_tokens(texts, prepend_bos=True)
        attention_mask = (tokens != model.tokenizer.pad_token_id).long()

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
                cache_store[layer_idx] = value[:, -1, :].detach().cpu().to(torch.float16)
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
        "probe_strategy": "last_token",
        "dataset_version": default_version
    }

    # Enforce transaction isolation safety via verified static types contract
    _atomic_write_json(out_dir / "metadata.json", meta)
    logger.info(f"Extraction execution metrics stored atomically inside {out_dir}")


def load_stimuli(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
