"""
Fase 2 - Estrazione Hidden States (Batch, Mask-based Indexing, Layer-wise)
Isola resid_post garantendo il tracciamento topologico e l'efficienza VRAM.

Compatibilità dataset v5
------------------------
In v5 tutte le proprietà sondabili (sign, parity) usano la strategia
"last_token": la rappresentazione viene estratta dall'ultimo token reale
della sequenza, che coincide sempre con il token "=" per gli stimoli CAT-*.

Con left-padding, l'ultimo token reale è sempre all'indice `max_len - 1`
indipendentemente dalla lunghezza della sequenza — il branch "equals_sign"
del codice v4 è stato rimosso perché ridondante e potenzialmente errato
(equals_sign_index è ora annidato in token_fields, non top-level).
"""

from __future__ import annotations

import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

def load_stimuli(path: str | Path) -> list[dict]:
    """Carica gli stimoli generati nella Fase 1."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_extraction_metadata(
    stimuli: list[dict],
    out_dir: Path,
    model: "HookedTransformer",
) -> None:
    """
    Costruisce e salva la mappatura tra l'indice di riga dei tensori estratti
    e gli ID degli stimoli, vitale per la decodifica (Fase 3).

    Chiave 'categories' (plurale) — coerente con il lettore in cka.py.
    In v5, probe_strategy è uniformemente 'last_token' per tutti gli stimoli:
    non è più un campo per-stimolo ma un attributo fisso del dataset.
    """
    metadata = {
        "stimuli_ids": [s["id"] for s in stimuli],
        "categories":  [s["category"] for s in stimuli],   # plurale — v5
        "probe_strategy": "last_token",                     # uniforme in v5
        "dataset_version": stimuli[0].get("dataset_version", "unknown") if stimuli else "unknown",
        "n_layers":  model.cfg.n_layers,
        "d_model":   model.cfg.d_model,
        "n_stimuli": len(stimuli),
        "labels": {
            "sign":   [s["labels"].get("sign",   -1) for s in stimuli],
            "parity": [s["labels"].get("parity", -1) for s in stimuli],
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def extract_layer_batched(
    model: "HookedTransformer",
    stimuli: list[dict], 
    layer_idx: int, 
    batch_size: int = 16
) -> torch.Tensor:
    """
    Estrae l'hidden state per un layer specifico procedendo a batch,
    utilizzando l'attention_mask per calcolare l'offset causato dal left-padding.

    Strategia di estrazione (v5)
    ----------------------------
    Tutti gli stimoli v5 usano la strategia 'last_token'.
    Con left-padding, il token reale finale è sempre all'indice `max_len - 1`:
    il padding si accumula a sinistra, il contenuto termina sempre a destra.

    Il ramo 'equals_sign' del codice v4 è stato rimosso perché:
      - equals_sign_index == last_token_index per costruzione in v5
      - equals_sign_index è annidato in token_fields (non top-level)
    """
    layer_activations = []
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    
    for i in tqdm(range(0, len(stimuli), batch_size), desc=f"Layer {layer_idx:02d}"):
        batch = stimuli[i : i + batch_size]
        texts = [s["text"] for s in batch]
        
        tokens_out = model.tokenizer(
            texts, 
            padding=True, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(model.cfg.device)
        
        input_ids     = tokens_out["input_ids"]        # [batch, seq_len]
        attention_mask = tokens_out["attention_mask"]
        max_len        = input_ids.shape[1]

        # Con left-padding, l'ultimo token reale è sempre all'indice max_len - 1.
        # Non serve calcolare l'offset per-stimolo: la posizione è fissa.
        target_idx = max_len - 1
        storage = []
        
        def hook_fn(value, hook):
            # value: [batch_size, seq_len, d_model]
            extracted = value[:, target_idx, :].detach().cpu()  # [batch, d_model]
            storage.append(extracted)
            return value

        with torch.no_grad():
            model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, hook_fn)])
            
        layer_activations.append(storage[0])
    
    torch.cuda.empty_cache()
    return torch.cat(layer_activations, dim=0)   # [N, d_model]

def main():
    from transformer_lens import HookedTransformer

    # Dataset master v5 (merge di CAT-SIGN, CAT-PARITY, CTRL-NEU, CTRL-NUM)
    DATA_PATH = Path("data/processed/dataset_master_v5.jsonl")
    OUT_DIR   = Path("data/processed/pythia-1.4b")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Inizializzazione HookedTransformer in FP16...")
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-1.4b",
        device="cuda", 
        dtype=torch.float16,
        fold_ln=True,    # fold LayerNorm per analisi mechanistic; resid_post invariato
    )
    
    # Left-padding: garantisce che l'indice max_len-1 punti sempre all'ultimo token reale
    model.tokenizer.padding_side = "left" 
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        
    stimuli  = load_stimuli(DATA_PATH)
    n_layers = model.cfg.n_layers   # 24 per Pythia-1.4B
    
    save_extraction_metadata(stimuli, OUT_DIR, model)
    print(f"Metadata salvati. "
          f"Avvio estrazione per {len(stimuli)} stimoli su {n_layers} layer.")
    
    for l in range(n_layers):
        layer_tensor = extract_layer_batched(model, stimuli, layer_idx=l, batch_size=32)
        out_file = OUT_DIR / f"layer_{l:02d}.pt"
        torch.save(layer_tensor.half(), out_file)
        print(f"  layer_{l:02d}.pt  shape={layer_tensor.shape}")

if __name__ == "__main__":
    import numpy as np
    import random
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    main()