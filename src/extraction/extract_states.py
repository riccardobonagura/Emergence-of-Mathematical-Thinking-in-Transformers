"""
Fase 2 - Estrazione Hidden States (Batch, Mask-based Indexing, Layer-wise)
Isola resid_post garantendo il tracciamento topologico e l'efficienza VRAM.
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformer_lens import HookedTransformer

def load_stimuli(path: str | Path) -> list[dict]:
    """Carica gli stimoli generati nella Fase 1."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_extraction_metadata(stimuli: list[dict], out_dir: Path) -> None:
    """
    Costruisce e salva la mappatura tra l'indice di riga dei tensori estratti
    e gli ID degli stimoli, vitale per la decodifica (Fase 3).
    """
    metadata = {
        "stimuli_ids": [s["id"] for s in stimuli],
        "categories": [s["category"] for s in stimuli],
        "probe_strategies": [s.get("probe_token_strategy", "last_token") for s in stimuli]
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def extract_layer_batched(
    model: HookedTransformer, 
    stimuli: list[dict], 
    layer_idx: int, 
    batch_size: int = 16
) -> torch.Tensor:
    """
    Estrae l'hidden state per un layer specifico procedendo a batch,
    utilizzando l'attention_mask per calcolare l'offset causato dal left-padding.
    """
    layer_activations = []
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    
    for i in tqdm(range(0, len(stimuli), batch_size), desc=f"Layer {layer_idx:02d}"):
        batch = stimuli[i : i + batch_size]
        texts = [s["text"] for s in batch]
        
        # Tokenizzazione esplicita per ottenere input_ids e attention_mask
        tokens_out = model.tokenizer(
            texts, 
            padding=True, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(model.cfg.device)
        
        input_ids = tokens_out["input_ids"]        # [batch, seq_len]
        attention_mask = tokens_out["attention_mask"]
        
        # Calcolo dinamico degli indici effettivi post-padding
        seq_lengths = attention_mask.sum(dim=1)
        max_len = input_ids.shape[1]
        pad_lengths = max_len - seq_lengths
        
        target_indices = []
        for b_idx, stim in enumerate(batch):
            strategy = stim.get("probe_token_strategy", "last_token")
            if strategy == "equals_sign":
                # L'indice originale è shiftato della quantità esatta di padding inserito
                idx = pad_lengths[b_idx] + stim["equals_sign_index"]
            else:
                # last_token è sempre l'ultimo elemento prima del padding finale (o fine sequenza)
                idx = max_len - 1 
            target_indices.append(int(idx))
        
        target_indices_tensor = torch.tensor(target_indices, device=model.cfg.device)
        storage = []
        
        # Definizione dell'hook puramente estrattivo (evita l'overhead di run_with_cache)
        def hook_fn(value, hook):
            # value: [batch_size, seq_len, d_model]
            batch_idx = torch.arange(value.shape[0], device=value.device)
            extracted = value[batch_idx, target_indices_tensor, :].detach().cpu()
            storage.append(extracted)
            return value

        # Forward pass con iniezione hook mirata
        with torch.no_grad():
            model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, hook_fn)])
            
        layer_activations.append(storage[0])
    
    # Pulizia VRAM
    torch.cuda.empty_cache()
    
    # [N, d_model]
    return torch.cat(layer_activations, dim=0)

def main():
    DATA_PATH = Path("data/raw/stimuli_arithmetic_v2.jsonl")
    OUT_DIR = Path("data/processed/pythia-1.4b")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Inizializzazione HookedTransformer in FP16...")
    model = HookedTransformer.from_pretrained(
        "pythia-1.4b", 
        device="cuda", 
        dtype="float16", 
        fold_ln=True
    )
    
    # Configurazione rigorosa del left padding per preservare l'allineamento finale
    model.tokenizer.padding_side = "left" 
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        
    stimuli = load_stimuli(DATA_PATH)
    n_layers = model.cfg.n_layers
    
    # Salvataggio del mapping N -> Metadata (Fondamentale per Fase 3)
    save_extraction_metadata(stimuli, OUT_DIR)
    print(f"Metadata salvati. Avvio estrazione batch per {len(stimuli)} stimoli su {n_layers} layer.")
    
    # Iterazione layer-wise come da protocollo architetturale
    for l in range(n_layers):
        layer_tensor = extract_layer_batched(model, stimuli, layer_idx=l, batch_size=32)
        
        out_file = OUT_DIR / f"layer_{l:02d}.pt"
        torch.save(layer_tensor.half(), out_file)
        print(f"Completato e salvato: {out_file} | Shape: {layer_tensor.shape}")

if __name__ == "__main__":
    import numpy as np
    import random
    
    # Fissaggio seed globale per stabilità d'estrazione
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    main()