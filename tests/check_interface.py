"""
Fase 0.1 - Test Interfacciamento Modello <-> Codice
Obiettivo: Verificare estrazione hidden states H ∈ [1, seq, d_model]
Modello: Pythia-1.4B (EleutherAI)
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer

def run_system_check():
    # 1. Verifica Hardware (RTX 5080 / Blackwell)
    print(f"--- Sistema ---")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}") # Dovrebbe essere (12, 0)
    
    # 2. Caricamento Modello via TransformerLens
    # Nota: Pythia-1.4B ha 24 layer, d_model = 2048
    print(f"\n--- Caricamento Modello ---")
    model = HookedTransformer.from_pretrained(
        "pythia-1.4b",
        device="cuda",
        fold_ln=True, # Semplifica l'analisi geometrica rimuovendo i bias LayerNorm
        center_writing_weights=True
    )
    
    # 3. Test di Inferenza con Hook (Fase 2 della Pipeline)
    prompt = "2 + 2 =" # Stimolo minimale (Fase 1)
    
    # Eseguiamo il forward pass catturando le attivazioni
    # hook_resid_post cattura l'output del layer (Fase 2)
    logits, cache = model.run_with_cache(prompt)
    
    # 4. Verifica Struttura Dati H
    # Estraiamo l'ultimo layer per il test (Layer 23)
    last_layer_key = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    h_vettore = cache[last_layer_key]
    
    print(f"\n--- Verifica Tensori ---")
    print(f"Input: '{prompt}'")
    print(f"Shape di H (resid_post): {h_vettore.shape}") 
    # Atteso: [batch=1, n_tokens, d_model=2048]
    
    # Verifica precisione (BF16/FP16)
    print(f"Dtype: {h_vettore.dtype}")
    
    # 5. Test di Isotropia Locale (RQ1)
    # Calcolo norma elementare per verificare che i dati non siano NaN
    norm = torch.norm(h_vettore).item()
    print(f"Norma di H: {norm:.4f}")
    
    if not np.isnan(norm) and norm > 0:
        print("\nRISULTATO: Interfaccia CORRETTA. Sistema pronto per Fase 1.")
    else:
        print("\nERRORE: Anomalie nei tensori estratti.")

if __name__ == "__main__":
    # Configurazione Seed per riproducibilità (Fase 6)
    torch.manual_seed(42)
    run_system_check()