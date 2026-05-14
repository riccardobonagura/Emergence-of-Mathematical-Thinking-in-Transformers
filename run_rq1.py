"""
Fase 2a/2b - Orchestratore Analisi Descrittiva (RQ1)
Calcola Isotropia e CKA inter-categoria layer-wise per identificare il layer di emergenza l*.
"""

import json
import torch
import numpy as np
from pathlib import Path

# Importiamo le utility che hai già preparato
from src.metrics.isotropy import run_isotropy_analysis
from src.metrics.cka import compute_cka_intercategory_all_layers, linear_cka

def main():
    # Setup delle directory in linea con l'architettura
    PROC_DIR = Path("data/processed/pythia-1.4b")
    STIMULI_PATH = Path("data/raw/stimuli_arithmetic_v2.jsonl") 
    RESULTS_DIR = Path("results/rq1_emergence")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check dipendenze dati (Fase 1 e Fase 2 concluse)
    if not (PROC_DIR / "metadata.json").exists():
        raise FileNotFoundError(f"Estrazione non completata. Manca {PROC_DIR}/metadata.json")

    print("\n--- AVVIO ANALISI RQ1 (Emergenza) ---")
    
    # 1. Isotropia Layer-wise (Media Similarità Coseno)
    print("\n1. Calcolo Isotropia per categoria...")
    # Sfruttiamo il tuo modulo che usa l'approccio adattivo esatto/Monte Carlo
    df_iso = run_isotropy_analysis(
        processed_dir=str(PROC_DIR),
        stimuli_path=str(STIMULI_PATH),
        output_path=str(RESULTS_DIR / "isotropy_pythia.csv"),
        n_layers=24, # Pythia-1.4B ha 24 layer
        seed=42
    )
    print(f"Isotropia salvata. Trovati {len(df_iso)} record layer/categoria.")

    # 2. CKA Inter-categoria (Matematica vs Testo Generico di Controllo)
    print("\n2. Calcolo CKA Inter-categoria...")
    with open(PROC_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    categories = np.array(metadata["categories"])
    
    # Isolare gli indici: la logica formale contro il background linguistico
    # Nota: aggiorna "CAT-CTRL" e "CAT-ARITH" coi nomi esatti che hai usato nel generatore
    math_indices = np.where(np.isin(categories, ["CAT-ARITH", "CAT-ALGEBRA", "CAT-VERBAL"]))[0]
    ctrl_indices = np.where(categories == "CAT-CTRL")[0]
    
    if len(math_indices) == 0 or len(ctrl_indices) == 0:
        print("ATTENZIONE: Categorie mancanti in metadata.json. Impossibile calcolare CKA.")
        print(f"Categorie rilevate: {np.unique(categories)}")
    else:
        print("\n2. Calcolo CKA Evolutiva (Layer(l) vs Layer(l-1))...")
        from src.metrics import cka

        cka_math_diff = [1.0] # Al layer 0 la similarità con se stesso è 1
        cka_ctrl_diff = [1.0]
        
        # Pre-carichiamo il layer 0 convertendo subito in FP64
        H_prev = torch.load(PROC_DIR / "layer_00.pt", map_location="cpu").numpy().astype(np.float64)
        H_prev_math = H_prev[math_indices]
        H_prev_ctrl = H_prev[ctrl_indices]
        
        for l in range(1, 24):
            # Carichiamo il layer l
            H_curr = torch.load(PROC_DIR / f"layer_{l:02d}.pt", map_location="cpu").numpy().astype(np.float64)
            H_curr_math = H_curr[math_indices]
            H_curr_ctrl = H_curr[ctrl_indices]
            
            # Calcolo evoluzione
            cka_m = linear_cka(H_prev_math, H_curr_math)
            cka_c = linear_cka(H_prev_ctrl, H_curr_ctrl)
            
            cka_math_diff.append(cka_m)
            cka_ctrl_diff.append(cka_c)
            
            H_prev_math = H_curr_math
            H_prev_ctrl = H_curr_ctrl
            print(f"Layer {l:02d} -> CKA Math: {cka_m:.4f} | CKA Ctrl: {cka_c:.4f}")
            
        np.save(RESULTS_DIR / "cka_math_evol.npy", np.array(cka_math_diff))
        np.save(RESULTS_DIR / "cka_ctrl_evol.npy", np.array(cka_ctrl_diff))
        print("CKA evolutiva calcolata e salvata.")
        
    print("\n--- ANALISI DESCRITTIVA COMPLETATA ---")

if __name__ == "__main__":
    main()