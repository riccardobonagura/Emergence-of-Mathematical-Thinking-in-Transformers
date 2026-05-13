"""
Metriche per il confronto tra spazi di rappresentazione.

Contenuti previsti (Fase 3 - CKA):
- Implementazione del Centered Kernel Alignment (CKA) in variante lineare
  (Linear CKA), utilizzando matrici di Gram lineari K = X X^T, L = Y Y^T.
- Centratura delle matrici di Gram tramite matrice di centratura H:
      H = I_n - (1/n) * 1 1^T
  e K_c = H K H, L_c = H L H.

Questo modulo espone un'API pubblica compatta per CKA lineare:
- `linear_cka(X, Y)`
- `cka_matrix_across_layers(activations_per_layer)`
"""

from __future__ import annotations

"""
cka_analysis.py
===============
Modulo per il calcolo della Centered Kernel Alignment (CKA) nella pipeline
"Dinamica Geometrica nei Transformer".

Implementa tre modalità d'uso (cfr. Pipeline.md, Fase 5 e file CKA.md):

  1. CKA intra-modello   → matrice L×L che confronta ogni coppia di layer
                           sullo stesso modello; produce la heatmap triangolare
                           che rivela la struttura gerarchica della rete.

  2. CKA inter-categoria → per ogni layer, confronta le rappresentazioni di
                           stimoli matematici vs. testo generico; identifica
                           il layer l* di biforcazione geometrica (RQ1).

  3. CKA cross-temporale → confronta, layer per layer, le rappresentazioni del
                           modello base con quelle dei checkpoint QLoRA;
                           misura la deformazione plastica del manifold (RQ3).

Dipendenze: numpy, torch, tqdm, pathlib
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Iterable


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 1 — Primitiva matematica: center_gram
# ─────────────────────────────────────────────────────────────────────────────

def center_gram(K: np.ndarray) -> np.ndarray:
    """
    Applica la matrice di centratura di Gower alla Gram matrix K.

    La centratura rimuove la dipendenza dalla media delle rappresentazioni,
    rendendo la metrica invariante a traslazioni costanti nello spazio latente.

    Matematicamente:
        H  = I_n  -  (1/n) * 1*1^T        ← matrice di centratura n×n
        K' = H K H                          ← Gram centrata

    Note implementative:
        Invece di costruire esplicitamente H (che ha dimensione n×n e
        richiederebbe O(n²) di memoria extra), si sfrutta l'identità algebrica:
            H K H = K - row_mean - col_mean + global_mean
        dove row_mean[i,j] = mean(K[i,:]) e col_mean[i,j] = mean(K[:,j]).
        Questo è numericamente equivalente ma più efficiente.

    Args:
        K (np.ndarray): Gram matrix di forma (n, n), tipicamente K = X @ X.T

    Returns:
        np.ndarray: Gram matrix centrata K', stessa forma di K.
    """
    n = K.shape[0]

    # Media per riga:  vettore (n,), broadcast a (n, n)
    row_mean = K.mean(axis=1, keepdims=True)   # shape (n, 1)

    # Media per colonna: vettore (n,), broadcast a (n, n)
    col_mean = K.mean(axis=0, keepdims=True)   # shape (1, n)

    # Media globale: scalare, serve per correggere il doppio conteggio
    global_mean = K.mean()

    # H K H  =  K  -  row_mean  -  col_mean  +  global_mean
    K_centered = K - row_mean - col_mean + global_mean

    return K_centered


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 2 — Primitiva matematica: linear_cka
# ─────────────────────────────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calcola la CKA lineare tra due insiemi di rappresentazioni X e Y.

    La CKA misura quanto la *struttura relazionale* tra i campioni sia simile
    nei due spazi — indipendentemente dalla dimensionalità (d1 può ≠ d2) e
    invariante a trasformazioni ortogonali e scalamenti isotropi.

    Formula:
        K  = X @ X.T    ← Gram matrix di X, shape (n, n)
        L  = Y @ Y.T    ← Gram matrix di Y, shape (n, n)
        K' = HKH        ← centratura
        L' = HLH

        CKA(X, Y) = <K', L'>_F / sqrt(<K', K'>_F * <L', L'>_F)

    dove <A, B>_F = tr(A^T B) = sum(A * B)  (prodotto di Frobenius).

    Il valore è in [0, 1]:
        1.0  → geometrie identiche (a meno di rotazioni/scalamenti)
        0.0  → geometrie completamente indipendenti

    Args:
        X (np.ndarray): Attivazioni del layer l1, forma (n_samples, d1).
                        n_samples deve essere uguale per X e Y.
        Y (np.ndarray): Attivazioni del layer l2, forma (n_samples, d2).

    Returns:
        float: Valore CKA in [0, 1].

    Raises:
        ValueError: Se X e Y hanno numero di campioni diverso.
        RuntimeError: Se la norma è zero (rappresentazioni costanti).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X e Y devono avere lo stesso numero di campioni. "
            f"Ricevuti: X.shape={X.shape}, Y.shape={Y.shape}"
        )

    n = X.shape[0]

    # ── Step 1: Gram matrices (kernel lineare) ──────────────────────────────
    # K[i,j] = X[i] · X[j]  (prodotto scalare tra il campione i e il campione j)
    # Questa matrice cattura la struttura relazionale interna allo spazio X.
    K = X @ X.T   # (n, n)
    L = Y @ Y.T   # (n, n)

    # ── Step 2: Centratura ──────────────────────────────────────────────────
    K_c = center_gram(K)   # K' = HKH
    L_c = center_gram(L)   # L' = HLH

    # ── Step 3: Prodotto di Frobenius (= tr(A^T B) = sum(A * B) element-wise)
    # <K', L'>_F misura quanto le due strutture di similarità si "allineano"
    hsic_kl = np.sum(K_c * L_c)   # HSIC(X, Y) ∝ tr(K' L')

    # Norme di Frobenius al quadrato (auto-similarità, usate per normalizzare)
    hsic_kk = np.sum(K_c * K_c)   # HSIC(X, X)
    hsic_ll = np.sum(L_c * L_c)   # HSIC(Y, Y)

    # ── Step 4: Normalizzazione ─────────────────────────────────────────────
    denom = np.sqrt(hsic_kk * hsic_ll)

    if denom < 1e-10:
        # Caso degenere: una delle due rappresentazioni è costante
        # (tutti i vettori identici → Gram matrix centrata = 0)
        raise RuntimeError(
            "Norma quasi-zero: le rappresentazioni sono costanti o quasi. "
            "Verifica che gli hidden state siano stati estratti correttamente."
        )

    return float(hsic_kl / denom)


def cka_matrix_across_layers(
    activations_per_layer: Iterable[np.ndarray | torch.Tensor],
) -> np.ndarray:
    """
    Costruisce una matrice CKA layer x layer a partire da una sequenza di
    attivazioni per layer.

    API high-level pensata per pipeline in memoria:
        - input: lista/iterabile di array [n_samples, d_l]
        - output: matrice simmetrica [L, L] con CKA lineare

    Args:
        activations_per_layer: sequenza di tensori/array, uno per layer.
            Ogni elemento deve avere forma [n_samples, d_l], con lo stesso
            numero di campioni n_samples tra layer.

    Returns:
        np.ndarray: Matrice CKA [L, L].
    """
    layers = [_to_numpy_2d(x) for x in activations_per_layer]
    n_layers = len(layers)
    if n_layers == 0:
        raise ValueError("activations_per_layer e' vuoto.")

    n_samples = layers[0].shape[0]
    for idx, arr in enumerate(layers):
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Numero campioni incoerente al layer {idx}: "
                f"{arr.shape[0]} vs atteso {n_samples}"
            )

    cka_mat = np.zeros((n_layers, n_layers), dtype=np.float64)
    for i in range(n_layers):
        for j in range(i, n_layers):
            val = linear_cka(layers[i], layers[j])
            cka_mat[i, j] = val
            cka_mat[j, i] = val
    return cka_mat


def _to_numpy_2d(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Atteso array 2D [n, d], ricevuto shape={arr.shape}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 3 — Subsampling riproducibile
# ─────────────────────────────────────────────────────────────────────────────

def subsample_indices(n_total: int, n_sub: int, seed: int = 42) -> np.ndarray:
    """
    Seleziona n_sub indici casuali da [0, n_total) in modo riproducibile.

    Il subsampling è necessario perché la Gram matrix ha dimensione (n × n):
    con n = 1500 stimoli → 1500² = 2.25M elementi per layer → 576 calcoli per
    la matrice L×L. Con n_sub = 512 il costo diventa gestibile su CPU in pochi
    minuti (cfr. nota computazionale in Pipeline.md, Fase 5).

    La stima CKA su subsample è statisticamente stabile per n_sub ≥ 256
    (verificato empiricamente in letteratura).

    Args:
        n_total (int): Numero totale di campioni disponibili.
        n_sub   (int): Numero di campioni da selezionare (n_sub ≤ n_total).
        seed    (int): Seed per la riproducibilità (default: 42).

    Returns:
        np.ndarray: Array di n_sub indici interi unici, ordinati.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_sub, n_total), replace=False)
    return np.sort(indices)  # ordinati per facilità di debug


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 4 — Uso 1: CKA intra-modello (matrice L×L)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cka_matrix_intramodel(
    hidden_states_dir: Path,
    n_layers: int,
    n_sub: int = 512,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    Calcola la matrice CKA L×L confrontando ogni coppia di layer dello stesso
    modello sullo stesso dataset di stimoli.

    Questa è la "Fase 5 — Invarianza Topologica" della pipeline (Pipeline.md).
    La heatmap risultante rivela:
      - Blocchi contigui ad alta CKA → "stadi" del processo di elaborazione
      - Salti bruschi → transizioni rappresentazionali, potenziale l* (RQ1)

    I file degli hidden state si aspettano in:
        hidden_states_dir/layer_00.pt  → Tensor (N, d)
        hidden_states_dir/layer_01.pt
        ...

    Args:
        hidden_states_dir (Path): Directory con i file layer_XX.pt.
        n_layers          (int):  Numero totale di layer del modello.
        n_sub             (int):  Dimensione del subsample (default: 512).
        seed              (int):  Seed per il subsampling (default: 42).
        device            (str):  "cpu" o "cuda" per il caricamento tensori.

    Returns:
        np.ndarray: Matrice simmetrica CKA di forma (n_layers, n_layers),
                    valori in [0, 1]. L'elemento [i, j] è CKA(layer_i, layer_j).
    """
    # ── Caricamento del primo layer per determinare N (numero stimoli) ──────
    first_layer_path = hidden_states_dir / "layer_00.pt"
    H_first = torch.load(first_layer_path, map_location=device).cpu().numpy()  # (N, d)
    N = H_first.shape[0]

    # ── Selezione fissa degli indici di subsample ───────────────────────────
    # IMPORTANTE: gli stessi indici per tutti i layer → confronto consistente
    sub_idx = subsample_indices(n_total=N, n_sub=n_sub, seed=seed)

    # ── Pre-caricamento degli hidden state per tutti i layer ────────────────
    # Carichiamo prima tutto in memoria (fattibile se n_sub=512 e d≤3072):
    # 512 × 3072 × 4 bytes ≈ 6 MB per layer × 32 layer ≈ 192 MB totali
    print(f"Caricamento hidden state per {n_layers} layer "
          f"(subsample N_sub={n_sub} su N={N})...")

    H_sub_all = []  # lista di array (n_sub, d), uno per layer
    for l in range(n_layers):
        layer_path = hidden_states_dir / f"layer_{l:02d}.pt"
        H_l = torch.load(layer_path, map_location=device).cpu().numpy()  # (N, d)
        H_sub_all.append(H_l[sub_idx])  # (n_sub, d) — subsample fisso

    # ── Calcolo della matrice CKA L×L ───────────────────────────────────────
    cka_matrix = np.zeros((n_layers, n_layers))

    # La matrice è simmetrica: CKA(l1, l2) = CKA(l2, l1)
    # → calcoliamo solo il triangolo superiore (inclusa diagonale) e specchiamo
    total_pairs = n_layers * (n_layers + 1) // 2
    pbar = tqdm(total=total_pairs, desc="CKA intra-modello")

    for l1 in range(n_layers):
        for l2 in range(l1, n_layers):
            cka_val = linear_cka(H_sub_all[l1], H_sub_all[l2])
            cka_matrix[l1, l2] = cka_val
            cka_matrix[l2, l1] = cka_val  # simmetria
            pbar.update(1)

    pbar.close()
    return cka_matrix


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 5 — Uso 2: CKA inter-categoria (RQ1 — biforcazione geometrica)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cka_intercategory(
    H_math: np.ndarray,
    H_generic: np.ndarray,
) -> float:
    """
    Calcola CKA tra le rappresentazioni di stimoli matematici e testo generico
    per un singolo layer.

    Un valore basso indica che le geometrie delle due categorie sono distanti:
    il modello le rappresenta come strutturalmente diverse in quel layer.
    Iterare su tutti i layer produce la curva CKA_inter(l) che identifica l*.

    Per il confronto layer successivo (RQ1, approccio alternativo descritto
    in CKA.md, sezione 2A), si chiama:
        cka_l = compute_cka_intercategory(H_math_l, H_math_l_minus_1)
    confrontandolo con:
        cka_l = compute_cka_intercategory(H_generic_l, H_generic_l_minus_1)

    Args:
        H_math    (np.ndarray): Hidden state stimoli matematici, forma (n1, d).
        H_generic (np.ndarray): Hidden state testo generico,     forma (n2, d).
                                n1 e n2 possono essere diversi (CKA lo gestisce
                                tramite la Gram matrix n×n separata per i due set).

    Returns:
        float: Valore CKA inter-categoria in [0, 1].

    Note:
        Se n1 ≠ n2, la CKA non è direttamente applicabile nella forma standard
        (che richiede le stesse n righe). In quel caso si usa un subsample
        bilanciato: min(n1, n2) campioni da ciascuna categoria.
    """
    n1, n2 = H_math.shape[0], H_generic.shape[0]

    if n1 != n2:
        # Subsampling bilanciato per garantire n uguale
        n_common = min(n1, n2)
        rng = np.random.default_rng(42)
        idx_math    = rng.choice(n1, size=n_common, replace=False)
        idx_generic = rng.choice(n2, size=n_common, replace=False)
        H_math    = H_math[idx_math]
        H_generic = H_generic[idx_generic]

    return linear_cka(H_math, H_generic)


def compute_cka_intercategory_all_layers(
    hidden_states_dir: Path,
    n_layers: int,
    math_indices: np.ndarray,
    generic_indices: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Calcola CKA inter-categoria per ogni layer del modello.

    Produce il vettore CKA_inter(l) per l in {0, ..., L-1}.
    Un calo brusco a partire da l* indica l'emergenza della biforcazione
    geometrica tra matematica e linguaggio generico (RQ1).

    Args:
        hidden_states_dir (Path):         Directory con layer_XX.pt.
        n_layers          (int):           Numero di layer.
        math_indices      (np.ndarray):    Indici degli stimoli matematici
                                           nel tensore degli hidden state.
        generic_indices   (np.ndarray):    Indici degli stimoli generici.
        device            (str):           "cpu" o "cuda".

    Returns:
        np.ndarray: Array di forma (n_layers,) con i valori CKA inter-categoria.
    """
    cka_intercategory = np.zeros(n_layers)

    for l in tqdm(range(n_layers), desc="CKA inter-categoria per layer"):
        layer_path = hidden_states_dir / f"layer_{l:02d}.pt"
        H_l = torch.load(layer_path, map_location=device).cpu().numpy().astype(np.float64)  # (N, d)

        H_math    = H_l[math_indices]     # (n_math,    d)
        H_generic = H_l[generic_indices]  # (n_generic, d)

        cka_intercategory[l] = compute_cka_intercategory(H_math, H_generic)

    return cka_intercategory


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 6 — Uso 3: CKA cross-temporale (RQ3 — fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cka_cross_temporal(
    base_hidden_states_dir: Path,
    checkpoint_hidden_states_dirs: dict[str, Path],
    n_layers: int,
    n_sub: int = 512,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Calcola CKA layer-per-layer tra il modello base e ogni checkpoint QLoRA.

    Per ogni checkpoint c e layer l, calcola:
        S_l^c = CKA(H_l^{base}, H_l^{checkpoint_c})

    Questo produce, per ogni checkpoint, un vettore (n_layers,) che mostra
    quali layer hanno subito la maggiore ristrutturazione geometrica durante
    il fine-tuning (RQ3, cfr. Pipeline.md Fase 10 e CKA.md sezione 2B).

    CKA_drift(l, c) = 1 - S_l^c  →  vicino a 0 = layer invariato,
                                      vicino a 1 = layer fortemente riorganizzato.

    Args:
        base_hidden_states_dir        (Path): Directory hidden state modello base.
        checkpoint_hidden_states_dirs (dict): Mapping step → Path directory ckpt.
                                              Es: {"ckpt_500": Path("data/..."),
                                                   "ckpt_1000": Path("data/...")}
        n_layers (int):  Numero di layer del modello.
        n_sub    (int):  Subsample per efficienza computazionale (default 512).
        seed     (int):  Seed per riproducibilità (default 42).
        device   (str):  "cpu" o "cuda".

    Returns:
        dict[str, np.ndarray]: Mapping step → array (n_layers,) di valori CKA.
                               La chiave "base" contiene il vettore di ones
                               (CKA con se stesso = 1.0 per sanity check).
    """
    # ── Subsampling fisso: stesso per tutti i checkpoint ────────────────────
    # Carichiamo il layer 0 del base solo per conoscere N
    H_tmp = torch.load(
        base_hidden_states_dir / "layer_00.pt", map_location=device
    ).numpy()
    N = H_tmp.shape[0]
    del H_tmp

    sub_idx = subsample_indices(n_total=N, n_sub=n_sub, seed=seed)

    # ── Caricamento hidden state del modello base (subsample) ───────────────
    print("Caricamento hidden state modello base...")
    H_base = []
    for l in range(n_layers):
        H_l = torch.load(
            base_hidden_states_dir / f"layer_{l:02d}.pt", map_location=device
        ).numpy().astype(np.float64)
        H_base.append(H_l[sub_idx])  # (n_sub, d)

    # ── CKA cross-temporale per ogni checkpoint ──────────────────────────────
    results = {}

    # Sanity check: CKA(base, base) deve essere 1.0 per ogni layer
    results["base"] = np.ones(n_layers)

    for ckpt_name, ckpt_dir in checkpoint_hidden_states_dirs.items():
        print(f"\nCalcolo CKA cross-temporale: base vs {ckpt_name}...")
        cka_values = np.zeros(n_layers)

        for l in tqdm(range(n_layers), desc=f"  Layer ({ckpt_name})"):
            H_ckpt_l = torch.load(
                ckpt_dir / f"layer_{l:02d}.pt", map_location=device
            ).numpy().astype(np.float64)
            H_ckpt_sub = H_ckpt_l[sub_idx]  # (n_sub, d) — stesso subsample

            # CKA tra rappresentazioni del base e del checkpoint al layer l
            cka_values[l] = linear_cka(H_base[l], H_ckpt_sub)

        results[ckpt_name] = cka_values

    return results


def compute_cka_drift(cka_cross_temporal: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Calcola il CKA drift: drift(l, c) = 1 - CKA(base_l, ckpt_c_l).

    Un drift alto indica forte riorganizzazione geometrica in quel layer.
    Rappresenta la "distanza" dal modello base nella metrica CKA.

    Args:
        cka_cross_temporal (dict): Output di compute_cka_cross_temporal().

    Returns:
        dict[str, np.ndarray]: Mapping ckpt_name → array drift (n_layers,).
                               La chiave "base" viene saltata (drift = 0).
    """
    drift_results = {}
    for ckpt_name, cka_values in cka_cross_temporal.items():
        if ckpt_name == "base":
            continue
        drift_results[ckpt_name] = 1.0 - cka_values

    return drift_results


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 7 — Salvataggio e caricamento risultati
# ─────────────────────────────────────────────────────────────────────────────

def save_cka_results(
    cka_matrix: np.ndarray,
    output_dir: Path,
    filename_stem: str = "cka_matrix",
) -> None:
    """
    Salva la matrice CKA sia in formato .npy (per uso programmatico) che
    in .csv (per leggibilità e ispezione rapida).

    Output (cfr. Pipeline.md Fase 5, Step 4):
        output_dir/cka_matrix.npy   → array numpy (L, L)
        output_dir/cka_matrix.csv   → tabella leggibile con header

    Args:
        cka_matrix   (np.ndarray): Matrice CKA di forma (L, L).
        output_dir   (Path):       Directory di destinazione (creata se assente).
        filename_stem (str):       Prefisso del nome file (default: "cka_matrix").
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salvataggio .npy (formato binario, accesso rapido)
    npy_path = output_dir / f"{filename_stem}.npy"
    np.save(npy_path, cka_matrix)
    print(f"Salvato: {npy_path}")

    # Salvataggio .csv (leggibilità: colonne = layer_00, layer_01, ...)
    csv_path = output_dir / f"{filename_stem}.csv"
    n_layers = cka_matrix.shape[0]
    header = ",".join([f"layer_{l:02d}" for l in range(n_layers)])
    np.savetxt(csv_path, cka_matrix, delimiter=",", header=header, comments="")
    print(f"Salvato: {csv_path}")


__all__ = [
    "linear_cka",
    "cka_matrix_across_layers",
    "center_gram",
    "compute_cka_matrix_intramodel",
    "compute_cka_intercategory",
    "compute_cka_intercategory_all_layers",
    "compute_cka_cross_temporal",
    "compute_cka_drift",
    "save_cka_results",
]


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO 8 — Entry point / esempio d'uso
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Esempio d'uso della pipeline CKA completa.

    Struttura attesa dei file su disco (generata dalla Fase 1 della pipeline):
        data/processed/gpt2-medium/
            layer_00.pt     → Tensor (N, 768), N stimoli, d=768
            layer_01.pt
            ...
            layer_23.pt
            metadata.json   → {"stimuli_ids": [...], "category": [...]}
    """
    import json

    # ── Configurazione ───────────────────────────────────────────────────────
    MODEL_NAME  = "gpt2-medium"       # o "phi2", "phi3-mini" per esperimenti main
    N_LAYERS    = 24                  # GPT-2 medium ha 24 layer transformer
    N_SUB       = 512                 # subsample per CKA (512 è stabile e veloce)
    SEED        = 42
    DEVICE      = "cpu"               # usa "cuda" se disponibile

    BASE_DIR    = Path("data/processed") / MODEL_NAME
    RESULTS_DIR = Path("results")

    # ── Uso 1: Matrice CKA intra-modello L×L ────────────────────────────────
    print("=" * 60)
    print("USO 1 — CKA intra-modello (matrice L×L)")
    print("=" * 60)

    cka_matrix = compute_cka_matrix_intramodel(
        hidden_states_dir=BASE_DIR,
        n_layers=N_LAYERS,
        n_sub=N_SUB,
        seed=SEED,
        device=DEVICE,
    )

    save_cka_results(
        cka_matrix=cka_matrix,
        output_dir=RESULTS_DIR,
        filename_stem="cka_matrix_intramodel",
    )

    # La diagonale deve essere 1.0 — sanity check
    diag_mean = np.diag(cka_matrix).mean()
    assert abs(diag_mean - 1.0) < 1e-6, f"Sanity check fallito: diag mean = {diag_mean}"
    print(f"Sanity check OK: diagonale media = {diag_mean:.6f}")

    # ── Uso 2: CKA inter-categoria per layer (curva RQ1) ────────────────────
    print("\n" + "=" * 60)
    print("USO 2 — CKA inter-categoria (curva per layer, RQ1)")
    print("=" * 60)

    # Carichiamo metadata per ottenere gli indici per categoria
    with open(BASE_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)

    # metadata["category"] è una lista parallela a stimuli_ids
    # con valori "arithmetic" | "algebra" | "gsm8k" | "generic"
    categories = np.array(metadata["category"])

    # Indici per ogni macro-categoria (cfr. Dataset.md)
    math_mask    = np.isin(categories, ["arithmetic", "algebra", "gsm8k"])
    generic_mask = categories == "generic"

    math_indices    = np.where(math_mask)[0]
    generic_indices = np.where(generic_mask)[0]

    print(f"  Stimoli matematici: {len(math_indices)}")
    print(f"  Stimoli generici:   {len(generic_indices)}")

    cka_inter = compute_cka_intercategory_all_layers(
        hidden_states_dir=BASE_DIR,
        n_layers=N_LAYERS,
        math_indices=math_indices,
        generic_indices=generic_indices,
        device=DEVICE,
    )

    # Salvataggio come array 1D
    np.save(RESULTS_DIR / "cka_intercategory.npy", cka_inter)
    print("\nCKA inter-categoria per layer:")
    for l, val in enumerate(cka_inter):
        print(f"  Layer {l:02d}: {val:.4f}")

    # ── Uso 3: CKA cross-temporale (curva RQ3) ───────────────────────────────
    print("\n" + "=" * 60)
    print("USO 3 — CKA cross-temporale (base vs checkpoint QLoRA, RQ3)")
    print("=" * 60)

    # Checkpoint generati durante il fine-tuning QLoRA (cfr. Pipeline.md Fase 9)
    # Ogni directory contiene i file layer_XX.pt estratti su quel checkpoint
    CKPT_BASE = Path("data/processed/checkpoints")
    checkpoint_dirs = {
        "ckpt_500":  CKPT_BASE / "ckpt_500",
        "ckpt_1000": CKPT_BASE / "ckpt_1000",
        "ckpt_1500": CKPT_BASE / "ckpt_1500",
        "ckpt_2000": CKPT_BASE / "ckpt_2000",
    }

    # Filtra solo i checkpoint che esistono su disco
    checkpoint_dirs = {k: v for k, v in checkpoint_dirs.items() if v.exists()}

    if checkpoint_dirs:
        cka_temporal = compute_cka_cross_temporal(
            base_hidden_states_dir=BASE_DIR,
            checkpoint_hidden_states_dirs=checkpoint_dirs,
            n_layers=N_LAYERS,
            n_sub=N_SUB,
            seed=SEED,
            device=DEVICE,
        )

        cka_drift = compute_cka_drift(cka_temporal)

        # Salvataggio: una riga per checkpoint, una colonna per layer
        drift_matrix = np.stack(list(cka_drift.values()))  # (n_ckpt, n_layers)
        np.save(RESULTS_DIR / "cka_drift_temporal.npy", drift_matrix)

        print("\nCKA drift (per checkpoint, layer con drift massimo):")
        for ckpt_name, drift in cka_drift.items():
            l_max = np.argmax(drift)
            print(f"  {ckpt_name}: drift massimo al layer {l_max:02d} "
                  f"(drift = {drift[l_max]:.4f})")
    else:
        print("  Nessun checkpoint trovato su disco — skip Uso 3.")
        print("  Avvia prima il fine-tuning QLoRA (Pipeline.md, Fase 9).")
