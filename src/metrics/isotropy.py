"""
Utility functions per la geometria degli spazi di rappresentazione.

Contenuti previsti (Fase 3 - Isotropia):
- Similarità coseno tra vettori di embedding
- Stima dell'isotropia tramite campionamento di coppie casuali
  E_{i != j}[cos(theta_ij)]

Nota: qui definiamo solo le firme delle funzioni e i commenti di
documentazione. Le implementazioni vere e proprie vanno aggiunte in
una fase successiva.
"""

"""
isotropy.py
===========
Fase 3 della pipeline: Analisi di Isotropia (Media Similarità Coseno).

Input:
    - data/processed/{model_name}/layer_XX.pt  : tensore (N, d) per ogni layer
    - data/processed/{model_name}/metadata.json : lista ordinata degli ID stimolo
    - data/stimuli/stimuli.jsonl                : dataset con campo "category"

Output:
    - results/isotropy.csv
      Colonne: layer, category, n_stimuli, estimator, iso_mean, iso_spread,
               ci_low, ci_high
      - iso_mean    : media esatta o stimata delle similarità coseno
      - iso_spread  : std delle similarità (spread della distribuzione)
      - ci_low/high : CI al 95% (NaN per estimatore esatto, bootstrap per MC)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
import warnings
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Configurazione logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Struttura risultato per singola (layer, categoria)
# ---------------------------------------------------------------------------
class IsotropyResult(NamedTuple):
    layer: int
    category: str
    n_stimuli: int
    estimator: str          # "exact" | "monte_carlo"
    iso_mean: float
    iso_spread: float       # std delle similarità coseno
    ci_low: float           # NaN se estimatore esatto
    ci_high: float          # NaN se estimatore esatto


# ---------------------------------------------------------------------------
# Seed globale — fissato una sola volta all'inizio, mai dentro le funzioni
# ---------------------------------------------------------------------------
def make_rng(seed: int) -> np.random.Generator:
    """Crea un generatore NumPy con seed esplicito da passare alle funzioni."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Estimatore esatto (gram matrix completa)
# ---------------------------------------------------------------------------
def isotropy_exact(H_cat: torch.Tensor) -> tuple[float, float]:
    """
    Calcola la media esatta e la std delle similarità coseno su tutte le coppie
    (i, j) con i ≠ j tramite la matrice di Gram normalizzata.

    Questa è la media della distribuzione discreta uniforme su tutte le
    N*(N-1) coppie ordinate — nessuna varianza campionaria.

    Args:
        H_cat: Tensore (N_cat, d) degli hidden state per una categoria.

    Returns:
        (iso_mean, iso_spread): media e std delle N*(N-1) similarità coseno.
    """
    N = H_cat.shape[0]
    assert N >= 2, "Servono almeno 2 stimoli per calcolare la similarità coseno."

    # Verifica vettori a norma nulla
    norms = H_cat.norm(dim=1)
    zero_mask = norms < 1e-8
    if zero_mask.any():
        n_zero = zero_mask.sum().item()
        warnings.warn(
            f"{n_zero} vettori hanno norma < 1e-8 e verranno esclusi. "
            "Controllare l'estrazione degli hidden state."
        )
        H_cat = H_cat[~zero_mask]
        N = H_cat.shape[0]
        if N < 2:
            return float("nan"), float("nan")

    # Normalizzazione L2
    H_norm = F.normalize(H_cat, p=2, dim=1)  # (N, d)

    # Matrice di Gram: C[i,j] = cos(h_i, h_j)
    C = H_norm @ H_norm.T  # (N, N)

    # Estrai solo gli elementi fuori diagonale
    mask = ~torch.eye(N, dtype=torch.bool, device=C.device)
    off_diag = C[mask]  # N*(N-1) valori

    iso_mean = float(off_diag.mean().item())
    iso_spread = float(off_diag.std().item())
    return iso_mean, iso_spread


# ---------------------------------------------------------------------------
# Estimatore Monte Carlo (per N_cat grande)
# ---------------------------------------------------------------------------
def isotropy_monte_carlo(
    H_cat: torch.Tensor,
    k_pairs: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """
    Stima la media della similarità coseno campionando k_pairs coppie casuali.
    Restituisce media, std delle similarità, e CI al 95% via bootstrap percentile.

    Il campionamento usa lo "shift trick" che garantisce distribuzione uniforme
    su tutte le coppie ordinate (i, j) con i ≠ j:
        i ~ Uniform{0,...,N-1}
        j' ~ Uniform{0,...,N-2}
        j  = j' + (j' >= i)
    Per ogni k ≠ i esiste un unico j' che mappa su k, quindi p(j=k|i) = 1/(N-1).

    Args:
        H_cat:       Tensore (N_cat, d) degli hidden state.
        k_pairs:     Numero di coppie da campionare.
        n_bootstrap: Numero di ricampionamenti bootstrap per il CI.
        rng:         Generatore NumPy con seed già fissato esternamente.

    Returns:
        (iso_mean, iso_spread, ci_low, ci_high)
    """
    N = H_cat.shape[0]
    assert N >= 2

    # Avviso se K >> coppie uniche disponibili
    max_unique_pairs = N * (N - 1)
    if k_pairs > max_unique_pairs:
        warnings.warn(
            f"k_pairs={k_pairs} > coppie uniche disponibili={max_unique_pairs} "
            f"(N={N}). Il campionamento avviene con reinserimento implicito. "
            "Considera di ridurre k_pairs o usare l'estimatore esatto."
        )

    # Verifica norma nulla
    norms = H_cat.norm(dim=1)
    zero_mask = norms < 1e-8
    if zero_mask.any():
        warnings.warn(f"{zero_mask.sum().item()} vettori a norma nulla esclusi.")
        H_cat = H_cat[~zero_mask]
        N = H_cat.shape[0]

    H_norm = F.normalize(H_cat, p=2, dim=1)  # (N, d)

    # Campionamento shift-trick con rng esterno (stato condiviso tra chiamate)
    idx_i = rng.integers(0, N, size=k_pairs)
    idx_j_raw = rng.integers(0, N - 1, size=k_pairs)
    idx_j = np.where(idx_j_raw >= idx_i, idx_j_raw + 1, idx_j_raw)

    # Similarità coseno vettoriale
    u = H_norm[torch.from_numpy(idx_i)]   # (K, d)
    v = H_norm[torch.from_numpy(idx_j)]   # (K, d)
    sims = (u * v).sum(dim=1).numpy()     # (K,)

    iso_mean = float(sims.mean())
    iso_spread = float(sims.std())

    # Bootstrap percentile CI al 95%
    boot_means = np.fromiter(
        (rng.choice(sims, size=k_pairs, replace=True).mean()
         for _ in range(n_bootstrap)),
        dtype=float,
        count=n_bootstrap,
    )
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return iso_mean, iso_spread, ci_low, ci_high


# ---------------------------------------------------------------------------
# Validazione allineamento tensore / metadati
# ---------------------------------------------------------------------------
def validate_alignment(
    H_l: torch.Tensor,
    stimuli_ids: list[str],
    stimuli_jsonl_path: Path,
) -> None:
    """
    Verifica che il numero di righe del tensore corrisponda al numero di ID
    in metadata.json e che tutti gli ID esistano nel dataset.

    Lancia ValueError se l'allineamento è violato.
    """
    N_tensor = H_l.shape[0]
    N_meta = len(stimuli_ids)
    if N_tensor != N_meta:
        raise ValueError(
            f"Disallineamento critico: il tensore ha {N_tensor} righe "
            f"ma metadata.json contiene {N_meta} ID. "
            "Riestrarre gli hidden state."
        )

    # Verifica che tutti gli ID siano presenti nel dataset
    ids_in_jsonl = set()
    with open(stimuli_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ids_in_jsonl.add(json.loads(line)["id"])

    missing = set(stimuli_ids) - ids_in_jsonl
    if missing:
        raise ValueError(
            f"{len(missing)} ID in metadata.json non trovati in stimuli.jsonl: "
            f"{list(missing)[:5]} ..."
        )


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def run_isotropy_analysis(
    processed_dir: str = "data/processed/phi3-mini",
    stimuli_path: str = "data/stimuli/stimuli.jsonl",
    output_path: str = "results/isotropy.csv",
    n_layers: int = 32,
    exact_threshold: int = 600,   # usa gram matrix esatta se N_cat <= soglia
    k_pairs: int = 8000,          # coppie Monte Carlo (solo se N_cat > soglia)
    n_bootstrap: int = 2000,      # campionamenti bootstrap
    seed: int = 42,
) -> pd.DataFrame:
    """
    Esegue l'analisi di isotropia su tutti i layer del modello.

    Strategia adattiva:
        N_cat <= exact_threshold  → estimatore esatto (gram matrix)
        N_cat >  exact_threshold  → Monte Carlo + bootstrap CI

    Args:
        processed_dir:    Directory con i file layer_XX.pt e metadata.json.
        stimuli_path:     Path al file JSONL del dataset.
        output_path:      Path del CSV di output.
        n_layers:         Numero totale di layer del modello.
        exact_threshold:  Soglia per la scelta dell'estimatore.
        k_pairs:          Coppie da campionare (regime Monte Carlo).
        n_bootstrap:      Ricampionamenti bootstrap (regime Monte Carlo).
        seed:             Seed globale per riproducibilità.

    Returns:
        DataFrame con i risultati.
    """
    # RNG unico, creato una sola volta — lo stato avanza in modo deterministico
    # attraverso tutti i layer e le categorie, senza reset interni
    rng = make_rng(seed)

    proc_path = Path(processed_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    stimuli_jsonl = Path(stimuli_path)

    # Caricamento metadati
    with open(proc_path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    stimuli_ids: list[str] = metadata["stimuli_ids"]

    # Mappa id → categoria
    id_to_cat: dict[str, str] = {}
    with open(stimuli_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            id_to_cat[s["id"]] = s["category"]

    categories = sorted(set(id_to_cat.values()))

    # Indici di riga per categoria — costruiti una sola volta
    cat_to_indices: dict[str, list[int]] = {c: [] for c in categories}
    for row_idx, sid in enumerate(stimuli_ids):
        if sid not in id_to_cat:
            raise ValueError(
                f"ID '{sid}' in metadata.json non trovato in stimuli.jsonl."
            )
        cat_to_indices[id_to_cat[sid]].append(row_idx)

    log.info("Categorie trovate: %s", categories)
    for cat, idxs in cat_to_indices.items():
        log.info("  %-20s  N=%d", cat, len(idxs))

    results: list[IsotropyResult] = []

    for l in range(n_layers):
        layer_file = proc_path / f"layer_{l:02d}.pt"
        if not layer_file.exists():
            log.warning("Layer %02d: file non trovato, skippato.", l)
            continue

        H_l: torch.Tensor = torch.load(layer_file, map_location="cpu")

        # Validazione allineamento (solo al primo layer per efficienza,
        # poi ci fidiamo della pipeline di estrazione)
        if l == 0:
            validate_alignment(H_l, stimuli_ids, stimuli_jsonl)

        for cat in categories:
            indices = cat_to_indices[cat]
            if len(indices) < 2:
                log.warning("Layer %02d  %-20s: N=%d < 2, skippata.", l, cat, len(indices))
                continue

            H_cat = H_l[indices]  # (N_cat, d)
            N_cat = len(indices)

            if N_cat <= exact_threshold:
                iso_mean, iso_spread = isotropy_exact(H_cat)
                results.append(IsotropyResult(
                    layer=l,
                    category=cat,
                    n_stimuli=N_cat,
                    estimator="exact",
                    iso_mean=iso_mean,
                    iso_spread=iso_spread,
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                ))
            else:
                iso_mean, iso_spread, ci_low, ci_high = isotropy_monte_carlo(
                    H_cat, k_pairs=k_pairs, n_bootstrap=n_bootstrap, rng=rng
                )
                results.append(IsotropyResult(
                    layer=l,
                    category=cat,
                    n_stimuli=N_cat,
                    estimator="monte_carlo",
                    iso_mean=iso_mean,
                    iso_spread=iso_spread,
                    ci_low=ci_low,
                    ci_high=ci_high,
                ))

        del H_l  # libera memoria esplicitamente

        if (l + 1) % 8 == 0:
            log.info("Layer %02d/%02d completato.", l + 1, n_layers)

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    log.info("Risultati salvati in: %s", out_file)
    return df


# ---------------------------------------------------------------------------
# API pubblica unificata (livello modulo)
# ---------------------------------------------------------------------------
def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Restituisce la matrice NxN delle similarità coseno tra tutti i vettori.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Atteso tensore 2D [n, d], ricevuto shape={tuple(embeddings.shape)}")

    norms = embeddings.norm(dim=1, keepdim=True)
    valid = (norms.squeeze(-1) >= 1e-8)
    if valid.sum().item() < 2:
        raise ValueError("Servono almeno 2 vettori con norma non nulla.")

    E = embeddings[valid]
    E = F.normalize(E, p=2, dim=1)
    return E @ E.T


def sample_random_cosine_pairs(
    embeddings: torch.Tensor,
    num_pairs: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Campiona `num_pairs` coppie ordinate (i, j), i != j, e ritorna cos(theta_ij).
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs deve essere > 0")
    if embeddings.ndim != 2:
        raise ValueError(f"Atteso tensore 2D [n, d], ricevuto shape={tuple(embeddings.shape)}")

    norms = embeddings.norm(dim=1)
    valid = norms >= 1e-8
    E = embeddings[valid]
    n = E.shape[0]
    if n < 2:
        raise ValueError("Servono almeno 2 vettori con norma non nulla.")

    E = F.normalize(E, p=2, dim=1)
    idx_i = torch.randint(0, n, (num_pairs,), generator=generator)
    idx_j_raw = torch.randint(0, n - 1, (num_pairs,), generator=generator)
    idx_j = idx_j_raw + (idx_j_raw >= idx_i).to(idx_j_raw.dtype)

    u = E[idx_i]
    v = E[idx_j]
    return (u * v).sum(dim=1)


def estimate_isotropy(
    embeddings: torch.Tensor,
    *,
    method: str = "auto",
    num_pairs: int = 8000,
    exact_threshold: int = 600,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict:
    """
    API unificata per isotropia.
    """
    if method not in {"auto", "exact", "sampled"}:
        raise ValueError("method deve essere uno tra: auto, exact, sampled")
    if embeddings.ndim != 2:
        raise ValueError(f"Atteso tensore 2D [n, d], ricevuto shape={tuple(embeddings.shape)}")

    n = int(embeddings.shape[0])
    estimator = "exact" if (method == "exact" or (method == "auto" and n <= exact_threshold)) else "sampled"

    if estimator == "exact":
        iso_mean, iso_spread = isotropy_exact(embeddings)
        return {
            "estimator": "exact",
            "iso_mean": float(iso_mean),
            "iso_spread": float(iso_spread),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_samples": n,
        }

    rng = make_rng(seed)
    if n_bootstrap > 0:
        iso_mean, iso_spread, ci_low, ci_high = isotropy_monte_carlo(
            embeddings,
            k_pairs=num_pairs,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
    else:
        sims = sample_random_cosine_pairs(
            embeddings=embeddings,
            num_pairs=num_pairs,
            generator=torch.Generator().manual_seed(seed),
        )
        iso_mean = float(sims.mean().item())
        iso_spread = float(sims.std().item())
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "estimator": "sampled",
        "iso_mean": float(iso_mean),
        "iso_spread": float(iso_spread),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_samples": n,
    }


__all__ = [
    "cosine_similarity_matrix",
    "sample_random_cosine_pairs",
    "estimate_isotropy",
    "isotropy_exact",
    "isotropy_monte_carlo",
    "run_isotropy_analysis",
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_isotropy_analysis()
