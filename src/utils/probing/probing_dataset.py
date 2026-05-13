"""
Modulo per la gestione dei dati di probing.
Responsabile: Caricamento, Allineamento ID, Undersampling e Split Stratificato.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split

from .seeds import get_seed

class ProbingDataset:
    """
    Gestisce l'interfaccia tra gli stimoli (JSONL) e gli indici dei tensori (Metadata).
    """

    def __init__(self, stimuli_path: Path, stimuli_ids: List[str]):
        """
        Args:
            stimuli_path: Path al file .jsonl con le label.
            stimuli_ids: Lista ordinata di ID proveniente dai metadati dei tensori.
        """
        self.stimuli_path = stimuli_path
        # Creiamo la mappa ID -> Indice Riga nel Tensore (fondamentale per l'allineamento)
        self.id_to_idx = {sid: i for i, sid in enumerate(stimuli_ids)}
        self.raw_df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Carica il JSONL in un DataFrame per manipolazione semplificata."""
        records = []
        with open(self.stimuli_path, "r") as f:
            for line in f:
                records.append(json.loads(line))
        return pd.DataFrame(records)

    def get_property_split(
        self, 
        prop_name: str, 
        prop_cfg: Dict[str, Any], 
        train_split: float, 
        global_seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Esegue la pipeline di preparazione per una specifica proprietà.
        """
        # 1. Estrazione e Filtraggio
        indices, labels = self._extract_valid_samples(prop_name, prop_cfg)
        
        # 2. Verifica Massa Critica
        unique, counts = np.unique(labels, return_counts=True)
        min_count = counts.min()
        if min_count < 10:
            raise ValueError(f"Classe minoritaria per '{prop_name}' troppo piccola: {min_count}")

        # 3. Undersampling Deterministico (Bilanciamento)
        indices, labels = self._apply_undersampling(
            indices, labels, min_count, prop_name, global_seed
        )

        # 4. Split Stratificato
        return self._split_data(indices, labels, train_split, prop_name, global_seed)

    def _extract_valid_samples(self, prop_name: str, prop_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Filtra gli stimoli e li mappa agli indici dei tensori."""
        valid_indices = []
        labels = []
        label_field = prop_cfg["label_field"]
        
        match_count = 0
        id_mismatch_example = None

        for _, row in self.raw_df.iterrows():
            val = row["labels"].get(label_field, -1)
            
            if val != -1 and val is not None:
                # GESTIONE ALLINEAMENTO ID
                if row["id"] in self.id_to_idx:
                    valid_indices.append(self.id_to_idx[row["id"]])
                    labels.append(val)
                    match_count += 1
                else:
                    if id_mismatch_example is None:
                        id_mismatch_example = row["id"]

        # --- CONTROLLO SOLID: Fail Fast con messaggio informativo ---
        if match_count == 0:
            example_meta = list(self.id_to_idx.keys())[0] if self.id_to_idx else "NESSUNO"
            raise ValueError(
                f"ERRORE DI ALLINEAMENTO: Nessun ID in stimuli.jsonl corrisponde ai metadati.\n"
                f"Esempio ID in JSONL: '{id_mismatch_example}'\n"
                f"Esempio ID in Metadata: '{example_meta}'"
            )

        return np.array(valid_indices), np.array(labels)

    def _apply_undersampling(self, indices, labels, min_count, prop_name, seed) -> Tuple[np.ndarray, np.ndarray]:
        """Bilancia le classi prendendo N campioni (N = dimensione classe minoritaria)."""
        rng = np.random.default_rng(get_seed(seed, "undersampling", hash(prop_name) % 10000))
        
        balanced_indices = []
        balanced_labels = []
        unique_classes = np.unique(labels)

        for cls in unique_classes:
            cls_mask = (labels == cls)
            cls_indices = indices[cls_mask]
            # Selezione casuale ma deterministica
            sampled = rng.choice(cls_indices, size=min_count, replace=False)
            balanced_indices.extend(sampled)
            balanced_labels.extend([cls] * min_count)

        return np.array(balanced_indices), np.array(balanced_labels)

    def _split_data(self, indices, labels, train_split, prop_name, seed):
        """Esegue lo split finale train/test."""
        return train_test_split(
            indices, labels, 
            train_size=train_split, 
            stratify=labels,
            random_state=get_seed(seed, "train_test_split", hash(prop_name) % 10000)
        )