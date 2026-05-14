"""Operazioni di Input/Output, casting tensoriale e setup dell'ambiente."""

import os
import sys
import json
import csv
import tempfile
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

class MetadataHandler:
    """Responsabile della validazione e sanitizzazione dei metadati del modello."""
    
    def __init__(self, metadata_path: Path):
        self.path = metadata_path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Metadati mancanti: {self.path}")
        with open(self.path, "r") as f:
            return json.load(f)

    def get_n_layers(self) -> int:
        """Infers n_layers counting .pt files if key is missing (Robustness)."""
        if "n_layers" in self.data:
            return self.data["n_layers"]
        
        # Fallback: conta i file layer_XX.pt nella cartella parente
        pt_files = list(self.path.parent.glob("layer_*.pt"))
        if not pt_files:
            raise ValueError(f"Impossibile determinare n_layers in {self.path.parent}")
        return len(pt_files)

    def get_d_model(self, default: int = 2048) -> int:
        return self.data.get("d_model", default)

    def get_stimuli_ids(self) -> list:
        # Se stimuli_ids è una lista gigante (come nel tuo file), la prendiamo così com'è
        ids = self.data.get("stimuli_ids", [])
        if not ids:
            raise ValueError("Il metadata non contiene stimuli_ids.")
        return ids


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configura l'handler di logging su stderr e su file (probing.log)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("probing")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        fh = logging.FileHandler(output_dir / "probing.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def load_hidden_states(layer_path: Path) -> np.ndarray:
    """
    Carica i tensori pre-estratti gestendo la transizione critica FP16 -> FP32.
    
    Args:
        layer_path: Path al file .pt.
    Returns:
        Array NumPy in FP32 di shape (N, d).
    Raises:
        FileNotFoundError: Se il file non esiste.
    """
    if not layer_path.exists():
        raise FileNotFoundError(f"Tensore non trovato in: {layer_path}")
    
    tensor = torch.load(layer_path, map_location="cpu")
    # Cast esplicito a float32 per evitare instabilità o eccezioni in sklearn
    return tensor.float().numpy().astype(np.float32)

def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Carica metadata.json per l'estrazione dinamica di n_layers e d_model."""
    with open(metadata_path, "r") as f:
        return json.load(f)

def _atomic_write_csv(output_path: Path, rows: List[Dict], fieldnames: List[str]):
    """Esegue una scrittura atomica in CSV prevenendo corruzioni da kill-signal."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, output_path)
    except Exception as e:
        os.remove(temp_path)
        raise e

def _atomic_write_json(output_path: Path, data: Dict):
    """Esegue una scrittura atomica in JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, output_path)
    except Exception as e:
        os.remove(temp_path)
        raise e

def save_test_indices(output_dir: Path, prop_name: str, test_indices: np.ndarray):
    """Persiste lo split per impedire data-leakage nel Contesto B."""
    out_dir = output_dir / "test_indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prop_name}_test_idx.npy", test_indices)

def load_test_indices(output_dir: Path, prop_name: str) -> np.ndarray:
    return np.load(output_dir / "test_indices" / f"{prop_name}_test_idx.npy")

def save_weights(output_dir: Path, layer_idx: int, prop_name: str, w_orig: np.ndarray, b_orig: np.ndarray):
    """Salva i pesi denormalizzati."""
    out_dir = output_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"layer_{layer_idx:02d}_{prop_name}.npy", w_orig)
    np.save(out_dir / f"layer_{layer_idx:02d}_{prop_name}_bias.npy", b_orig)