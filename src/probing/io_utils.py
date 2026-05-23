# io_utils.py — I/O, metadata parsing, tensor loading, and atomic file writes.

import csv
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ── Metadata ──────────────────────────────────────────────────────────────────

class MetadataHandler:
    """Reads and validates the metadata.json produced by extract_states.py."""

    def __init__(self, metadata_path: Path) -> None:
        self.path = metadata_path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.path}")
        with open(self.path) as f:
            return json.load(f)

    def get_n_layers(self) -> int:
        """Return n_layers from metadata; falls back to counting .pt files."""
        if "n_layers" in self.data:
            return int(self.data["n_layers"])
        pt_files = list(self.path.parent.glob("layer_*.pt"))
        if not pt_files:
            raise ValueError(f"Cannot determine n_layers from {self.path.parent}")
        return len(pt_files)

    def get_d_model(self, default: int = 2048) -> int:
        # 2048 is correct for Pythia-1.4B; override via metadata if needed.
        return int(self.data.get("d_model", default))

    def get_stimuli_ids(self) -> List[str]:
        # "stimuli_ids" is guaranteed by extract_states.save_extraction_metadata().
        ids = self.data.get("stimuli_ids", [])
        if not ids:
            raise ValueError("metadata.json contains no stimuli_ids.")
        return ids

    def get_n_stimuli(self) -> int:
        """Return total number of stimuli; derived from stimuli_ids if key absent."""
        if "n_stimuli" in self.data:
            return int(self.data["n_stimuli"])
        return len(self.get_stimuli_ids())

    def get_labels(self, field: str) -> np.ndarray:
        """Return label array parallel to stimuli_ids for the given field.

        Requires enriched metadata (extract_states.py Intervento 2).
        Raises KeyError when metadata was produced before the enrichment.
        """
        labels_block = self.data.get("labels")
        if labels_block is None:
            raise KeyError(
                "metadata.json does not contain a 'labels' block. "
                "Re-run extract_states.py to produce enriched metadata."
            )
        if field not in labels_block:
            raise KeyError(f"Label field {field!r} not found in metadata labels block.")
        return np.array(labels_block[field], dtype=np.int32)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure stderr + file handler for the probing logger."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("probing")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        sh  = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh  = logging.FileHandler(output_dir / "probing.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Tensor loading ────────────────────────────────────────────────────────────

def load_hidden_states(layer_path: Path) -> np.ndarray:
    """Load a layer_XX.pt tensor; casts FP16 → FP32 for sklearn compatibility."""
    if not layer_path.exists():
        raise FileNotFoundError(f"Tensor not found: {layer_path}")
    return torch.load(layer_path, map_location="cpu").float().numpy()


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Thin wrapper; prefer MetadataHandler for validated access."""
    with open(metadata_path) as f:
        return json.load(f)


# ── Atomic writes ─────────────────────────────────────────────────────────────

def _atomic_write_csv(output_path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    """Write CSV atomically via temp file + os.replace (safe against kill-signals)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, output_path)
    except Exception:
        os.remove(tmp)
        raise


def _atomic_write_json(output_path: Path, data: Dict) -> None:
    """Write JSON atomically via temp file + os.replace."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, output_path)
    except Exception:
        os.remove(tmp)
        raise


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_test_indices(output_dir: Path, prop_name: str, test_indices: np.ndarray) -> None:
    """Persist test split indices to prevent data leakage across evaluation contexts."""
    d = output_dir / "test_indices"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{prop_name}_test_idx.npy", test_indices)


def load_test_indices(output_dir: Path, prop_name: str) -> np.ndarray:
    return np.load(output_dir / "test_indices" / f"{prop_name}_test_idx.npy")


def save_weights(
    output_dir: Path,
    layer_idx: int,
    prop_name: str,
    w_orig: np.ndarray,
    b_orig: np.ndarray,
) -> None:
    """Save denormalised probe weights for downstream direction analysis."""
    d = output_dir / "weights"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"layer_{layer_idx:02d}_{prop_name}.npy",      w_orig)
    np.save(d / f"layer_{layer_idx:02d}_{prop_name}_bias.npy", b_orig)