# src/probing/io_utils.py
"""
io_utils.py — Hardened I/O, metadata parsing, tensor loading, and atomic file writes.
Enforces strict UTF-8 compliance, PyTorch 2.x/2.4+ warnings prevention, type consistency,
and atomic durability patterns to prevent file corruption upon unexpected execution kills.
"""

import csv
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


# ── SECTION 1 — METADATA PARSING & VALIDATION ─────────────────────────────────

class MetadataHandler:
    """Reads and validates the metadata.json produced by extract_states.py."""

    def __init__(self, metadata_path: Path) -> None:
        self.path = metadata_path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.path}")
        # FIX IO-01: Explicit encoding prevents crash on non-UTF-8 host setups (e.g. native WSL2 environments)
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def get_n_layers(self) -> int:
        """Return n_layers from metadata; falls back to counting .pt files on disk."""
        if "n_layers" in self.data:
            return int(self.data["n_layers"])
        pt_files = list(self.path.parent.glob("layer_*.pt"))
        if not pt_files:
            raise ValueError(f"Cannot determine n_layers from {self.path.parent}")
        return len(pt_files)

    def get_d_model(self, default: int = 2048) -> int:
        """Return d_model from metadata layer metrics (defaults to Pythia-1.4B footprint)."""
        return int(self.data.get("d_model", default))

    def get_stimuli_ids(self) -> List[str]:
        """Retrieve the ordered list of unique stimulus identifiers."""
        return self.data.get("stimuli_ids", [])

    def get_n_stimuli(self) -> int:
        """Return total count of compiled stimuli tokens."""
        return int(self.data.get("n_stimuli", len(self.get_stimuli_ids())))

    def get_labels(self, field: str) -> np.ndarray:
        """
        Extract targets block from metadata.
        FIX IO-05: Enforces np.int64 to maintain strict compatibility with NumPy index
        arrays and scikit-learn interfaces, eliminating silent truncation and casting warnings.
        """
        labels_block = self.data.get("labels", {})
        if field not in labels_block:
            raise KeyError(f"Label field '{field}' missing from metadata labels block.")
        return np.array(labels_block[field], dtype=np.int64)


# ── SECTION 2 — LOGGING & TENSOR LOADING ──────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configures centralized console and file-based logging contexts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("probing")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(output_dir / "probing.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_hidden_states(layer_path: Path) -> np.ndarray:
    """
    Loads pre-extracted activation arrays safely into memory.
    FIX IO-02: Explicitly passes weights_only=True to silence PyTorch 2.x deprecation
    warnings and secure the unpickling loop against future structural breaking changes.
    """
    if not layer_path.exists():
        raise FileNotFoundError(f"Hidden states tensor missing: {layer_path}")
    return torch.load(layer_path, map_location="cpu", weights_only=True).float().numpy()


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Decodes standalone dictionary structures from extraction metadata paths."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file missing: {metadata_path}")
    # FIX IO-01: Explicit encoding forced for stability across unaligned OS defaults
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


# ── SECTION 3 — ATOMIC FILE WRITERS & PERSISTENCE HELPERS ─────────────────────

def _atomic_write_csv(output_path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    """Atomically commits structured log metadata records via OS replacement swaps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, output_path)
    except Exception:
        os.remove(tmp)
        raise


def _atomic_write_json(output_path: Path, data: Dict) -> None:
    """Atomically dumps metrics payloads to mitigate corruption upon unexpected termination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".json")
    try:
        # FIX IO-01: Enforces explicit encoding contract inside low-level fd wrappers
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, output_path)
    except Exception:
        os.remove(tmp)
        raise


def _atomic_save_npy(output_path: Path, arr: np.ndarray) -> None:
    """
    FIX IO-04: Helper utility to handle binary NumPy serialization atomically.
    Allocates an isolated file descriptor and commits a complete array buffer before
    executing an atomic metadata override swap via os.replace.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, output_path)
    except Exception:
        os.remove(tmp)
        raise


def save_test_indices(output_dir: Path, prop_name: str, test_indices: np.ndarray) -> None:
    """Persist train/test split partitions to guarantee frozen spatial isolation evaluations."""
    d = output_dir / "test_indices"
    # Unified with the new atomic saving layer to guarantee data-split persistence safety
    _atomic_save_npy(d / f"{prop_name}_test_idx.npy", test_indices)


def load_test_indices(output_dir: Path, prop_name: str) -> np.ndarray:
    """
    Retrieves the frozen validation splits pre-allocated before linear probing.
    FIX IO-03: Integrates a fail-fast assertion checkpoint to enforce rule E-P-03
    from the epistemological standards manual, providing contextual recovery guidelines.
    """
    path = output_dir / "test_indices" / f"{prop_name}_test_idx.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Test indices for property '{prop_name}' not found at expected path: {path}. "
            f"Violation of Principle E-P-03: indices must be saved and frozen before training. "
            f"Please run run_rq2.py first to establish the baseline splits."
        )
    return np.load(path)


def save_weights(
    output_dir: Path,
    layer_idx: int,
    prop_name: str,
    w_orig: np.ndarray,
    b_orig: np.ndarray,
) -> None:
    """
    Persists hyperplane projection coefficients back into original unscaled activation spaces.
    FIX IO-04: Encapsulates weight arrays inside the atomic save loop to prevent truncated
    null files if a cluster or workstation process kill signal interrupts execution mid-epoch.
    """
    d = output_dir / "weights"
    _atomic_save_npy(d / f"layer_{layer_idx:02d}_{prop_name}.npy", w_orig)
    _atomic_save_npy(d / f"layer_{layer_idx:02d}_{prop_name}_bias.npy", b_orig)
