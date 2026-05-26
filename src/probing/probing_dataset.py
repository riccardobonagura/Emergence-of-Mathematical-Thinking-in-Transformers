# probing_dataset.py — data preparation for linear probing.
#
# Expected prop_cfg schema (v5):
#   {
#       "label_field": "sign" | "parity",
#       "category":    "CAT-SIGN" | "CAT-PARITY" | None
#   }
#
# The "category" key is essential in v5.  Without it the sign probe is silently
# contaminated by CAT-PARITY stimuli (which have sign=0 for all 1 000 samples),
# producing a 3:1 class imbalance that the sentinel check `!= -1` cannot catch:
#
#   sign probe, no filter  →  1 500 sign=0 / 500 sign=1  (3:1)
#   sign probe, CAT-SIGN   →  500 sign=0  / 500 sign=1   (1:1)  ✓
#
# Passing category=None falls back to the sentinel-free global filter (useful
# for custom probes or future categories).

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Optional

from sklearn.model_selection import train_test_split

from .seeds import get_seed

log = logging.getLogger(__name__)


class PropConfig(TypedDict, total=False):
    """Schema for a single probe property configuration.

    Required fields:
        label_field: name of the label in stimulus["labels"] (e.g. "sign", "parity").
        category:    dataset category to filter on. Use None to disable filtering
                     (risks cross-category contamination — see module docstring).

    Optional fields:
        type:        "binary" (default) or "multiclass". Used by run_rq2/rq3 to
                     determine inference method. In v5 all probes are binary.
    """
    label_field: str
    category:    str | None    # required but typed as possibly None
    type:        Literal["binary", "multiclass"]


class ProbingDataset:
    """Bridges JSONL stimuli and pre-extracted tensor indices for a probing run."""

    def __init__(self, stimuli_path: Path, stimuli_ids: List[str], cfg: Optional[Dict[str, Any]] = None) -> None:
        self.stimuli_path = stimuli_path
        # id → row index in the layer tensor (established at extraction time)
        self.id_to_idx: Dict[str, int] = {sid: i for i, sid in enumerate(stimuli_ids)}
        
        # FIX: threshold dinamico per i test, default 10 per la produzione
        self._min_class_samples = cfg.get("min_class_samples", 10) if cfg else 10
        
        self._df = self._load()

    # ── public API ────────────────────────────────────────────────────────────

    def get_property_split(
        self,
        prop_name: str,
        prop_cfg:  Any,  # Sostituisci con PropConfig se è importato esplicitamente
        train_split: float,
        global_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full preparation pipeline: filter → balance → stratified split.

        Returns (X_train_idx, X_test_idx, y_train, y_test) where indices
        reference rows in the per-layer .pt tensors.
        """
        indices, labels = self._extract(prop_name, prop_cfg)
        
        # Sostituito threshold=10 con il parametro di istanza
        self._check_min_class(labels, prop_name, threshold=self._min_class_samples)
        
        indices, labels = self._undersample(indices, labels, prop_name, global_seed)
        return self._split(indices, labels, train_split, prop_name, global_seed)

    # ── private helpers ───────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        # Load JSONL once; labels column stays as dict for .get() access.
        records = [json.loads(l) for l in open(self.stimuli_path, encoding="utf-8")]
        return pd.DataFrame(records)

    def _extract(
        self,
        prop_name: str,
        prop_cfg:  PropConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter by category (when specified) then align ids to tensor indices."""
        label_field  = prop_cfg["label_field"]
        target_cat   = prop_cfg.get("category")   # None → use all rows

        # Category pre-filter prevents cross-category label contamination.
        df = (self._df[self._df["category"] == target_cat].copy()
              if target_cat is not None else self._df)

        valid_idx, labels, unmatched = [], [], []

        for _, row in df.iterrows():
            val = row["labels"].get(label_field)
            if val is None:
                continue
            sid = row["id"]
            if sid in self.id_to_idx:
                valid_idx.append(self.id_to_idx[sid])
                labels.append(val)
            else:
                unmatched.append(sid)

        if not valid_idx:
            meta_ex  = next(iter(self.id_to_idx), "NONE")
            jsonl_ex = unmatched[0] if unmatched else "NONE"
            raise ValueError(
                f"Alignment error for '{prop_name}': "
                f"no JSONL id found in metadata.\n"
                f"  JSONL example : {jsonl_ex!r}\n"
                f"  Metadata ex.  : {meta_ex!r}"
            )

        if unmatched:
            log.warning("%d ids in JSONL not found in metadata (prop=%s).",
                        len(unmatched), prop_name)

        return np.array(valid_idx), np.array(labels)

    @staticmethod
    def _check_min_class(labels: np.ndarray, prop_name: str, threshold: int) -> None:
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() < threshold:
            raise ValueError(
                f"Minority class for '{prop_name}' has only {counts.min()} samples "
                f"(threshold={threshold}). Check category filter and dataset size."
            )

    def _undersample(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        prop_name: str,
        global_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes to the minority size; deterministic via seeded RNG."""
        # For v5 the dataset guarantees 50/50 balance, so this is typically a no-op.
        # Kept as a safety net if any ids are missing from the metadata.
        min_count = int(np.unique(labels, return_counts=True)[1].min())
        rng = np.random.default_rng(
            get_seed(global_seed, "undersampling", hash(prop_name) % 10_000)
        )
        bal_idx, bal_lbl = [], [] # type: ignore
        for cls in np.unique(labels):
            pool = indices[labels == cls]
            chosen = rng.choice(pool, size=min_count, replace=False)
            bal_idx.extend(chosen)
            bal_lbl.extend([cls] * min_count)
        return np.array(bal_idx), np.array(bal_lbl)

    def _split(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        train_split: float,
        prop_name: str,
        global_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train/test split; seed is prop-name-specific for isolation."""
        return train_test_split(
            indices, labels,
            train_size=train_split,
            stratify=labels,
            random_state=get_seed(
                global_seed, "train_test_split", hash(prop_name) % 10_000
            ),
        )