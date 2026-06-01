"""
probing_dataset.py — Data loading, splitting, and category filtering.
Enforces type safety contracts and cryptographic cross-session determinism.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

from src.config.schemas import PropConfig
from src.probing.seeds import get_seed

log = logging.getLogger("probing")


class ProbingDataset:
    def __init__(self, stimuli_path: Path, stimuli_ids: List[str], cfg: dict = None) -> None:
        self.stimuli_path = stimuli_path
        self.stimuli_ids = stimuli_ids
        self.cfg = cfg if cfg is not None else {}
        self._df = self._load()

    def _load(self) -> List[dict]:
        with open(self.stimuli_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        id_to_record = {r["id"]: r for r in records}

        unmatched = set(id_to_record.keys()) - set(self.stimuli_ids)
        if unmatched:
            log.warning(f"[{self.stimuli_path.name}] {len(unmatched)} IDs in dataset not found in metadata.")

        aligned_records = []
        for sid in self.stimuli_ids:
            if sid in id_to_record:
                aligned_records.append(id_to_record[sid])
            else:
                raise ValueError(f"Pre-flight alignment defect: Metadata ID '{sid}' missing from JSONL file.")

        self._pair_ids = [
            r.get("contrast", {}).get("pair_id") for r in aligned_records
        ]

        return aligned_records

    def _extract(self, prop_name: str, prop_cfg: PropConfig) -> Tuple[List[int], List[int]]:
        label_field = prop_cfg["label_field"]
        target_category = prop_cfg.get("category")

        indices = []
        labels = []

        for idx, r in enumerate(self._df):
            if target_category and r.get("category") != target_category:
                continue
            lbl_val = r.get("labels", {}).get(label_field, -1)
            if lbl_val != -1:
                indices.append(idx)
                labels.append(int(lbl_val))

        return indices, labels

    def _check_min_class(self, labels: np.ndarray, prop_name: str) -> None:
        """Ensures that the extracted target space contains sufficient samples for at least two classes."""
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2:
            raise ValueError(
                f"Property '{prop_name}' lacks separate class boundaries to compute splits. "
                f"Found active classes: {classes.tolist()}."
            )

        # BUG FIX: Restored threshold validation check to guarantee dataset split sufficiency
        min_class_samples = self.cfg.get("min_class_samples", 10)
        if counts.min() < min_class_samples:
            raise ValueError(
                f"Property '{prop_name}' has insufficient samples in the smallest class matrix partition. "
                f"Minimum required: {min_class_samples}, found: {counts.min()}."
            )

    def get_property_split(
        self, prop_name: str, prop_cfg: PropConfig, train_split: float, base_seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates balanced undersampled splits under strict cross-session determinism.
        """
        indices, labels = self._extract(prop_name, prop_cfg)
        indices = np.array(indices, dtype=np.int64)
        labels = np.array(labels, dtype=np.int64)

        # Enforce pre-flight verification guards before downsampling arrays
        self._check_min_class(labels, prop_name)

        classes, counts = np.unique(labels, return_counts=True)
        min_count = min(counts)

        # MD5 cryptographic seed routing replaces random session salt behaviors
        purpose_key = f"split_balancing_{prop_name}"
        seed_val = get_seed(base_seed, purpose_key, 0)
        rng = np.random.default_rng(seed_val)

        balanced_indices = []
        balanced_labels = []

        for c in classes:
            c_mask = labels == c
            c_indices = indices[c_mask]
            c_labels = labels[c_mask]

            chosen_sub_idx = rng.choice(len(c_indices), size=min_count, replace=False)
            balanced_indices.extend(c_indices[chosen_sub_idx])
            balanced_labels.extend(c_labels[chosen_sub_idx])

        balanced_indices = np.array(balanced_indices, dtype=np.int64)
        balanced_labels = np.array(balanced_labels, dtype=np.int64)

        pair_ids_arr = np.array([self._pair_ids[i] for i in balanced_indices])
        has_pairs = any(pid is not None for pid in pair_ids_arr)

        if has_pairs:
            unique_pairs = np.array(list({pid for pid in pair_ids_arr if pid is not None}))
            split_rng = np.random.default_rng(seed_val)
            split_rng.shuffle(unique_pairs)
            n_train_pairs = int(len(unique_pairs) * train_split)
            train_pairs = set(unique_pairs[:n_train_pairs])

            train_mask = np.array([pid in train_pairs for pid in pair_ids_arr])
            orphan_mask = np.array([pid is None for pid in pair_ids_arr])
            orphan_train, orphan_test = np.where(orphan_mask)[0], np.array([], dtype=int)
            if orphan_mask.any():
                orphan_idx = np.where(orphan_mask)[0]
                n_orphan_train = int(len(orphan_idx) * train_split)
                split_rng.shuffle(orphan_idx)
                orphan_train = orphan_idx[:n_orphan_train]
                orphan_test = orphan_idx[n_orphan_train:]

            paired_train = np.where(train_mask & ~orphan_mask)[0]
            paired_test = np.where(~train_mask & ~orphan_mask)[0]
            train_pos = np.concatenate([paired_train, orphan_train])
            test_pos = np.concatenate([paired_test, orphan_test])

            return (
                balanced_indices[train_pos], balanced_indices[test_pos],
                balanced_labels[train_pos], balanced_labels[test_pos],
            )

        return train_test_split(
            balanced_indices,
            balanced_labels,
            train_size=train_split,
            stratify=balanced_labels,
            random_state=seed_val
        )
