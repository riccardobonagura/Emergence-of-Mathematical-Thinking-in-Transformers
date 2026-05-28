"""
build_control.py — Structural control categories generator.
Sources unique generic sentences from pool files when available, falls back
to a hard-coded diversity pool otherwise. Filters by token length when a
tokenizer is provided.

Pairs with build_stimuli.py: both produce v5 Stimulus dicts (TypedDict).
"""

import logging
import random
from pathlib import Path
from typing import Any, List, Optional

from src.probing.seeds import get_seed

from .build_stimuli import DATASET_VERSION, Stimulus

log = logging.getLogger(__name__)


class LinguisticDiversityPool:
    """Hard-coded structural pool for the combinatorial fallback path."""
    NEU_SUBJECTS = [
        "The artist", "A software", "The mountain", "A citizen", "The dog",
        "A philosopher", "The river", "An engine", "The melody", "A chef",
    ]
    NEU_VERBS = [
        "creates", "compiles", "stands", "votes", "sleeps",
        "argues", "flows", "operates", "resolves", "cooks",
    ]
    NEU_OBJECTS = [
        "a masterpiece", "the data", "in silence", "for change", "on the rug",
        "about ethics", "to the sea", "with precision", "the tension", "a meal",
    ]

    NUM_SUBJECTS = [
        "The temperature", "Chapter", "The project", "Exactly",
        "The patient", "Phase", "The company", "A fraction",
    ]
    NUM_VERBS = [
        "reached", "starts on page", "requires", "costs",
        "needs", "ends at step", "hired", "equals",
    ]
    NUM_OBJECTS = [
        "degrees", "of the manual", "months of work", "dollars",
        "days of rest", "of the process", "employees", "of the total",
    ]
    NUMBERS = [str(i) for i in range(10, 100)]


class ControlGenerator:
    """Generates generic prose reference stimuli for CTRL-NEU / CTRL-NUM."""

    LENGTH_MIN = 4
    LENGTH_MAX = 9

    def __init__(self, category: str, seed: int) -> None:
        if category not in ("CTRL-NEU", "CTRL-NUM"):
            raise ValueError(f"Unsupported control category: {category!r}")
        self.category = category
        self.rng = random.Random(seed)
        self.pool_dir = Path("data/processed")

    # ── Pool loading ─────────────────────────────────────────────────────────

    def _load_pool_file(self, filename: str) -> List[str]:
        path = self.pool_dir / filename
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # ── Fallback generators ──────────────────────────────────────────────────

    def _generate_neu_fallback(self) -> str:
        s = self.rng.choice(LinguisticDiversityPool.NEU_SUBJECTS)
        v = self.rng.choice(LinguisticDiversityPool.NEU_VERBS)
        o = self.rng.choice(LinguisticDiversityPool.NEU_OBJECTS)
        return f"{s} {v} {o}."

    def _generate_num_fallback(self) -> str:
        s = self.rng.choice(LinguisticDiversityPool.NUM_SUBJECTS)
        v = self.rng.choice(LinguisticDiversityPool.NUM_VERBS)
        n = self.rng.choice(LinguisticDiversityPool.NUMBERS)
        o = self.rng.choice(LinguisticDiversityPool.NUM_OBJECTS)
        if self.rng.random() > 0.5:
            return f"{s} {v} {n} {o}."
        return f"{n} {o} {v} by {s.lower()}."

    def _generate_fallback(self) -> str:
        if self.category == "CTRL-NEU":
            return self._generate_neu_fallback()
        return self._generate_num_fallback()

    # ── Build loop ───────────────────────────────────────────────────────────

    def build(self, n_stimuli: int, tokenizer: Any = None) -> List[Stimulus]:
        """
        Generate `n_stimuli` control stimuli. When `tokenizer` is provided
        (HuggingFace fast tokenizer interface — `.encode(text)`), each text is
        accepted only if its token length falls in [LENGTH_MIN, LENGTH_MAX].
        Without a tokenizer, no length filtering is applied.
        """
        pool_filename = "ctrl_neu_pool.txt" if self.category == "CTRL-NEU" else "ctrl_num_pool.txt"
        pool_sentences = self._load_pool_file(pool_filename)

        if pool_sentences and tokenizer is not None:
            filtered = [
                s for s in pool_sentences
                if self.LENGTH_MIN <= len(tokenizer.encode(s)) <= self.LENGTH_MAX
            ]
            if len(filtered) >= n_stimuli:
                pool_sentences = filtered
            else:
                log.warning(
                    "Filtered pool (%d) under target (%d); engaging combinatorial fallback.",
                    len(filtered), n_stimuli,
                )

        stimuli: List[Stimulus] = []
        seen_texts: set = set()
        attempts = 0
        max_attempts = n_stimuli * 20

        while len(stimuli) < n_stimuli and attempts < max_attempts:
            attempts += 1

            text = self.rng.choice(pool_sentences) if pool_sentences else self._generate_fallback()
            if text in seen_texts:
                continue

            if tokenizer is not None:
                t_len = len(tokenizer.encode(text))
                if not (self.LENGTH_MIN <= t_len <= self.LENGTH_MAX):
                    continue

            seen_texts.add(text)
            stimuli.append(self._make_stim(text, idx=len(stimuli)))

        if len(stimuli) < n_stimuli:
            raise ValueError(
                f"Generation loop failed to yield {n_stimuli} valid {self.category} stimuli "
                f"within [{self.LENGTH_MIN}, {self.LENGTH_MAX}] tokens "
                f"(produced {len(stimuli)}). Pool may be missing or undersized."
            )

        return stimuli

    def _make_stim(self, text: str, idx: int) -> Stimulus:
        return {
            "id": f"{self.category}_{idx:05d}",
            "text": text,
            "split": "geometric_eval",
            "category": self.category,
            "template_id": f"{self.category}-TPL",
            "macro_format": "natural_language",
            "n_reasoning_steps": 0,
            "labels": {
                "result": 0,
                "sign": -1,
                "parity": -1,
                "operand1": 0,
                "operand2": 0,
            },
            "contrast": {
                "pair_id": "N/A",
                "varying_axis": "N/A",
                "controlled_axes": (),
            },
            "token_fields": {},
            "ood_target": "control",
            "dataset_version": DATASET_VERSION,
            "probe_layer_strategy": "all_layers",
        }


# ── Public factory helpers (used by test_dataset.py and pipeline scripts) ────

def generate_neutral_stimuli(n: int, seed: int, tokenizer: Any = None) -> List[Stimulus]:
    """Convenience wrapper: deterministic CTRL-NEU stimuli via project seed discipline."""
    derived = get_seed(seed, "build_control_neu", 0)
    return ControlGenerator("CTRL-NEU", derived).build(n, tokenizer=tokenizer)


def generate_numeric_stimuli(n: int, seed: int, tokenizer: Any = None) -> List[Stimulus]:
    """Convenience wrapper: deterministic CTRL-NUM stimuli via project seed discipline."""
    derived = get_seed(seed, "build_control_num", 0)
    return ControlGenerator("CTRL-NUM", derived).build(n, tokenizer=tokenizer)


def length_match_to_arithmetic(
    ctrl_stimuli: List[Stimulus],
    arith_stimuli: List[Stimulus],
    seed: int,
) -> List[Stimulus]:
    """
    Resample `ctrl_stimuli` so its token_length_strata distribution matches
    the empirical distribution of `arith_stimuli`. Both lists must be tokenised
    (populate_token_fields already called).

    Stratified resampling with replacement within each stratum.
    """
    from collections import Counter

    def strata(s: Stimulus) -> Optional[str]:
        return s.get("token_fields", {}).get("token_length_strata")

    arith_strata = [strata(s) for s in arith_stimuli if strata(s) is not None]
    ctrl_pool_by_stratum: dict = {}
    for s in ctrl_stimuli:
        st = strata(s)
        if st is None:
            continue
        ctrl_pool_by_stratum.setdefault(st, []).append(s)

    if not arith_strata or not ctrl_pool_by_stratum:
        raise RuntimeError("length_match_to_arithmetic requires tokenised inputs on both sides.")

    target_counts = Counter(arith_strata)
    rng = random.Random(get_seed(seed, "length_match", 0))

    matched: List[Stimulus] = []
    for stratum, target_n in target_counts.items():
        pool = ctrl_pool_by_stratum.get(stratum, [])
        if not pool:
            raise RuntimeError(
                f"Cannot length-match: arithmetic asks for {target_n} stimuli in stratum "
                f"{stratum!r} but the control pool has none."
            )
        # Sample with replacement so we hit target_n exactly.
        matched.extend(rng.choices(pool, k=target_n))

    return matched
