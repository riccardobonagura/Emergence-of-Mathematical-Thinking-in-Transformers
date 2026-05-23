"""
build_stimuli.py
===================
Motore sperimentale per dataset aritmetico minimally contrastive.
Fase 0 del progetto: "Dinamica Geometrica nei Transformer".

Correzioni applicate rispetto a v3:
  1. Label-matching nelle coppie: ogni coppia condivide segno e parità del risultato.
  2. Tokenizzazione a runtime: token_length_strata e indici calcolati dal modello target.
  3. Generazione divisione controllata via a = b * k (distribuzione uniforme).
  4. Split stratificato su (parity, sign), non solo su indice di coppia.
  5. Quattro contrasti bilanciati: ogni operatore appare esattamente 2 volte.
  6. extraction_strategy_by_property: mapping granulare per estrarre l'operatore dal token corretto.
  7. Normalizzazione chiave per operatori commutativi (evita (5,3) e (3,5) come distinti).

UTILIZZO
--------
# Senza tokenizer — genera il JSONL con token_fields = null:
    python build_stimuli_v4.py

# Con tokenizer HuggingFace — popola i campi di tokenizzazione a runtime:
    python build_stimuli_v4.py --tokenizer microsoft/phi-2 --pairs 250

OUTPUT
------
    data/raw/stimuli_arithmetic_v4.jsonl
    data/raw/stimuli_arithmetic_v4_meta.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

Operator = str
PairKey = Tuple[int, int]  # (a, b) normalizzato

TEMPLATES: Dict[str, str] = {
    "TPL-CALC-01": "Calcola il risultato: {a} {op} {b} =",
    "TPL-CALC-02": "Risolvi la seguente espressione: {a} {op} {b} = ?",
    "TPL-CALC-03": "{a} {op} {b} =",
}

# Quattro contrasti: ogni operatore compare esattamente due volte.
# Ordine: (op_A, op_B) — stimolo A usa op_A, stimolo B usa op_B.
CONTRASTS: List[Tuple[Operator, Operator]] = [
    ("+", "*"),
    ("-", "+"),
    ("*", "/"),
    ("/", "-"),
]

# Root progetto: .../tesi_triennale
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Labels:
    operator: str
    result: int
    sign: int              # 0 = positivo o zero, 1 = negativo
    parity: int            # 0 = pari, 1 = dispari
    magnitude_log10: float # log10(|result|), 0.0 se result == 0


@dataclass(frozen=True)
class TokenFields:
    """
    Popolato dallo step di preprocessing con tokenizer reale.
    Tutti i campi sono None se il generatore viene eseguito senza tokenizer.
    """
    n_tokens: Optional[int]
    token_ids: Optional[List[int]]
    token_strs: Optional[List[str]]
    token_length_strata: Optional[str]   # "short" | "medium" | "long"
    equals_sign_index: Optional[int]     # -1 se assente nel template
    operator_token_index: Optional[int]  # Indice del token dell'operatore (+, -, *, /)
    last_token_index: Optional[int]


@dataclass(frozen=True)
class Contrast:
    pair_id: str
    varying_axis: str                # "operator"
    controlled_axes: List[str]       # ["operands", "format", "sign", "parity"]


@dataclass(frozen=True)
class Stimulus:
    id: str
    text: str
    split: str                       # "geometric_eval" | "finetuning_train"
    template_id: str
    macro_format: str                # "symbolic"
    category: str                    # "CAT-ARITH"
    extraction_strategy_by_property: Dict[str, str] # Mapping per probing accurato
    n_reasoning_steps: int           # 1 per aritmetica elementare
    labels: Labels
    contrast: Contrast
    token_fields: TokenFields
    ood_target: str                  # "in_distribution"
    dataset_version: str             # "v4"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Generatore
# ---------------------------------------------------------------------------

class BalancedArithmeticGenerator:
    """
    Genera stimoli aritmetici controllati, bilanciati e contrastivi.

    Garanzie:
    - Ogni coppia (A, B) condivide (a, b, template, sign, parity).
    - Il dominio per la divisione è generato via a = b * k (uniforme).
    - La chiave di coppia è normalizzata per operatori commutativi.
    - Lo split è stratificato su (parity, sign).
    """

    def __init__(
        self,
        min_n: int = 1,
        max_n: int = 200,
        seed: int = 42,
    ) -> None:
        self.min_n = min_n
        self.max_n = max_n
        self.rng = random.Random(seed)
        self.pair_counter = 0

    # ------------------------------------------------------------------
    # Calcolo etichette
    # ------------------------------------------------------------------

    def _compute_result(self, a: int, op: Operator, b: int) -> int:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a // b
        raise ValueError(f"Operatore non supportato: {op}")

    def _make_labels(self, a: int, op: Operator, b: int) -> Labels:
        res = self._compute_result(a, op, b)
        sign = 1 if res < 0 else 0
        parity = abs(res) % 2
        mag = round(math.log10(abs(res)), 2) if res != 0 else 0.0
        return Labels(
            operator=op,
            result=res,
            sign=sign,
            parity=parity,
            magnitude_log10=mag,
        )

    # ------------------------------------------------------------------
    # Normalizzazione chiave coppia
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_key(a: int, b: int, op1: Operator, op2: Operator) -> PairKey:
        """
        Per operatori commutativi ({+,*} con {+,*}), tratta (a,b) e (b,a)
        come identici per evitare quasi-duplicati semantici.
        """
        commutative = {"+", "*"}
        if op1 in commutative and op2 in commutative:
            return (min(a, b), max(a, b))
        return (a, b)

    # ------------------------------------------------------------------
    # Generazione dominio valido per divisione
    # ------------------------------------------------------------------

    def _division_domain(self, other_op: Operator) -> List[Tuple[int, int]]:
        """
        Genera coppie (a, b) per un contrasto che include '/'.
        Usa a = b * k per ottenere una distribuzione uniforme sui divisori.
        Verifica la validità anche per other_op.
        """
        pairs: List[Tuple[int, int]] = []
        b_range = range(self.min_n, self.max_n + 1)
        for b in b_range:
            # k tale che a = b*k resti in [min_n, max_n]
            k_max = self.max_n // b
            for k in range(1, k_max + 1):
                a = b * k
                if a < self.min_n or a > self.max_n:
                    continue
                # Verifica anche per l'altro operatore
                if other_op == "/" and (b == 0 or a % b != 0):
                    continue
                pairs.append((a, b))
        return pairs

    def _standard_domain(self) -> List[Tuple[int, int]]:
        """Dominio completo per operatori senza vincoli di divisione."""
        return list(itertools.product(range(self.min_n, self.max_n + 1), repeat=2))

    def _get_valid_domain(self, op1: Operator, op2: Operator) -> List[Tuple[int, int]]:
        if "/" in (op1, op2):
            other = op2 if op1 == "/" else op1
            return self._division_domain(other)
        return self._standard_domain()

    # ------------------------------------------------------------------
    # Label-matching: filtra coppie con stesso (sign, parity) per entrambi gli op
    # ------------------------------------------------------------------

    def _is_label_compatible(
        self,
        a: int,
        b: int,
        op1: Operator,
        op2: Operator,
    ) -> bool:
        """
        Restituisce True solo se i due operatori producono risultati con
        identici sign e parity. Questo garantisce che le variabili dipendenti
        del probing (sign, parity) non siano confounded con l'operatore.
        """
        lab1 = self._make_labels(a, op1, b)
        lab2 = self._make_labels(a, op2, b)
        return lab1.sign == lab2.sign and lab1.parity == lab2.parity

    # ------------------------------------------------------------------
    # Split stratificato
    # ------------------------------------------------------------------

    def _assign_split(
        self,
        parity: int,
        sign: int,
        stratum_counters: Dict[Tuple[int, int], Dict[str, int]],
        eval_fraction: float = 0.8,
    ) -> str:
        """
        Assegna lo split in modo stratificato per (parity, sign).
        Ogni strato mantiene il proprio contatore per garantire
        che la proporzione 80/20 valga dentro ogni strato.
        """
        key = (parity, sign)
        if key not in stratum_counters:
            stratum_counters[key] = {"geometric_eval": 0, "finetuning_train": 0}
        counts = stratum_counters[key]
        total = counts["geometric_eval"] + counts["finetuning_train"]
        if total == 0:
            split = "geometric_eval"
        else:
            current_eval_frac = counts["geometric_eval"] / total
            split = "geometric_eval" if current_eval_frac < eval_fraction else "finetuning_train"
        counts[split] += 1
        return split

    # ------------------------------------------------------------------
    # Costruzione dataset
    # ------------------------------------------------------------------

    def build_balanced_dataset(self, pairs_per_contrast: int) -> List[Stimulus]:
        """
        Genera il dataset completo.

        pairs_per_contrast: numero di coppie (A, B) per ogni tipo di contrasto.
        Totale stimoli = pairs_per_contrast × len(CONTRASTS) × 2.
        """
        dataset: List[Stimulus] = []
        seen_keys: Set[PairKey] = set()
        stratum_counters: Dict[Tuple[int, int], Dict[str, int]] = {}

        for op1, op2 in CONTRASTS:
            raw_domain = self._get_valid_domain(op1, op2)

            # Filtra: label-compatible + chiave non già usata
            compatible: List[Tuple[int, int]] = []
            for a, b in raw_domain:
                if not self._is_label_compatible(a, b, op1, op2):
                    continue
                norm_key = self._normalize_key(a, b, op1, op2)
                if norm_key in seen_keys:
                    continue
                compatible.append((a, b))

            if len(compatible) < pairs_per_contrast:
                raise RuntimeError(
                    f"Dominio label-compatible esaurito per {op1} vs {op2}. "
                    f"Richieste {pairs_per_contrast} coppie, disponibili {len(compatible)}. "
                    f"Aumenta max_n o riduci pairs_per_contrast."
                )

            sampled = self.rng.sample(compatible, pairs_per_contrast)

            for a, b in sampled:
                norm_key = self._normalize_key(a, b, op1, op2)
                seen_keys.add(norm_key)

                self.pair_counter += 1
                pair_id = f"PAIR-OP-{self.pair_counter:04d}"
                template_id = self.rng.choice(list(TEMPLATES.keys()))

                # Calcola labels per op1 (sign/parity identici per op2 per costruzione)
                lab_ref = self._make_labels(a, op1, b)

                split = self._assign_split(
                    lab_ref.parity,
                    lab_ref.sign,
                    stratum_counters,
                )

                for idx, op in enumerate([op1, op2]):
                    labels = self._make_labels(a, op, b)
                    text = TEMPLATES[template_id].format(a=a, op=op, b=b)

                    stimulus = Stimulus(
                        id=f"{pair_id}-{'A' if idx == 0 else 'B'}",
                        text=text,
                        split=split,
                        template_id=template_id,
                        macro_format="symbolic",
                        category="CAT-ARITH",
                        extraction_strategy_by_property={
                            "operator": "operator_token",
                            "sign": "equals_token",
                            "parity": "equals_token",
                            "magnitude": "equals_token"
                        },
                        n_reasoning_steps=1,
                        labels=labels,
                        contrast=Contrast(
                            pair_id=pair_id,
                            varying_axis="operator",
                            controlled_axes=["operands", "format", "sign", "parity"],
                        ),
                        token_fields=TokenFields(
                            n_tokens=None,
                            token_ids=None,
                            token_strs=None,
                            token_length_strata=None,
                            equals_sign_index=None,
                            operator_token_index=None,
                            last_token_index=None,
                        ),
                        ood_target="in_distribution",
                        dataset_version="v4",
                    )
                    dataset.append(stimulus)

        return dataset

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write_jsonl(self, output_path: str | Path, stimuli: List[Stimulus]) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for s in stimuli:
                f.write(s.to_json() + "\n")
        return output

    def write_metadata(
        self, output_path: str | Path, stimuli: List[Stimulus]
    ) -> Path:
        """Salva statistiche aggregate e distribuzione delle classi."""
        from collections import Counter

        meta: dict = {
            "dataset_version": "v4",
            "total_stimuli": len(stimuli),
            "total_pairs": self.pair_counter,
            "config": {
                "min_n": self.min_n,
                "max_n": self.max_n,
                "contrasts": CONTRASTS,
            },
            "split_distribution": Counter(s.split for s in stimuli),
            "operator_distribution": Counter(s.labels.operator for s in stimuli),
            "parity_distribution": Counter(s.labels.parity for s in stimuli),
            "sign_distribution": Counter(s.labels.sign for s in stimuli),
            "template_distribution": Counter(s.template_id for s in stimuli),
            "label_compatibility_verified": True,
            "stratified_split": True,
            "token_fields_populated": all(
                s.token_fields.n_tokens is not None for s in stimuli
            ),
        }
        # Converti Counter in dict per JSON
        for k in ("split_distribution", "operator_distribution",
                   "parity_distribution", "sign_distribution", "template_distribution"):
            meta[k] = dict(meta[k])

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return output


# ---------------------------------------------------------------------------
# Preprocessing tokenizer (step opzionale, separato dalla generazione)
# ---------------------------------------------------------------------------

def populate_token_fields(
    stimuli: List[Stimulus],
    tokenizer_name: str,
) -> List[Stimulus]:
    """
    Esegue la tokenizzazione reale e popola TokenFields per ogni stimolo.
    Calcola n_tokens, token_ids, token_strs, indici di posizione e strata.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers non installata. "
            "Esegui: pip install transformers"
        )

    print(f"Caricamento tokenizer: {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _find_equals_index(token_strs: List[str]) -> int:
        """Trova l'indice del token '=' (o varianti) nella sequenza."""
        for i, t in enumerate(token_strs):
            clean = t.replace("Ġ", "").replace("▁", "").strip()
            if clean == "=":
                return i
        return -1

    def _find_operator_index(token_strs: List[str]) -> int:
        """Trova l'indice del token dell'operatore (+, -, *, /)."""
        operators = {"+", "-", "*", "/"}
        for i, t in enumerate(token_strs):
            clean = t.replace("Ġ", "").replace("▁", "").strip()
            if clean in operators:
                return i
        return -1

    def _strata(n: int) -> str:
        if n <= 5:
            return "short"
        if n <= 10:
            return "medium"
        return "long"

    updated: List[Stimulus] = []
    for s in stimuli:
        enc = tokenizer(s.text, return_tensors=None)
        ids: List[int] = enc["input_ids"]
        strs: List[str] = tokenizer.convert_ids_to_tokens(ids)
        n = len(ids)

        tf = TokenFields(
            n_tokens=n,
            token_ids=ids,
            token_strs=strs,
            token_length_strata=_strata(n),
            equals_sign_index=_find_equals_index(strs),
            operator_token_index=_find_operator_index(strs),
            last_token_index=n - 1,
        )

        updated.append(
            Stimulus(
                id=s.id,
                text=s.text,
                split=s.split,
                template_id=s.template_id,
                macro_format=s.macro_format,
                category=s.category,
                extraction_strategy_by_property=s.extraction_strategy_by_property,
                n_reasoning_steps=s.n_reasoning_steps,
                labels=s.labels,
                contrast=s.contrast,
                token_fields=tf,
                ood_target=s.ood_target,
                dataset_version=s.dataset_version,
            )
        )

    return updated


# ---------------------------------------------------------------------------
# Validazione post-generazione
# ---------------------------------------------------------------------------

def validate_dataset(stimuli: List[Stimulus]) -> None:
    """
    Checklist di sanity check. Solleva AssertionError se qualcosa non torna.
    Da eseguire sempre prima di chiudere la Fase 0.
    """
    from collections import defaultdict, Counter

    # 1. Nessun id duplicato
    ids = [s.id for s in stimuli]
    assert len(ids) == len(set(ids)), "ID duplicati nel dataset"

    # 2. Ogni pair_id compare esattamente due volte (A e B)
    pair_counts = Counter(s.contrast.pair_id for s in stimuli)
    bad_pairs = {k: v for k, v in pair_counts.items() if v != 2}
    assert not bad_pairs, f"Coppie con numero di stimoli != 2: {bad_pairs}"

    # 3. Verifica label-compatibility dentro ogni coppia
    pair_map = defaultdict(list)
    for s in stimuli:
        pair_map[s.contrast.pair_id].append(s)

    for pid, pair in pair_map.items():
        assert len(pair) == 2, f"Coppia {pid} ha {len(pair)} stimoli"
        s_a, s_b = pair
        assert s_a.labels.sign == s_b.labels.sign, (
            f"Coppia {pid}: sign diverso ({s_a.labels.sign} vs {s_b.labels.sign})"
        )
        assert s_a.labels.parity == s_b.labels.parity, (
            f"Coppia {pid}: parity diversa ({s_a.labels.parity} vs {s_b.labels.parity})"
        )
        assert s_a.labels.operator != s_b.labels.operator, (
            f"Coppia {pid}: stesso operatore nei due stimoli"
        )
        assert s_a.split == s_b.split, (
            f"Coppia {pid}: i due stimoli di una coppia sono in split diversi"
        )
        assert s_a.template_id == s_b.template_id, (
            f"Coppia {pid}: template diversi nei due stimoli della coppia"
        )

    # 4. Split stratificato: verifica proporzione 80/20 per ogni strato
    from collections import defaultdict
    strata: Dict = defaultdict(lambda: {"geometric_eval": 0, "finetuning_train": 0})
    for s in stimuli:
        key = (s.labels.parity, s.labels.sign)
        strata[key][s.split] += 1

    for key, counts in strata.items():
        total = sum(counts.values())
        if total < 5:
            continue  # strato troppo piccolo per verificare
        eval_frac = counts["geometric_eval"] / total
        assert 0.70 <= eval_frac <= 0.90, (
            f"Strato {key}: proporzione eval fuori range ({eval_frac:.2f})"
        )

    # 5. Operatori bilanciati (ogni operatore deve avere frequenza simile)
    op_counts = Counter(s.labels.operator for s in stimuli)
    freqs = list(op_counts.values())
    assert max(freqs) / min(freqs) < 2.0, (
        f"Operatori sbilanciati: {dict(op_counts)}"
    )

    print("✓ Validazione superata.")
    print(f"  Stimoli totali:  {len(stimuli)}")
    print(f"  Coppie uniche:   {len(pair_map)}")
    print(f"  Split:           {dict(Counter(s.split for s in stimuli))}")
    print(f"  Operatori:       {dict(op_counts)}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generatore dataset v4")
    parser.add_argument(
        "--pairs", type=int, default=250,
        help="Coppie per contrasto (default 250 → 2000 stimoli totali)"
    )
    parser.add_argument(
        "--min_n", type=int, default=1,
        help="Valore minimo degli operandi (default 1)"
    )
    parser.add_argument(
        "--max_n", type=int, default=200,
        help="Valore massimo degli operandi (default 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed per riproducibilità (default 42)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Nome HuggingFace del tokenizer per popolare token_fields (opzionale)"
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
        help="Directory di output (default: <repo_root>/data/raw)"
    )
    args = parser.parse_args()

    generator = BalancedArithmeticGenerator(
        min_n=args.min_n,
        max_n=args.max_n,
        seed=args.seed,
    )

    print(f"Generazione dataset: {args.pairs} coppie × {len(CONTRASTS)} contrasti × 2 stimoli "
          f"= {args.pairs * len(CONTRASTS) * 2} stimoli attesi")

    dataset = generator.build_balanced_dataset(pairs_per_contrast=args.pairs)

    if args.tokenizer:
        print(f"\nPopolo token_fields con tokenizer: {args.tokenizer}")
        dataset = populate_token_fields(dataset, args.tokenizer)

    out_dir_arg = Path(args.out_dir)
    out_dir = out_dir_arg if out_dir_arg.is_absolute() else (PROJECT_ROOT / out_dir_arg)
    jsonl_path = generator.write_jsonl(out_dir / "stimuli_arithmetic_v4.jsonl", dataset)
    meta_path = generator.write_metadata(out_dir / "stimuli_arithmetic_v4_meta.json", dataset)

    print(f"\nOutput:")
    print(f"  {jsonl_path}")
    print(f"  {meta_path}")

    validate_dataset(dataset)