"""
build_control.py
================
Generatori per i gruppi di controllo CTRL-NEU e CTRL-NUM.

Architettura:
  ControlGenerator (ABC)
    ├── NeutralGenerator   — prodotto cartesiano su liste lessicali
    └── NumericGenerator   — prodotto cartesiano su (template, slot_sets) espliciti

Garanzie:
  - Zero duplicati: il pool è costruito via prodotto cartesiano, unicità per
    costruzione senza deduplicazione post-hoc.
  - Copertura di tre fasce strutturali (short / medium / long), campionate in
    proporzione uniforme per mitigare il length confound nell'analisi geometrica.
  - Schema JSONL identico a build_stimuli.py (accoppiamento tramite import diretto).

UTILIZZO
--------
    python build_control.py --category neu --n_stimuli 200 --tokenizer gpt2 \
                            --output data/raw/stimuli_control_neu_v4.jsonl
    python build_control.py --category num --n_stimuli 200 --tokenizer gpt2 \
                            --output data/raw/stimuli_control_num_v4.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from build_stimuli import (
        Contrast, Labels, Stimulus, TokenFields, populate_token_fields,
    )
except ImportError:
    print("[ERRORE] build_stimuli.py non trovato nella stessa cartella.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Tipi
# ---------------------------------------------------------------------------

SlotSets  = Dict[str, List]           # {"a": [...], "code": [...], ...}
Template  = Tuple[str, SlotSets]      # ("testo {a} ...", {"a": [...], ...})
TierPool  = Dict[str, List[str]]      # {"short": [...], "medium": [...], "long": [...]}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _expand(templates: List[Template]) -> List[str]:
    """
    Genera tutti i testi possibili per una lista di (template, slot_sets).

    Per ogni template, il prodotto cartesiano dei valori di ogni slot produce
    un testo distinto. L'unicità è garantita per costruzione, senza dedup.

    Esempio:
        _expand([("{a} + {b}", {"a": [1,2], "b": [3,4]})])
        → ["1 + 3", "1 + 4", "2 + 3", "2 + 4"]
    """
    texts = []
    for tpl, slots in templates:
        keys = list(slots.keys())
        for values in product(*slots.values()):
            texts.append(tpl.format(**dict(zip(keys, values))))
    return texts


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ControlGenerator(ABC):
    """
    Logica condivisa: sampling stratificato per tier, costruzione schema, I/O.
    I sottoclassi implementano solo `_build_pool_by_tier()`.
    """

    CATEGORY  : str = ""
    ID_PREFIX : str = ""
    _STRATEGY = {k: "last_token" for k in ("operator", "sign", "parity", "magnitude")}

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    @abstractmethod
    def _build_pool_by_tier(self) -> TierPool: ...

    def build(
        self,
        n: int,
        tier_weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    ) -> List[Stimulus]:
        """
        Genera n stimoli campionati uniformemente sui tre tier di lunghezza.

        tier_weights: proporzione (short, medium, long). Default: uniforme 1/3.
        Raises ValueError se il pool di un tier è insufficiente per la quota.
        """
        pool   = self._build_pool_by_tier()
        counts = {t: round(n * w) for t, w in zip(("short", "medium", "long"), tier_weights)}
        counts["medium"] += n - sum(counts.values())  # corregge arrotondamento

        selected: List[str] = []
        for tier, count in counts.items():
            if len(pool[tier]) < count:
                raise ValueError(
                    f"{self.__class__.__name__}: pool '{tier}' = {len(pool[tier])} testi, "
                    f"richiesti {count}. Riduci n o aggiungi componenti al generatore."
                )
            selected.extend(self.rng.sample(pool[tier], count))

        self.rng.shuffle(selected)
        return [
            self._make_stimulus(f"{self.ID_PREFIX}-{i:04d}", text)
            for i, text in enumerate(selected)
        ]

    def _make_stimulus(self, stim_id: str, text: str) -> Stimulus:
        return Stimulus(
            id=stim_id, text=text,
            split="geometric_eval",
            template_id=f"{self.CATEGORY}-TPL",
            macro_format="natural_language",
            category="CAT-CTRL",
            extraction_strategy_by_property=dict(self._STRATEGY),
            n_reasoning_steps=0,
            labels=Labels(operator="none", result=0, sign=-1, parity=-1, magnitude_log10=0.0),
            contrast=Contrast(pair_id="none", varying_axis="none", controlled_axes=[]),
            token_fields=TokenFields(
                n_tokens=None, token_ids=None, token_strs=None,
                token_length_strata=None,
                equals_sign_index=-1, operator_token_index=-1, last_token_index=None,
            ),
            ood_target="control",
            dataset_version="v4",
        )

    def write_jsonl(self, path: str | Path, stimuli: List[Stimulus]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(s.to_json() for s in stimuli) + "\n", encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# NeutralGenerator
# ---------------------------------------------------------------------------

class NeutralGenerator(ControlGenerator):
    """
    Prosa senza numeri né struttura matematica.

    I tre tier sono costruiti aggiungendo componenti frasali:
      short  = S + P                →  14 × 10          =    140 testi
      medium = S + P + O            →  14 × 10 × 10     =  1.400 testi
      long   = S + P + O + ADV      →  14 × 10 × 10 × 8 = 11.200 testi
    """

    CATEGORY  = "CTRL-NEU"
    ID_PREFIX = "CTRL-NEU"

    _S = [
        "Il vento del mattino", "La pioggia leggera", "Il sole di mezzogiorno",
        "La luna piena", "Il cielo terso", "La nebbia autunnale",
        "Il silenzio della notte", "L'aria fresca", "La luce del tramonto",
        "Il freddo invernale", "Il caldo estivo", "La brezza marina",
        "Il temporale estivo", "La neve silenziosa",
    ]
    _P = [
        "accarezza lentamente", "attraversa in silenzio", "illumina con forza",
        "avvolge con delicatezza", "copre piano piano", "porta via con sé",
        "trasforma gradualmente", "bagna ogni superficie",
        "riempie ogni angolo", "cambia il paesaggio",
    ]
    _O = [
        "i campi aperti", "le strade deserte", "i tetti antichi",
        "le finestre chiuse", "i giardini silenziosi", "i vicoli stretti",
        "le piazze vuote", "i muri bianchi", "le colline lontane", "i boschi profondi",
    ]
    _ADV = [
        "al mattino presto", "nel tardo pomeriggio", "durante la notte",
        "in questa stagione", "senza fare rumore", "con grande intensità",
        "ogni volta che cambia il tempo", "fin dalle prime ore del giorno",
    ]

    # Struttura dichiarativa: template e slot. _expand() fa il resto.
    # Nota: lo slot "{sp}" concatena soggetto e predicato per consentire
    # il prodotto cartesiano con un singolo dizionario per tutti e tre i tier.
    def _build_pool_by_tier(self) -> TierPool:
        sp = [f"{s} {p}" for s, p in product(self._S, self._P)]  # 140 combinazioni
        return {
            "short":  [f"{x}."           for x        in sp],
            "medium": [f"{x} {o}."       for x, o     in product(sp, self._O)],
            "long":   [f"{x} {o} {adv}." for x, o, adv in product(sp, self._O, self._ADV)],
        }


# ---------------------------------------------------------------------------
# NumericGenerator
# ---------------------------------------------------------------------------

class NumericGenerator(ControlGenerator):
    """
    Testo con cifre contestualmente non matematiche (orari, misure, codici).

    Ogni tier è definito come lista di (template, slot_sets).
    I range discreti per ogni slot coprono tutti gli ordini di grandezza
    (1-cifra, 2-cifre, 3-cifre) per produrre variazione di lunghezza in tokenizzazione.

      short  : 5 template × |_A_FULL|       =  5 × 200 = 1.000 testi
      medium : 5 template × varie combo     ≈ 1.000–3.000 testi
      long   : 5 template × varie combo     ≈ 5.000–15.000 testi
    """

    CATEGORY  = "CTRL-NUM"
    ID_PREFIX = "CTRL-NUM"

    # Slot sets. Scelti per coprire le tre fasce numeriche rilevanti per il tokenizer.
    _A_FULL = list(range(1, 201))                                    # 200 valori
    _A      = [1, 3, 7, 12, 25, 47, 78, 100, 134, 167, 199]         #  11 valori
    _B      = [1, 8, 15, 30, 59, 99]                                 #   6 valori
    _C      = [0, 10, 30, 59]                                        #   4 valori (minuti)
    _D      = [1, 15, 50, 99]                                        #   4 valori
    _CODE   = [f"{p}{n}" for p in "ABFGRZ" for n in (12, 37, 61, 88, 99)]  # 30 codici
    _STREET = ["Roma", "Milano", "Napoli", "Torino", "Firenze",
               "Venezia", "Genova", "Palermo", "Bari", "Trieste"]
    _CITY   = ["Parigi", "Vienna", "Berlino", "Madrid", "Zurigo",
               "Londra", "Amsterdam", "Praga", "Lisbona", "Varsavia"]

    def _build_pool_by_tier(self) -> TierPool:
        # Ogni entry è (template_string, {slot_name: lista_valori}).
        # _expand() applica il prodotto cartesiano per noi.
        short_tpls: List[Template] = [
            ("Il binario {a} è temporaneamente chiuso.",              {"a": self._A_FULL}),
            ("Sono arrivate {a} persone all'ingresso.",               {"a": self._A_FULL}),
            ("Mancano {a} giorni alla scadenza.",                     {"a": self._A_FULL}),
            ("L'archivio contiene {a} fascicoli numerati.",           {"a": self._A_FULL}),
            ("Ci sono ancora {a} posti disponibili.",                 {"a": self._A_FULL}),
        ]
        medium_tpls: List[Template] = [
            ("Il treno {code} parte dal binario {a} alle ore {b}.",   {"code": self._CODE, "a": self._A, "b": self._B}),
            ("L'appartamento al piano {a} misura {b} metri quadri.",  {"a": self._A, "b": self._B}),
            ("Il documento {code} è valido per {a} mesi e {b} giorni.", {"code": self._CODE, "a": self._A, "b": self._B}),
            ("L'edificio in via {street} è alto {a} metri e ha {b} piani.", {"street": self._STREET, "a": self._A, "b": self._B}),
            ("Il paziente in sala {a} attende da {b} minuti.",        {"a": self._A, "b": self._B}),
        ]
        long_tpls: List[Template] = [
            ("Il volo {code} parte dal terminal {a} alle {b}:{c:02d}, con scalo di {d} min a {city}.",
             {"code": self._CODE, "a": self._A, "b": self._B, "c": self._C, "d": self._D, "city": self._CITY}),
            ("Il referto riporta {a} cm, {b} kg, pressione {c} su {d}.",
             {"a": self._A, "b": self._B, "c": self._C, "d": self._D}),
            ("La struttura ha {a} locali, {b} bagni, {c} posti auto e {d} mq di giardino.",
             {"a": self._A, "b": self._B, "c": self._C, "d": self._D}),
            ("Il contratto {code} copre {a} mesi, {b} rate da {c} euro, scadenza giorno {d}.",
             {"code": self._CODE, "a": self._A, "b": self._B, "c": self._C, "d": self._D}),
            ("L'impianto ha {a} anni, consuma {b} kWh, serve {c} utenze su {d} edifici.",
             {"a": self._A, "b": self._B, "c": self._C, "d": self._D}),
        ]
        return {
            "short":  _expand(short_tpls),
            "medium": _expand(medium_tpls),
            "long":   _expand(long_tpls),
        }


# ---------------------------------------------------------------------------
# Backward-compatible factory (per non rompere test_dataset.py)
# ---------------------------------------------------------------------------

def generate_neutral_stimuli(n: int = 200, seed: int = 84) -> List[Stimulus]:
    """Wrapper compatibile con build_control_neutral.py."""
    return NeutralGenerator(seed=seed).build(n)


def generate_control_stimuli(n: int = 200, seed: int = 42) -> List[Stimulus]:
    """Wrapper compatibile con build_control_numeric.py."""
    return NumericGenerator(seed=seed).build(n)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generatore stimoli di controllo v4")
    parser.add_argument("--category",   choices=["neu", "num"], required=True)
    parser.add_argument("--n_stimuli",  type=int,  default=200)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--tokenizer",  type=str,  default=None)
    parser.add_argument("--output",     type=str,  default=None)
    args = parser.parse_args()

    gen = NeutralGenerator(seed=args.seed) if args.category == "neu" else NumericGenerator(seed=args.seed)
    default_out = f"data/raw/stimuli_control_{args.category}_v4.jsonl"

    print(f"Generazione {args.n_stimuli} stimoli ({gen.CATEGORY})...")
    stimuli = gen.build(args.n_stimuli)

    if args.tokenizer:
        print(f"Tokenizzazione con: {args.tokenizer}")
        stimuli = populate_token_fields(stimuli, args.tokenizer)

    out = gen.write_jsonl(args.output or default_out, stimuli)
    print(f"✓ {len(stimuli)} stimoli salvati in {out}")