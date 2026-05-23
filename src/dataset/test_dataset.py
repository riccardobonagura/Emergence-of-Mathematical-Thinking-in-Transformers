"""
test_dataset.py
===============
Test suite per la Fase 0 del dataset.
Copre build_stimuli.py e build_control.py.

Sezioni:
  1. UNIT            — correttezza aritmetica e struttura dati
  2. INTEGRATION     — generazione + tokenizzazione + validate_dataset
  3. STATISTICAL     — distribuzioni e bilanciamento (aritmetica)
  4. ROUND-TRIP      — serializzazione JSONL (aritmetica)
  5. CONTROL         — schema, tier coverage, zero duplicati, compatibilità merge
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

try:
    from build_stimuli import (
        CONTRASTS,
        TEMPLATES,
        BalancedArithmeticGenerator,
        Stimulus,
        populate_token_fields,
        validate_dataset,
    )
except ImportError as e:
    print(f"[ERRORE] Impossibile importare build_stimuli: {e}")
    sys.exit(1)

# build_control.py sostituisce build_control_neutral.py e build_control_numeric.py.
# I wrapper backward-compatible espongono la stessa interfaccia funzionale.
try:
    from build_control import (
        NeutralGenerator,
        NumericGenerator,
        generate_neutral_stimuli,
        generate_control_stimuli as generate_numeric_stimuli,
    )
except ImportError as e:
    print(f"[ERRORE] Impossibile importare build_control: {e}")
    sys.exit(1)

try:
    from merge_stimuli import REQUIRED_ROOT_KEYS, validate_schema
except ImportError as e:
    print(f"[ERRORE] Impossibile importare merge_stimuli: {e}")
    sys.exit(1)


PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = {"passed": 0, "failed": 0}

TIERS = ("short", "medium", "long")


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  {PASS} {name}")
        results["passed"] += 1
    else:
        print(f"  {FAIL} {name}")
        if detail:
            print(f"      → {detail}")
        results["failed"] += 1
    return condition


def section(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ---------------------------------------------------------------------------
# Sezione 1 — Unit
# ---------------------------------------------------------------------------

def test_unit() -> None:
    section("1. UNIT — Correttezza aritmetica e struttura dati")
    gen = BalancedArithmeticGenerator(min_n=1, max_n=50, seed=0)

    for a, op, b, expected in [(8,"+",3,11),(8,"-",3,5),(3,"-",8,-5),(6,"*",4,24),(20,"/",4,5)]:
        res = gen._compute_result(a, op, b)
        check(f"_compute_result({a} {op} {b}) == {expected}", res == expected, f"ottenuto {res}")

    lab = gen._make_labels(3, "-", 8)
    check("sign negativo corretto", lab.sign == 1)
    check("parity di -5 è dispari", lab.parity == 1)


# ---------------------------------------------------------------------------
# Sezione 2 — Integration
# ---------------------------------------------------------------------------

def test_integration(pairs: int, tokenizer_name: str) -> List[Stimulus]:
    section("2. INTEGRATION — Generazione + tokenizzazione + validate_dataset")
    gen     = BalancedArithmeticGenerator(min_n=1, max_n=200, seed=42)
    dataset = gen.build_balanced_dataset(pairs_per_contrast=pairs)

    check("dataset_version == 'v4'", all(s.dataset_version == "v4" for s in dataset))

    print(f"\n  Caricamento tokenizer: {tokenizer_name} ...")
    try:
        tokenized = populate_token_fields(dataset, tokenizer_name)
    except Exception as e:
        check("Tokenizzazione senza errori", False, str(e))
        return dataset

    check("Tutti i n_tokens sono popolati",
          all(s.token_fields.n_tokens is not None for s in tokenized))
    check("equals_sign_index trovato per tutti gli stimoli",
          all(s.token_fields.equals_sign_index >= 0 for s in tokenized))
    check("operator_token_index trovato per tutti gli stimoli",
          all(s.token_fields.operator_token_index >= 0 for s in tokenized))

    try:
        validate_dataset(tokenized)
        check("validate_dataset() supera tutti i check interni", True)
    except AssertionError as e:
        check("validate_dataset() supera tutti i check interni", False, str(e))

    return tokenized


# ---------------------------------------------------------------------------
# Sezione 3 — Statistical
# ---------------------------------------------------------------------------

def test_statistical(dataset: List[Stimulus]) -> None:
    section("3. STATISTICAL — Distribuzioni e bilanciamento (aritmetica)")

    pair_counts = Counter(s.contrast.pair_id for s in dataset)
    check("Ogni pair_id compare 2 volte", all(v == 2 for v in pair_counts.values()))

    check("extraction_strategy_by_property ha le 4 chiavi corrette",
          all(
              set(s.extraction_strategy_by_property.keys()) == {"operator", "sign", "parity", "magnitude"}
              for s in dataset
          ))


# ---------------------------------------------------------------------------
# Sezione 4 — Round-trip
# ---------------------------------------------------------------------------

def test_roundtrip(dataset: List[Stimulus], tmp_path: Path) -> None:
    section("4. ROUND-TRIP — Serializzazione JSONL (aritmetica)")
    out_file = tmp_path / "test_roundtrip.jsonl"
    BalancedArithmeticGenerator().write_jsonl(out_file, dataset)

    required = {
        "id", "text", "split", "template_id", "labels", "contrast",
        "token_fields", "extraction_strategy_by_property",
        "dataset_version", "category", "n_reasoning_steps",
    }
    malformed = [
        (i, set(required) - obj.keys())
        for i, line in enumerate(out_file.read_text(encoding="utf-8").strip().split("\n"))
        for obj in [json.loads(line)]
        if required - obj.keys()
    ]
    check("JSONL valido e schema rispettato", not malformed, f"Errori: {malformed[:3]}")


# ---------------------------------------------------------------------------
# Sezione 5 — Control categories
# ---------------------------------------------------------------------------

def test_control_categories(tokenizer_name: str, tmp_path: Path) -> None:
    """
    5a. Schema radice completo (REQUIRED_ROOT_KEYS)
    5b. Tipi sentinel corretti (sign/parity == -1 come int)
    5c. extraction_strategy_by_property: dict con le 4 chiavi
    5d. Compatibilità cross-categoria: stesse chiavi di CAT-ARITH
    5e. Round-trip JSON: tipi int preservati dopo dumps/loads
    5f. Unicità degli ID nel gruppo
    5g. validate_schema di merge_stimuli supera dopo tokenizzazione
    5h. Tier coverage: tutti e tre i tier (short/medium/long) rappresentati
    5i. Zero duplicati testuali nel campione generato
    """
    section("5. CONTROL CATEGORIES — Schema, tier coverage, zero duplicati")

    neutral_stimuli = generate_neutral_stimuli(n=90)   # 30 per tier
    numeric_stimuli = generate_numeric_stimuli(n=90)

    # Chiavi CAT-ARITH calcolate una volta sola (usata in 5d)
    arith_keys = set(
        BalancedArithmeticGenerator(min_n=1, max_n=10, seed=0)
        .build_balanced_dataset(pairs_per_contrast=1)[0]
        .to_dict().keys()
    )

    for label, stimuli in [("CTRL-NEU", neutral_stimuli), ("CTRL-NUM", numeric_stimuli)]:

        # 5a — Schema radice completo
        missing_keys_list = [
            (s.id, REQUIRED_ROOT_KEYS - s.to_dict().keys())
            for s in stimuli
            if REQUIRED_ROOT_KEYS - s.to_dict().keys()
        ]
        check(f"{label} 5a: schema radice completo",
              not missing_keys_list,
              f"Stimoli con chiavi mancanti: {missing_keys_list[:2]}")

        # 5b — Tipi e valori sentinel
        wrong_types = [s.id for s in stimuli
                       if not isinstance(s.labels.sign, int) or not isinstance(s.labels.parity, int)]
        check(f"{label} 5b: sign e parity sono int",
              not wrong_types, f"ID con tipo errato: {wrong_types[:3]}")

        wrong_val = [s.id for s in stimuli if s.labels.sign != -1 or s.labels.parity != -1]
        check(f"{label} 5b: sign == -1 e parity == -1",
              not wrong_val, f"ID con valore errato: {wrong_val[:3]}")

        # 5c — extraction_strategy_by_property
        expected_keys = {"operator", "sign", "parity", "magnitude"}
        wrong_strat = [
            (s.id, set(s.extraction_strategy_by_property.keys()))
            for s in stimuli
            if not isinstance(s.extraction_strategy_by_property, dict)
            or s.extraction_strategy_by_property.keys() != expected_keys
        ]
        check(f"{label} 5c: extraction_strategy_by_property corretto",
              not wrong_strat, f"Errori: {wrong_strat[:2]}")

        # 5d — Compatibilità cross-categoria con CAT-ARITH
        ctrl_keys      = set(stimuli[0].to_dict().keys())
        symmetric_diff = arith_keys.symmetric_difference(ctrl_keys)
        check(f"{label} 5d: chiavi identiche a CAT-ARITH",
              not symmetric_diff, f"Chiavi asimmetriche: {symmetric_diff}")

        # 5e — Round-trip JSON
        type_errors = [
            (s.id, field, type(json.loads(s.to_json())["labels"][field]).__name__)
            for s in stimuli
            for field in ("sign", "parity")
            if not isinstance(json.loads(s.to_json())["labels"][field], int)
        ]
        check(f"{label} 5e: tipi int preservati dopo round-trip JSON",
              not type_errors, f"Errori: {type_errors[:3]}")

        # 5f — Unicità ID
        ids = [s.id for s in stimuli]
        check(f"{label} 5f: ID unici",
              len(ids) == len(set(ids)),
              f"Duplicati: {[x for x in ids if ids.count(x) > 1][:3]}")

        # 5h — Tier coverage: tutti e tre i tier strutturali devono comparire.
        # La struttura del testo è il proxy pre-tokenizzazione per la lunghezza.
        # Usiamo la lunghezza in caratteri come discriminante (short < 30, long > 50).
        char_lens   = [len(s.text) for s in stimuli]
        has_short   = any(l < 35  for l in char_lens)
        has_long    = any(l > 55  for l in char_lens)
        has_medium  = any(35 <= l <= 55 for l in char_lens)
        check(f"{label} 5h: tutti e tre i tier di lunghezza rappresentati",
              has_short and has_medium and has_long,
              f"char_len range: [{min(char_lens)}, {max(char_lens)}] "
              f"short={has_short} medium={has_medium} long={has_long}")

        # 5i — Zero duplicati testuali
        texts = [s.text for s in stimuli]
        n_dup = len(texts) - len(set(texts))
        check(f"{label} 5i: zero duplicati testuali ({len(texts)} stimoli)",
              n_dup == 0, f"Duplicati trovati: {n_dup}")

    # 5g — validate_schema post-tokenizzazione (richiede tokenizer, eseguito una sola volta)
    print(f"\n  Tokenizzazione stimoli di controllo ({tokenizer_name}) per 5g ...")
    try:
        tok_neu = populate_token_fields(neutral_stimuli, tokenizer_name)
        tok_num = populate_token_fields(numeric_stimuli, tokenizer_name)
        check("5g: tokenizzazione stimoli di controllo riuscita", True)
    except Exception as e:
        check("5g: tokenizzazione stimoli di controllo riuscita", False, str(e))
        return

    for label, stimuli in [("CTRL-NEU", tok_neu), ("CTRL-NUM", tok_num)]:
        schema_errors = [
            (s.id, msg)
            for s in stimuli
            for ok, msg in [validate_schema(s.to_dict())]
            if not ok
        ]
        check(f"{label} 5g: validate_schema supera post-tokenizzazione",
              not schema_errors, f"Errori: {schema_errors[:2]}")

        # Bonus: distribuzione token_length_strata dopo tokenizzazione reale
        strata = Counter(s.token_fields.token_length_strata for s in stimuli)
        missing_tiers = [t for t in TIERS if strata.get(t, 0) == 0]
        check(f"{label} 5g+: tutti i tier presenti in token_length_strata reale",
              not missing_tiers,
              f"Tier assenti: {missing_tiers} — distribuzione: {dict(strata)}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",     type=int, default=50)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--tmp_dir",   type=str, default="/tmp/test_dataset")
    args = parser.parse_args()

    tmp_path = Path(args.tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

    test_unit()
    dataset = test_integration(args.pairs, args.tokenizer)
    test_statistical(dataset)
    test_roundtrip(dataset, tmp_path)
    test_control_categories(args.tokenizer, tmp_path)

    total = results["passed"] + results["failed"]
    print(f"\n  Risultati: {results['passed']}/{total} test superati")
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()