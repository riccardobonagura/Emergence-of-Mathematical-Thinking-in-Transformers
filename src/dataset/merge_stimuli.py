"""
merge_stimuli.py
===================
Validatore e fustellatore centrale per la Fase 0.
Fonde multipli file JSONL (es. Aritmetica, Controllo) in un unico dataset.
Garantisce la rigorosa consistenza di schema per l'estrattore.

UTILIZZO
--------
    python merge_stimuli.py \
        --inputs data/raw/stimuli_arithmetic_v4.jsonl \
                 data/raw/stimuli_control_num_v4.jsonl \
                 data/raw/stimuli_control_neutral_v4.jsonl \
        --output data/processed/dataset_master.jsonl

    L'ordinamento corretto è: genera → tokenizza → merge.
    Per default il merge BLOCCA se token_fields non sono popolati.
    Usa --allow-untokenized per abbassare l'errore a warning (es. in sviluppo).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Chiavi radice obbligatorie per la Fase 1
REQUIRED_ROOT_KEYS = {
    "id", "text", "split", "template_id", "macro_format", "category",
    "extraction_strategy_by_property", "n_reasoning_steps", "labels",
    "contrast", "token_fields", "ood_target", "dataset_version"
}


def validate_schema(stimulus: dict, require_tokenized: bool = True) -> tuple[bool, str]:
    """
    Verifica che lo stimolo rispetti il contratto di interfaccia della Fase 0.

    Args:
        stimulus:           dizionario deserializzato da una riga JSONL.
        require_tokenized:  se True (default), restituisce errore su n_tokens=None.
                            se False, quel controllo viene saltato (utile in sviluppo).
    """
    missing = REQUIRED_ROOT_KEYS - stimulus.keys()
    if missing:
        return False, f"Chiavi radice mancanti: {missing}"

    if not isinstance(stimulus["labels"], dict):
        return False, "Il campo 'labels' deve essere un dizionario."

    if not isinstance(stimulus["token_fields"], dict):
        return False, "Il campo 'token_fields' deve essere un dizionario."

    if not isinstance(stimulus["extraction_strategy_by_property"], dict):
        return False, "Il campo 'extraction_strategy_by_property' deve essere un dizionario."

    if require_tokenized and stimulus["token_fields"].get("n_tokens") is None:
        return False, (
            "token_fields non popolati (n_tokens è null). "
            "Esegui la tokenizzazione prima del merge, "
            "oppure usa --allow-untokenized per abbassare a warning."
        )

    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge e validazione di dataset JSONL per la Fase 0.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Lista di file JSONL di input (generati e tokenizzati)")
    parser.add_argument("--output", type=str, required=True,
                        help="File JSONL di output (dataset master)")
    parser.add_argument("--allow-untokenized", action="store_true",
                        help="Abbassa l'errore su token_fields mancanti a semplice warning. "
                             "Utile durante lo sviluppo; non usare per esperimenti finali.")
    args = parser.parse_args()

    require_tokenized = not args.allow_untokenized

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    master_dataset      = []
    seen_ids            = set()
    category_counts     = Counter()
    split_counts        = Counter()
    strata_counts       = Counter()
    untokenized_warning = 0

    print(f"\nInizio merge e validazione di {len(args.inputs)} file...")
    if not require_tokenized:
        print("  [WARNING] --allow-untokenized attivo: token_fields mancanti non bloccano il merge.")

    for in_file in args.inputs:
        file_path = Path(in_file)
        if not file_path.exists():
            print(f"[ERRORE] File non trovato: {file_path}")
            sys.exit(1)

        print(f"  Analisi di {file_path.name}...")

        with file_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    stimulus = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[ERRORE] JSON corrotto in {file_path.name} alla riga {line_idx}.")
                    sys.exit(1)

                is_valid, err_msg = validate_schema(stimulus, require_tokenized=require_tokenized)
                if not is_valid:
                    print(f"[ERRORE] Schema non valido in {file_path.name} riga {line_idx}: {err_msg}")
                    sys.exit(1)

                # Warning separato per token_fields mancanti quando --allow-untokenized è attivo
                if not require_tokenized and stimulus["token_fields"].get("n_tokens") is None:
                    untokenized_warning += 1

                s_id = stimulus["id"]
                if s_id in seen_ids:
                    print(f"[ERRORE] Collisione ID rilevata: '{s_id}'.")
                    sys.exit(1)

                seen_ids.add(s_id)
                master_dataset.append(stimulus)
                category_counts[stimulus["category"]] += 1
                split_counts[stimulus.get("split", "unknown")] += 1
                strata_counts[
                    (stimulus.get("token_fields") or {}).get("token_length_strata", "unpopulated")
                ] += 1

    if untokenized_warning:
        print(f"\n  [WARNING] {untokenized_warning} stimoli con token_fields non popolati.")

    # Scrittura del file master
    print(f"\nScrittura dataset master in {out_path}...")
    with out_path.open("w", encoding="utf-8") as out_f:
        for stim in master_dataset:
            out_f.write(json.dumps(stim, ensure_ascii=False) + "\n")

    # Metadati estesi — utili per il debug del dataset e per la sezione Metodologia
    meta_path = out_path.with_suffix(".meta.json")
    meta_info = {
        "total_stimuli":              len(master_dataset),
        "source_files":               [Path(p).name for p in args.inputs],
        "allow_untokenized":          not require_tokenized,
        "schema_verified":            True,
        "category_distribution":      dict(category_counts),
        "split_distribution":         dict(split_counts),
        "token_length_strata":        dict(strata_counts),
    }
    with meta_path.open("w", encoding="utf-8") as meta_f:
        json.dump(meta_info, meta_f, ensure_ascii=False, indent=2)

    print("\n✓ MERGE COMPLETATO CON SUCCESSO.")
    print(f"  Totale stimoli validati: {len(master_dataset)}")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count}")


if __name__ == "__main__":
    main()