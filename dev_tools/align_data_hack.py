import json
from pathlib import Path

# 1. Leggiamo i metadati REALI di Pythia
metadata_path = Path("data/processed/pythia-1.4b/metadata.json")
with open(metadata_path, "r") as f:
    meta = json.load(f)
real_ids = meta["stimuli_ids"]

# 2. Generiamo il nuovo stimuli.jsonl allineato
stimuli_output = Path("data/stimuli/stimuli.jsonl")
stimuli_output.parent.mkdir(parents=True, exist_ok=True)

print(f"Riallineamento di {len(real_ids)} stimoli...")

with open(stimuli_output, "w") as f:
    for i, sid in enumerate(real_ids):
        # Logica di assegnazione label per il test di produzione:
        # Per ora usiamo una logica basata sull'indice per avere classi bilanciate
        # (In seguito userai le label reali del tuo dataset aritmetico)
        record = {
            "id": sid,
            "category": "arithmetic" if "PAIR" in sid else "control",
            "labels": {
                "parity": i % 2,        # 0: even, 1: odd
                "sign": (i // 2) % 2,   # 0: positive, 1: negative
                "operator": 0 if "PAIR" in sid else -1
            }
        }
        f.write(json.dumps(record) + "\n")

print(f"✅ File {stimuli_output} rigenerato e allineato ai tensori di Pythia.")