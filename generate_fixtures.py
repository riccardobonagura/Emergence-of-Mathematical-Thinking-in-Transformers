#!/usr/bin/env python
"""Generatore di fixture per il test end-to-end del modulo Linear Probing."""

import json
import torch
import numpy as np
from pathlib import Path

def create_fixtures():
    base_dir = Path("data")
    processed_dir = base_dir / "processed" / "fixture-model"
    stimuli_dir = base_dir / "stimuli"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    N, d, L = 100, 64, 3

    # 1. Hidden States (.pt) - Salvati direttamente in FP32 per questo test
    print(f"Generazione tensori: {L} layer di shape ({N}, {d})")
    for l in range(L):
        H = torch.randn(N, d, dtype=torch.float32)
        torch.save(H, processed_dir / f"layer_{l:02d}.pt")

    # 2. Metadata (metadata.json)
    stimuli_ids = [f"stim_{i:03d}" for i in range(N)]
    metadata = {
        "model_name": "fixture-model",
        "n_layers": L,
        "d_model": d,
        "n_stimuli": N,
        "stimuli_ids": stimuli_ids
    }
    with open(processed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. Stimuli (stimuli.jsonl)
    print("Generazione label bilanciate per parity e sign...")
    with open(stimuli_dir / "stimuli.jsonl", "w") as f:
        for i, sid in enumerate(stimuli_ids):
            # Alterniamo 0 e 1 per garantire che lo split non fallisca
            record = {
                "id": sid,
                "category": "arithmetic",
                "labels": {
                    "parity": i % 2,
                    "sign": (i // 2) % 2,
                    "operator": -1  # Proprietà assente per testare l'esclusione
                }
            }
            f.write(json.dumps(record) + "\n")

    print(f" Fixture create in {processed_dir.absolute()}")

if __name__ == "__main__":
    create_fixtures()