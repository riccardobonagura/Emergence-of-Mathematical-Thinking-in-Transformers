"""
test_pipeline_e2e.py — CPU-only End-to-End Pipeline Integration Test.
Validates the actual execution of RQ1, RQ2, and RQ3 using synthetic v5 data
and mocked hidden states, without requiring GPU, real models, or physical disk I/O.
"""

import json
import yaml
import pytest
import sys
from pathlib import Path
from unittest.mock import patch
import torch

# Direct imports of the pipeline entry-points
import run_rq1
import run_rq2
import run_rq3

def generate_v5_mock_stimuli() -> list[dict]:
    """
    Generates 32 synthetic stimuli: 4 pairs across 4 categories (16 stimuli per property).
    Strictly injects -1 sentinel values for controls per v5 specifications.
    """
    categories = ["CAT-SIGN", "CAT-PARITY", "CTRL-NEU", "CTRL-NUM"]
    stimuli = []
    
    for cat in categories:
        for pair_id in range(4): # 4 pairs = 8 stimuli per category
            for val1, val2 in [(0, 0), (1, 1)]:
                # Sentinel value resolution for controls
                if "CTRL" in cat:
                    sign_val, parity_val = -1, -1
                else:
                    sign_val = val1 if "SIGN" in cat else 0
                    parity_val = val2 if "PARITY" in cat else 0
                    
                stimuli.append({
                    "id": f"{cat}_p{pair_id}_{val1}",
                    "text": f"Mock stimulus for {cat} = ",
                    "category": cat,
                    "dataset_version": "v5",
                    "labels": {
                        "sign": sign_val,
                        "parity": parity_val
                    }
                })
    return stimuli

@pytest.fixture
def e2e_env(tmp_path, monkeypatch):
    """
    Builds the pipeline's expected file system within tmp_path
    and uses chdir to bypass relative paths hardcoded in the scripts.
    """
    monkeypatch.chdir(tmp_path)
    
    # 1. Setup Directory Structure
    data_dir = tmp_path / "data" / "processed"
    results_dir = tmp_path / "results"
    configs_dir = tmp_path / "configs"
    model_dir = data_dir / "pythia-1.4b"
    ckpt_dir = data_dir / "checkpoints_extracted" / "checkpoint-500"
    
    for d in [model_dir, ckpt_dir, results_dir, configs_dir, results_dir / "figures", results_dir / "rq1_emergence"]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Write Config (Aggressively downscaled for fast CPU execution)
    config = {
        "model_name": "pythia-1.4b",
        "output_dir": "results/rq2_probing",
        "figures_dir": "results/figures/rq2",
        "train_split": 0.80,
        "seed": 42,
        "n_jobs": 1, # Prevents multiprocess overhead in CI
        "properties": {
            "sign": {"label_field": "sign", "category": "CAT-SIGN", "type": "binary"},
            "parity": {"label_field": "parity", "category": "CAT-PARITY", "type": "binary"}
        },
        "max_iter": 5, # Low iter for immediate Scikit-Learn fit
        "C": 1.0,
        "solver": "lbfgs",
        "multiclass_strategy": "ovr",
        "bootstrap_n_samples": 5, 
        "bootstrap_ci": 0.95,
        "eval_subset_size": 10,
        "min_class_samples": 2  # <-- FIX: Bypass hardcoded 10 threshold for tests
    }
    with open(configs_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    # 3. Write Master Dataset
    stimuli = generate_v5_mock_stimuli()
    with open(data_dir / "dataset_master_v5.jsonl", "w", encoding="utf-8") as f:
        for s in stimuli:
            f.write(json.dumps(s) + "\n")
            
    # 4. Write Metadata & Dummy Layer Files
    metadata = {
        "stimuli_ids": [s["id"] for s in stimuli],
        "categories": [s["category"] for s in stimuli],
        "probe_strategy": "last_token",
        "dataset_version": "v5",
        "n_layers": 2,
        "d_model": 2048,
        "n_stimuli": len(stimuli),
        "labels": {
            "sign": [s["labels"]["sign"] for s in stimuli],
            "parity": [s["labels"]["parity"] for s in stimuli]
        }
    }
    
    # Baseline
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    (model_dir / "layer_00.pt").touch()
    (model_dir / "layer_01.pt").touch()
    
    # Checkpoint (RQ3)
    with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    (ckpt_dir / "layer_00.pt").touch()
    (ckpt_dir / "layer_01.pt").touch()
    
    return len(stimuli)


def test_full_pipeline_execution(e2e_env):
    """
    Executes real pipeline entry-points while injecting CPU-bound mock tensors
    directly into the execution namespaces.
    """
    n_stimuli = e2e_env
    
    # Pre-allocate random FP32 tensor (Scikit-Learn prefers FP32 over FP16 on CPU)
    mock_tensor = torch.randn(n_stimuli, 2048, dtype=torch.float32).numpy()
    
    # --- EXECUTE RQ1 (Isotropy and CKA) ---
    with patch("src.probing.io_utils.load_hidden_states", return_value=mock_tensor) as mock_load_rq1:
        with patch.object(sys, "argv", ["run_rq1.py", "--config", "configs/config.yaml"]):
            run_rq1.main()
            
    assert mock_load_rq1.call_count > 0, "run_rq1.load_hidden_states was not invoked."
    assert (Path("results/rq1_emergence") / "isotropy_pythia.csv").exists(), \
        "run_rq1.main() failed to generate isotropy_pythia.csv"
    
    # --- EXECUTE RQ2 (Linear Probing) ---
    with patch("run_rq2.load_hidden_states", return_value=mock_tensor) as mock_load_rq2:
        with patch.object(sys, "argv", ["run_rq2.py", "--config", "configs/config.yaml"]):
            run_rq2.main()
            
    assert mock_load_rq2.call_count > 0, "run_rq2.load_hidden_states was not invoked."
    rq2_out_dir = Path("results/rq2_probing")
    accuracy_matrix = rq2_out_dir / "accuracy_matrix.csv"
    assert accuracy_matrix.exists(), "run_rq2.main() failed to generate accuracy_matrix.csv"
    
    with open(accuracy_matrix, "r", encoding="utf-8") as f:
        content = f.read()
        assert "sign" in content
        assert "parity" in content
        assert "layer_00" in content or "0" in content
    
    assert (rq2_out_dir / "weights" / "layer_01_sign.npy").exists(), "Probe weights not serialized correctly."

    # --- EXECUTE RQ3 (Dynamic Evaluation) ---
    ckpt_path = "data/processed/checkpoints_extracted/checkpoint-500"
    with patch("run_rq3.load_hidden_states", return_value=mock_tensor) as mock_load_rq3:
        with patch.object(sys, "argv", ["run_rq3.py", "--config", "configs/config.yaml", "--checkpoint_dir", ckpt_path]):
            run_rq3.main()
            
    assert mock_load_rq3.call_count > 0, "run_rq3.load_hidden_states was not invoked."
    
    csv_files = list(Path("results").rglob("*.csv"))
    drift_files = [f for f in csv_files if "drift" in f.name or "angle" in f.name or "rq3" in f.name]
    assert len(drift_files) > 0, "run_rq3.main() failed to generate geometric drift report."