"""
io_smoke_test.py — NVMe Controller / WSL I/O Stress Test (T05).
Validates parallel disk read capabilities to prevent bottlenecking or crashes
during the massive concurrent probing evaluation (run_rq2.py).
"""

import sys
import time
import json
import logging
import tempfile
from pathlib import Path

import torch
import numpy as np
from joblib import Parallel, delayed

def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("io_smoke_test")

def worker_read(file_path: Path) -> float:
    """Reads a tensor from disk and returns the latency in seconds."""
    start_time = time.perf_counter()
    # Read the tensor; weights_only=True ensures safe unpickling
    tensor = torch.load(file_path, map_location="cpu", weights_only=True)
    # Trivial operation to force materialization in RAM
    _ = tensor.shape
    end_time = time.perf_counter()
    return end_time - start_time

def main() -> None:
    logger = setup_logger()
    
    # Hardcoded test parameters per T05 specifications
    N_FILES = 48
    N_WORKERS = 16
    TENSOR_SHAPE = (3000, 2048)
    
    logger.info(f"Initiating I/O Smoke Test (N_FILES={N_FILES}, N_WORKERS={N_WORKERS})")
    
    bytes_per_tensor = TENSOR_SHAPE[0] * TENSOR_SHAPE[1] * 2  # FP16 = 2 bytes
    total_volume_gb = (N_FILES * bytes_per_tensor) / 1e9

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        file_paths = []
        
        # 1. Generate Synthetic FP16 Tensors
        logger.info(f"Generating {total_volume_gb:.2f} GB of synthetic FP16 tensors in tmp_path...")
        for i in range(N_FILES):
            t_path = tmp_path / f"synth_layer_{i:02d}.pt"
            dummy_tensor = torch.randn(*TENSOR_SHAPE, dtype=torch.float16)
            torch.save(dummy_tensor, t_path)
            file_paths.append(t_path)
            
        logger.info("Generation complete. Launching parallel read stress test...")
        
        # 2. Parallel Read Stress Test
        start_total = time.perf_counter()
        
        # Using loky backend to accurately simulate run_rq2 multiprocess isolation
        read_latencies = Parallel(n_jobs=N_WORKERS, backend="loky")(
            delayed(worker_read)(p) for p in file_paths
        )
        
        end_total = time.perf_counter()
        total_time = end_total - start_total

    # 3. Metrics Calculation
    median_ms = float(np.median(read_latencies) * 1000)
    throughput_gbps = total_volume_gb / total_time
    
    logger.info(f"Test finished in {total_time:.2f} seconds.")
    logger.info(f"Aggregate Throughput: {throughput_gbps:.3f} GB/s")
    logger.info(f"Median Read Latency:  {median_ms:.2f} ms")

    # 4. Threshold Logic & Recommendations
    if median_ms < 200:
        status = "OK"
        recommendation = "n_jobs: -1 confermato"
        logger.info(f"[OK] {recommendation}. NVMe controller is handling the parallelism flawlessly.")
        exit_code = 0
    elif 200 <= median_ms < 500:
        status = "WARN"
        recommendation = "n_jobs: 8 consigliato"
        logger.warning(f"[WARN] Elevated latency detected. {recommendation} in configs/config.yaml.")
        exit_code = 0
    else:
        status = "FAIL"
        recommendation = "n_jobs: 4 obbligatorio"
        logger.error(f"[FAIL] Severe I/O throttling detected. {recommendation}.")
        logger.error("-> INSTRUCTION: Open configs/config.yaml and set 'n_jobs: 4' before running T06.")
        exit_code = 1

    # 5. Save Artifact
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "io_smoke_test.json"
    
    payload = {
        "median_ms": median_ms,
        "throughput_gbps": throughput_gbps,
        "n_workers": N_WORKERS,
        "status": status,
        "recommendation": recommendation
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        
    logger.info(f"Report saved to {out_file}")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()