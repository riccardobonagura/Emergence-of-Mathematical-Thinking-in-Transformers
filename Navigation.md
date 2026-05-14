# Project Navigation Guide

project_root/
├── configs/                  # YAML configuration files for experiments
├── data/                     # Data storage (Git-ignored)
├── dev_tools/                # Ad-hoc scripts and patches
├── results/                  # Generated artifacts, metrics, and figures
├── src/                      # Core modules and mathematical logic
├── tests/                    # TDD, hardware validation, and fixtures
├── run_rq1.py                # Orchestrator: Emergence Threshold (Iso, CKA, PCA+FLD)
├── run_rq2.py                # Orchestrator: Semantic Decoding (Linear Probing)
├── run_rq3.py                # Orchestrator: Dynamic Trajectory (Cross-temporal Drift)
├── reproduce.py              # CLI tool to rerun experiments from YAML hashes
└── requirements.txt          # Frozen Python dependencies