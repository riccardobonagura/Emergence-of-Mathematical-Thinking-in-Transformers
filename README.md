# Emergence of Mathematical Reasoning in Transformers

**Geometric Analysis of Representation Spaces and Fine-Tuning Dynamics** *Bachelor’s Thesis in Computer Engineering, University of Naples Federico II*

## Overview

Modern Large Language Models (LLMs) exhibit surprising capabilities in logical-mathematical reasoning, yet the internal mechanisms through which these abilities emerge remain largely opaque. This work applies **Mechanistic Interpretability** to investigate the geometric structure of internal representation spaces (hidden states). The goal is to understand exactly *where* and *how* mathematical reasoning takes shape within the layered hierarchy of a Transformer.

The project is structured into three distinct phases: a **Descriptive Phase** (topology), a **Decoding Phase** (semantic extraction), and a **Dynamic Phase** (cross-temporal learning trajectories). To ensure scientific validity, the codebase is built on a rigorous, hardware-agnostic, and CI/CD-validated software architecture.

## Research Questions

The study addresses three interconnected questions:

1. **Topological Emergence (RQ1):** Within the Transformer hierarchy, is there a specific layer threshold beyond which the internal representations of mathematical entities acquire a geometric structure distinguishable from that of generic language?
2. **Semantic Decoding (RQ2):** Are specific mathematical properties (such as *sign* and *parity*) linearly decodable from the residual stream? Can we isolate these directions without positional or magnitude confounds?
3. **Dynamic Trajectory (RQ3):** If model performance on mathematical reasoning benchmarks (GSM8K) improves through a fine-tuning cycle, does this improvement correspond to a measurable geometric drift (Frobenius norm) of internal embeddings? If so, in which layers is this drift concentrated?

---

## Project Architecture & Navigation

The codebase is designed as a modular, 5-layer Lego-like system, featuring centralized configuration, fail-fast tokenization guards, and CPU-only CI validation.

```text
project_root/
├── configs/                  # YAML configuration files (config.yaml, lora_config.yaml)
├── data/                     # Data storage (Git-ignored: raw, processed, extracted tensors)
├── dev_tools/                # Ad-hoc scripts, experimental patches, and scratchpads
├── results/                  # Generated artifacts (CSVs, JSONs, probe weights) and figures
├── src/                      # Core modules and mathematical logic
│   ├── config/               # Single source of truth (models.py registry, categories.py v5 schema)
│   ├── extraction/           # Hardware orchestration (extract_states.py, checkpoint_loop.py)
│   ├── finetuning/           # Training engines and validation (train_qlora.py, run_gsm8k.py)
│   ├── probing/              # Static representations & controls (io_utils.py, run_confound_checks.py)
│   ├── utils/                # Cross-domain guards (validate_configs.py, io_smoke_test.py)
│   └── viz/                  # Graphical projection engines (plot_rq3_dynamics.py, pca_umap_viz.py)
├── tests/                    # CPU-only CI validation, TDD, and mock fixtures (test_pipeline_e2e.py)
├── run_rq1.py                # Orchestrator: Emergence Threshold (Isotropy, CKA)
├── run_rq2.py                # Orchestrator: Semantic Decoding (Linear Probing)
├── run_rq3.py                # Orchestrator: Dynamic Trajectory (Cross-temporal Drift)
└── requirements.txt          # Frozen Python dependencies

```

---

## Pipeline & Methodology

### 1. Dataset Construction (v5 Schema)

A strictly typed, highly controlled dataset of stimuli divided into distinct categories: `CAT-SIGN`, `CAT-PARITY`, `CTRL-NUM`, and `CTRL-NEU`.

* **Property-Contrastive Pairs:** Stimuli are generated in pairs that differ by only one target element (e.g., $a - b$ vs $b - a$). This isolates the mathematical structure and defends against dataset artifacts.
* **Confound Mitigation:** Sentinel values (`-1`) and dedicated magnitude control probes ensure the model learns actual arithmetic properties rather than simple positional heuristics (N-01/N-02 bugs).

### 2. Descriptive Phase (RQ1)

Hidden states are extracted on bare metal using `TransformerLens`. The representation space is analyzed through:

* **Isotropy:** Measured as the average cosine similarity between random vector pairs to detect measure concentration.
* **Centered Kernel Alignment (CKA):** Producing a layer × layer heatmap to map the flow of topological similarity across the network.

### 3. Decoding Phase (RQ2)

For each layer, robust linear probing classifiers (Logistic Regression) are trained on specific mathematical targets.

* The output is the central curve of the study: *Probing Accuracy vs. Layer Depth*.
* The weights of optimal probes are extracted and interpreted as linear direction angles in the high-dimensional vector space.

### 4. Dynamic Phase (RQ3)

A fine-tuning cycle utilizing **QLoRA** on the *MetaMathQA* dataset is orchestrated with regular checkpointing.

* **Checkpoint Loop:** At each interval, adapter weights are merged into the base model.
* **Benchmarking:** External validation is performed via EleutherAI's `lm-evaluation-harness` to measure 0-shot accuracy on **GSM8K**.
* **Drift Calculation:** The geometric metrics are recomputed, correlating the Frobenius norm drift of the representations with the actual cognitive improvements of the LLM.

---

## Mathematical Tools

* **Applied Linear Algebra:** Dot products, cosine similarity, PCA, Gram matrix algebra, and Frobenius norms.
* **Statistics:** Logistic regression, Scikit-learn solver optimization (L-BFGS), generalization bounding, and bootstrap confidence intervals.
* **Kernel Methods:** Centered Kernel Alignment formulated as a normalized inner product between centered Gram matrices.

## Candidate Models

The primary focus of the case study is on **EleutherAI/pythia-1.4b** due to its stable architectural footprint and transparent training lineage.
Thanks to the custom `ModelRegistry` architecture, the software is modular and natively pre-configured to scale experiments to different tokenizer behaviors and structures, including:

* *Mistral-7B*
* *Gemma-2-2B*
* *Phi-2*

## Core Tech Stack

* **Extraction & Interpretability:** `TransformerLens`
* **Fine-tuning & Memory:** `HuggingFace Transformers`, `PEFT`, `bitsandbytes` (NF4 Quantization)
* **Mathematical Core:** `PyTorch`, `scikit-learn`, `NumPy`
* **Validation & External Evals:** `pytest`, `lm-evaluation-harness`
