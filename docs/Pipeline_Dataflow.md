---
config:
  layout: elk
  flowchart:
    curve: basis
    nodeSpacing: 45
    rankSpacing: 55
---
flowchart LR
    subgraph RQ1 ["<span style='background:#1f2937; color:#fff; padding:6px 12px; border-radius:6px; font-size:14px;'><b>RQ1 — emergence geometry (two distinct metrics)</b></span>"]
        direction TB
        ISO["ΔIso per category<br/>rq1_emergence/isotropy_aggregated_balanced.csv<br/><i>ΔIso ≈ −0.1056 @ L15 · floor≈0 in d=2048 (E-G-01)</i>"]
        CKA["CKA — two metrics<br/>rq1_emergence/cka_results_annotated.csv<br/><i>evolutionary (l vs l−1) @ L23 = 0.9257 / 0.8865<br/>inter-category (math↔ctrl) @ L23 ≈ 0.010 ≈ matched baselines</i>"]
    end

    subgraph RQ2 ["<span style='background:#1f2937; color:#fff; padding:6px 12px; border-radius:6px; font-size:14px;'><b>RQ2 — linear probing (two probes, two confounds)</b></span>"]
        direction TB
        SIGN["Sign probe<br/>rq2_probing/accuracy_metrics_corrected.csv<br/><i>emerge L0 · peak 1.000 @ L3</i>"]
        PAR["Parity probe<br/>same CSV<br/><i>emerge L6 · peak 0.990 @ L15 · jump L12→L13</i>"]
        SCONF["Sign N-01 confound<br/>rq2_probing/confound_checks_hardened.csv<br/><i>logit↔op1 r 0.55–0.67 ⚠</i>"]
        PCONF["Parity N-02 confound<br/>rq2_probing/parity_confound_checks.csv<br/><i>pred↔op1 r=0.090 · logit↔op2-parity r≤0.199 ✓ clean</i>"]
        WTS["Frozen probe weights (w,b)<br/>rq2_probing/weights/*.npy + rq2_config_hash.json"]
    end

    subgraph RQ3 ["<span style='background:#1f2937; color:#fff; padding:6px 12px; border-radius:6px; font-size:14px;'><b>RQ3 — dynamics of the RQ1 geometry across fine-tuning</b></span>"]
        direction TB
        R3CSV["RQ1 geometry recomputed per checkpoint<br/>rq3_ft_dynamics/rq3_dynamics.csv<br/><i>ΔIso, inter-cat CKA, cross-temporal CKA(base→ckpt)<br/>L9 1−CKA only 0.054 (quasi-rigid) · brightest at L23 final step</i>"]
    end

    subgraph RQ4 ["<span style='background:#1f2937; color:#fff; padding:6px 12px; border-radius:6px; font-size:14px;'><b>RQ4 — geometric reorganization under QLoRA (drift + frozen-probe decay)</b></span>"]
        direction TB
        R4ACC["Frozen-probe acc trajectory<br/>rq4_drift/trajectories_probing.csv<br/><i>sign 0.994 → 0.739 · L9 holds (sign 0.995) · parity L20 = 0.965</i>"]
        R4DRIFT["Dual Frobenius drift<br/>same CSV<br/><i>rel math max 0.690 @ L9 · math 1.86× ctrl · NF4 floor 0.153 · SNR 4.51×</i>"]
    end

    subgraph RQ5 ["<span style='background:#1f2937; color:#fff; padding:6px 12px; border-radius:6px; font-size:14px;'><b>RQ5 — behavioral determinization at '=' (inference-only)</b></span>"]
        direction TB
        R5DET["Determinization metrics per (step, category)<br/>rq5_determinization/determinization.csv<br/><i>entropy ↓ · top1−top2 margin ↑ · P(answer) ↑</i>"]
        R5SINGLE["Single-token-restricted entropy/margin<br/>same CSV (*_single columns)<br/><i>isolates CAT-SIGN digit half from sign-token half (B6)</i>"]
    end

    MODEL["<b>Pythia-1.4B (base)</b><br/>24 layers · d_model=2048"] -- "tokenize stimuli → forward pass →<br/>gather '=' terminal token h_l" --> EXT["<b>Base hidden states</b><br/>data/processed/pythia-1.4b/layer_00..23.pt · FP16 [n,2048]<br/>+ metadata.json"]
    DS["<b>Dataset</b><br/>4 cats · 3000 stimuli<br/>CAT-SIGN | CAT-PARITY<br/>CTRL-NEU | CTRL-NUM"] -- "stimuli text + category labels" --> EXT
    EXT -- "mean off-diagonal cosine sim<br/>cos = ⟨ĥ_i, ĥ_j⟩ over i≠j<br/>(exact Gram / Monte-Carlo) + bootstrap CI" --> ISO
    EXT -- "linear CKA = HSIC(K,L) / √(HSIC(K,K)·HSIC(L,L))<br/>K=XXᵀ, L=YYᵀ centered: K_c = H·K·H" --> CKA
    EXT -- "StandardScaler → LogReg(L2, C)<br/>label = sign · acc + bootstrap CI<br/>+ 5-fold permutation test" --> SIGN
    EXT -- "StandardScaler → LogReg(L2, C)<br/>label = parity · acc + bootstrap CI<br/>+ 5-fold permutation test" --> PAR
    SIGN -- "Pearson r( sign_logit , operand₁ )" --> SCONF
    PAR -- "Pearson r( parity_logit , op2-parity )" --> PCONF
    SIGN -- "denormalize: w_orig=w/σ, b_orig=b−w·μ" --> WTS
    PAR -- "denormalize: w_orig=w/σ, b_orig=b−w·μ" --> WTS
    MODEL -- "QLoRA NF4 · r=16 · QKV-only<br/>(MLP frozen) · ~3.1M trainable" --> FT["<b>QLoRA checkpoints</b><br/>data/processed/checkpoints/<br/>2500/5000/7500/10000 + final_adapter<br/><i>step 12343 · 1 epoch · train loss ≈ 2.58</i>"]
    MMQA["<b>MetaMathQA</b><br/>fine-tuning corpus"] -- "causal-LM SFT · effective batch 32" --> FT
    FT -- "merge adapter into base →<br/>re-extract '=' token states" --> CKEXT["<b>Re-extracted checkpoint states</b><br/>data/processed/checkpoints_extracted/<br/>5 checkpoints × 24 layers"]

    %% RQ3 — FT geometry dynamics (recompute RQ1 metrics across checkpoints)
    EXT -- "H_base reference (cross-temporal CKA)" --> R3CSV
    CKEXT -- "reuse isotropy_exact + linear_cka /<br/>compute_cka_intercategory (no refit)" --> R3CSV
    R3CSV --> VIZ3["RQ3 FT-dynamics dashboard<br/>results/figures/rq3/rq3_ft_dynamics.html"]
    GSM -- "descriptive overlay only (n=6)" --> VIZ3

    %% RQ4 — drift + frozen-probe decay + GSM8K + NF4
    WTS -- "apply frozen probe (NO refit):<br/>ŷ = 1 if X·w + b > 0 (hash-verified)" --> R4ACC
    CKEXT -- "X_test = checkpoint states[test_idx]" --> R4ACC
    CKEXT -- "‖H_ckpt − H_base‖_F  ÷(N·d)  and  ÷‖H_base‖_F<br/>(dim-normalized + relative, math & ctrl)" --> R4DRIFT
    EXT -- "H_base reference" --> R4DRIFT
    R4ACC --> TRAJ["<b>Unified trajectory CSV</b><br/>rq4_drift/trajectories_probing.csv<br/>(+ gsm8k_acc, gsm8k_ci_*)"]
    R4DRIFT --> TRAJ
    GSM -- "merge gsm8k_acc + CI per step" --> TRAJ
    TRAJ --> VIZ4["RQ4 drift/trajectory dashboard<br/>results/figures/rq4/rq4_dashboard.html"]
    NF4 -- "degradation disclosure (floor for SNR)" --> VIZ4

    %% RQ5 — behavioral determinization at '='
    MODEL -- "step 0 = un-merged base (no adapter)<br/>logits at '=' · float32 softmax" --> R5DET
    FT -- "merge adapter → HookedTransformer (fp16, fold_ln)<br/>forward → logits[:, '=' idx, :] per step" --> R5DET
    DS -- "math rows only (CAT-SIGN, CAT-PARITY)<br/>target = first continuation token after '='" --> R5DET
    R5DET -- "mask to single-token results<br/>(1500/2000; 500 negatives = sign token ' -')" --> R5SINGLE

    %% Shared baselines
    MODEL -- "bf16-ref vs double-quant NF4, per layer:<br/>rel Frobenius ‖Δ‖_F/‖ref‖_F + mean cosine" --> NF4["NF4 degradation baseline (T16)<br/>nf4_degradation/summary.json<br/><i>mean rel Frob 0.153 · cos 0.98–0.999 · SNR 4.51×</i>"]
    FT -- "0-shot generate → exact-match acc<br/>+ bootstrap CI (per checkpoint)" --> GSM["GSM8K 0-shot accuracy<br/>gsm8k/gsm8k_*.json<br/><i>0.000 → 0.099 → 0.136 → 0.152 → 0.161 → 0.171<br/>(step 0 → 12343 · reaches 17.1% from 0 baseline, monotonic)</i>"]

    %% Visualization fan-out
    ISO --> VIZ1["RQ1 emergence dashboard<br/>results/figures/rq1_emergence/"]
    CKA --> VIZ1
    SIGN --> VIZ2["RQ2 linear probe dashboard<br/>results/figures/rq2/"]
    PAR --> VIZ2
    SCONF -- "effect size vs significance bars (E-M-03)" --> VIZ2
    PCONF -- "effect size vs significance bars (E-M-03)" --> VIZ2
    CKA -- "math-vs-ctrl 2-class PCA @ L23<br/>(separation tracks terminal-'=' axis, not math)" --> VIZ2
    R5DET --> VIZ5["RQ5 determinization dashboard<br/>results/figures/rq5/rq5_determinization.html"]
    R5SINGLE --> VIZ5
    GSM -- "descriptive overlay only (n=6)" --> VIZ5

    %% ───────────────────────── Class Attachments ─────────────────────────
    ISO:::res
    CKA:::res
    SIGN:::res
    PAR:::res
    SCONF:::warn
    PCONF:::res
    WTS:::data
    R3CSV:::res
    R4ACC:::res
    R4DRIFT:::res
    R5DET:::res
    R5SINGLE:::res
    MODEL:::src
    EXT:::data
    DS:::src
    FT:::data
    MMQA:::src
    CKEXT:::data
    NF4:::res
    GSM:::res
    TRAJ:::data
    VIZ1:::viz
    VIZ2:::viz
    VIZ3:::viz
    VIZ4:::viz
    VIZ5:::viz

    %% ───────────────────────── Styling ─────────────────────────
    classDef src  fill:#1f2937,stroke:#9ca3af,color:#f9fafb,stroke-width:1px
    classDef data fill:#0e7490,stroke:#67e8f9,color:#f0fdff,stroke-width:1px
    classDef res  fill:#155e75,stroke:#22d3ee,color:#ecfeff,stroke-width:1px
    classDef warn fill:#7c2d12,stroke:#fb923c,color:#fff7ed,stroke-width:1px
    classDef viz  fill:#4c1d95,stroke:#c4b5fd,color:#f5f3ff,stroke-width:1px

    style RQ1 fill:#FFBD59,stroke:#fdba74,color:#000000
    style RQ2 fill:#FFBD59,stroke:#fdba74,color:#000000
    style RQ3 fill:#FFBD59,stroke:#fdba74,color:#000000
    style RQ4 fill:#FFBD59,stroke:#fdba74,color:#000000
    style RQ5 fill:#FFBD59,stroke:#fdba74,color:#000000
