#!/usr/bin/env python3
"""oracolo.py — Il Banco della Pizia.

Single-file orchestrator: every entrypoint catalogued in docs/RECON.md §2 is
exposed through a menu, run as a subprocess, parsed, and reported. The hardcoded
REGISTRY is the contract — no runtime re-discovery. Stdlib only; rich optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

try:
    import resource  # POSIX peak-RSS
except ImportError:  # pragma: no cover
    resource = None
try:
    import yaml  # project dep; flat fallback below if absent
except ImportError:  # pragma: no cover
    yaml = None
try:
    from rich.console import Console
    from rich.table import Table
    _RICH = Console()
except ImportError:  # pragma: no cover
    _RICH = None

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG_DIR = REPO_ROOT / "logs" / "oracolo"
Cost = Literal["fast", "medium", "long", "very_long"]

# ─── Theme strings ──────────────────────────────────────────────────────────

LANG = "it"
USE_COLOR = True

TRIPOD = r"""
        _   _   _
       | | | | | |
       | |_| |_| |
        \  PIZIA /
         \__|__/
          /_|_\
"""

DELPHIC = {
    "banner": "« Parla, pellegrino. La Pizia ascolta. »",
    "start": ["« Il fumo si leva dall'àdyton. »",
              "« La Pizia inala i vapori e china il capo. »",
              "« Conosci te stesso, e conoscerai la macchina. »",
              "« Nulla di troppo: solo ciò che chiedi. »"],
    "end_ok": ["« La Pizia ha parlato. »", "« L'oracolo tace. Interpreta tu. »",
               "« È compiuto. Misura, non credere. »"],
    "end_fail": ["« Gli dèi tacciono. Il responso è oscuro. »",
                 "« Il tripode trema: qualcosa si è rotto. »"],
}
_ANSI = {"dim": "\033[2m", "bold": "\033[1m", "red": "\033[31m", "green": "\033[32m",
         "yellow": "\033[33m", "cyan": "\033[36m", "magenta": "\033[35m", "reset": "\033[0m"}

def c(text: str, *styles: str) -> str:
    if not USE_COLOR:
        return text
    pre = "".join(_ANSI.get(s, "") for s in styles)
    return f"{pre}{text}{_ANSI['reset']}" if pre else text

def t(it: str, en: str) -> str:
    return it if LANG == "it" else en

def say(line: str) -> None:
    print(c(line, "magenta", "bold"))

def info(msg: str) -> None:
    print(c("· ", "dim") + msg)

def warn(msg: str) -> None:
    print(c("⚠ ", "yellow") + c(msg, "yellow"))

def err(msg: str) -> None:
    print(c("✗ ", "red") + c(msg, "red"))

# ─── Data model ─────────────────────────────────────────────────────────────

class Category(Enum):
    SETUP = ("Allestimento", "🛠")
    DATASET = ("Dataset", "𝝳")
    EXTRACTION = ("Estrazione", "🜍")
    RQ1 = ("RQ1 · geometria", "△")
    RQ2 = ("RQ2 · probing", "⊕")
    FINETUNING = ("Fine-tuning", "🜚")
    RQ3 = ("RQ3 · deriva", "⇌")
    RQ4 = ("RQ4 · determinazione", "⚖")
    EVAL = ("Valutazione", "𝛴")
    VIZ = ("Visualizzazioni", "◐")
    TESTS = ("Test", "✓")
    UTILS = ("Utilità", "·")

    def __init__(self, label: str, glyph: str) -> None:
        self.label = label
        self.glyph = glyph

@dataclass(frozen=True)
class Entrypoint:
    key: str
    title: str
    title_en: str
    category: Category
    command: list[str]
    required_args: list[str] = field(default_factory=list)
    optional_args: dict[str, str] = field(default_factory=dict)
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    gpu_required: bool = False
    cost: Cost = "fast"
    description: str = ""

    @property
    def name(self) -> str:
        return self.title if LANG == "it" else self.title_en

# Defaults for required flags. Flags absent here (--checkpoint_dir, --model_path,
# --tag) must come from the user or a composite override.
RESOLVERS: dict[str, str] = {
    "--config": "configs/config_rq2.yaml", "--lora_config": "configs/lora_config.yaml",
    "--inputs": "data/raw/stimuli_arithmetic_v5.jsonl",
    "--output": "data/processed/dataset_master_v5.jsonl",
}

# ─── REGISTRY — 28 entrypoints, values verbatim from RECON §2 ────────────────

P = Path
D = P("data/processed")
R = P("results")
DS = D / "dataset_master_v5.jsonl"
BASE = D / "pythia-1.4b"
R2 = R / "rq2_probing"
CFG = P("configs/config_rq2.yaml")
LORA = P("configs/lora_config.yaml")
CAT = Category

REGISTRY: list[Entrypoint] = [
    Entrypoint("chk-iface", "Prova d'interfaccia", "Interface smoke check", CAT.SETUP,
        [PY, "tests/check_interface.py"], gpu_required=True, cost="medium",
        description="Loads Pythia via TransformerLens; checks hidden-state extraction (NOT CPU-only)."),
    Entrypoint("build-stim", "Conio degli stimoli", "Generate stimuli", CAT.DATASET,
        [PY, "-m", "src.dataset.build_stimuli"],
        optional_args={"--n_pairs": "500", "--seed": "42",
                       "--tokenizer": "EleutherAI/pythia-1.4b",
                       "--output": "data/raw/stimuli_arithmetic_v5.jsonl"},
        outputs=[P("data/raw/stimuli_arithmetic_v5.jsonl")],
        description="Generate arithmetic minimal-pair stimuli."),
    Entrypoint("merge-stim", "Fusione degli stimoli", "Merge stimuli", CAT.DATASET,
        [PY, "-m", "src.dataset.merge_stimuli"], required_args=["--inputs", "--output"],
        optional_args={"--allow-untokenized": ""},
        inputs=[P("data/raw/stimuli_arithmetic_v5.jsonl")], outputs=[DS],
        description="Merge + tokenize-validate stimuli into the master JSONL."),
    Entrypoint("regen", "Rigenerazione del dataset", "Regenerate dataset", CAT.DATASET,
        [PY, "-m", "src.dataset.regenerate_dataset"],
        optional_args={"--n_pairs": "500", "--n_control": "500", "--seed": "42"},
        outputs=[D / "dataset_master_v5_regenerated.jsonl"],
        description="Full dataset regen orchestrator (optional re-extract/RQ2/confound)."),
    Entrypoint("ds-test", "Vaglio del dataset", "Dataset integrity test", CAT.DATASET,
        [PY, "-m", "src.dataset.test_dataset"], inputs=[DS],
        description="Standalone dataset integrity assertions."),
    Entrypoint("extract", "Cattura del residuo", "Extract hidden states", CAT.EXTRACTION,
        [PY, "-m", "src.extraction.extract_states"], required_args=["--config"],
        inputs=[CFG, DS], outputs=[BASE], gpu_required=True, cost="long",
        description='Forward-pass, gather "=" terminal token at hook_resid_post.'),
    Entrypoint("loop", "Pellegrinaggio dei checkpoint", "Checkpoint loop", CAT.EXTRACTION,
        [PY, "-m", "src.extraction.checkpoint_loop"], required_args=["--config"],
        inputs=[D / "checkpoints", DS], outputs=[D / "checkpoints_extracted"],
        gpu_required=True, cost="very_long",
        description="Merge adapter -> re-extract -> run_rq3 per checkpoint."),
    Entrypoint("rq1", "Geometria dell'emergenza", "RQ1 emergence geometry", CAT.RQ1,
        [PY, "run_rq1.py"], required_args=["--config"], inputs=[BASE],
        outputs=[R / "rq1_emergence", R / "rq1_emergence/cka_intercategory.npy"], cost="medium",
        description="DeltaIso + evolutionary/inter-category CKA + reviewer baselines."),
    Entrypoint("rq1-dyn", "Geometria nel tempo", "RQ1 dynamics (suppl.)", CAT.RQ1,
        [PY, "run_rq1_dynamics.py"], required_args=["--config"],
        inputs=[BASE, D / "checkpoints_extracted"],
        outputs=[R / "rq1_emergence/dynamic/rq1_dynamics.csv"], cost="medium",
        description="Recompute RQ1 geometry + cross-temporal CKA per checkpoint."),
    Entrypoint("cka-main", "Dimostrazione CKA", "CKA self-demo", CAT.RQ1,
        [PY, "-m", "src.metrics.cka"], description="Standalone CKA __main__ self-demo."),
    Entrypoint("iso-main", "Dimostrazione isotropia", "Isotropy self-demo", CAT.RQ1,
        [PY, "-m", "src.metrics.isotropy"], description="Standalone isotropy __main__ self-demo."),
    Entrypoint("rq2", "Decifrazione lineare", "RQ2 linear probing", CAT.RQ2,
        [PY, "run_rq2.py"], required_args=["--config"], inputs=[BASE, DS],
        outputs=[R2 / "accuracy_metrics_corrected.csv", R2 / "weights", R2 / "test_indices",
                 R2 / "weights/rq2_config_hash.json"], cost="medium",
        description="48 logistic probes (joblib), bootstrap+perm+BH, frozen weights."),
    Entrypoint("confound-sign", "Confondente N-01 (segno)", "Sign confound N-01", CAT.RQ2,
        [PY, "-m", "src.probing.run_confound_checks"], required_args=["--config"],
        inputs=[R2 / "weights", R2 / "test_indices", DS, BASE],
        outputs=[R2 / "confound_checks_hardened.csv"], cost="medium",
        description="N-01: sign-vs-operand1 magnitude triangulation."),
    Entrypoint("confound-par", "Confondente N-02 (parità)", "Parity confound N-02", CAT.RQ2,
        [PY, "-m", "src.probing.run_parity_confound_checks"], required_args=["--config"],
        inputs=[R2 / "weights", R2 / "test_indices", DS, BASE],
        outputs=[R2 / "parity_confound_checks.csv"], cost="medium",
        description="N-02: parity-vs-operand2 parity triangulation."),
    Entrypoint("train", "Forgia QLoRA", "QLoRA fine-tuning", CAT.FINETUNING,
        [PY, "-m", "src.finetuning.train_qlora"], required_args=["--config", "--lora_config"],
        inputs=[CFG, LORA], outputs=[D / "checkpoints"], gpu_required=True, cost="very_long",
        description="QLoRA NF4 r=16 QKV-only, 1 epoch on MetaMathQA."),
    Entrypoint("rq3", "Deriva di Frobenius", "RQ3 drift trajectory", CAT.RQ3,
        [PY, "run_rq3.py"], required_args=["--config", "--checkpoint_dir"],
        inputs=[BASE, R2 / "weights"], outputs=[R2 / "dynamic/trajectories_probing.csv"],
        description="Frozen-probe acc + dual Frobenius drift per step."),
    Entrypoint("rq4", "Determinazione al segno '='", "RQ4 determinization", CAT.RQ4,
        [PY, "run_rq4.py"], required_args=["--config"], inputs=[CFG, DS],
        outputs=[R / "rq4_determinization/determinization.csv"], gpu_required=True, cost="long",
        description='"=" next-token determinization (entropy/margin/P(ans)).'),
    Entrypoint("gsm8k", "Prova di GSM8K", "GSM8K 0-shot eval", CAT.EVAL,
        [PY, "-m", "src.eval.eval_gsm8k"], required_args=["--model_path", "--tag", "--config"],
        optional_args={"--loading_strategy": "peft"}, inputs=[CFG], outputs=[R / "gsm8k"],
        gpu_required=True, cost="long",
        description="0-shot GSM8K via lm_eval + Wald CI; appends to trajectory."),
    Entrypoint("nf4", "Degradazione NF4 (T16)", "NF4 degradation (T16)", CAT.EVAL,
        [PY, "-m", "src.eval.nf4_degradation"], required_args=["--config"], inputs=[CFG, DS],
        outputs=[R / "nf4_degradation/summary.json", R / "nf4_degradation/per_layer_stats.csv"],
        gpu_required=True, cost="medium",
        description="T16 FP16-vs-NF4 native-HF-hook degradation baseline."),
    Entrypoint("viz-rq1", "Cruscotto RQ1", "RQ1 dashboard", CAT.VIZ,
        [PY, "-m", "src.viz.plot_rq1_emergence"], inputs=[R / "rq1_emergence"],
        outputs=[R / "figures/rq1_emergence/rq1_emergence.html"],
        description="DeltaIso + evolutionary CKA dashboard."),
    Entrypoint("viz-rq2", "Cruscotto RQ2", "RQ2 dashboard", CAT.VIZ,
        [PY, "-m", "src.viz.plot_rq2_probing"],
        optional_args={"--results_dir": "results/rq2_probing", "--out_dir": "results/figures/rq2"},
        inputs=[R2 / "accuracy_metrics_corrected.csv"],
        outputs=[R / "figures/rq2/accuracy_curves.html"],
        description="Probe accuracy curves + effect-size bars."),
    Entrypoint("viz-rq3", "Cruscotto RQ3", "RQ3 dashboard", CAT.VIZ,
        [PY, "-m", "src.viz.plot_rq3_trajectory"],
        inputs=[R2 / "dynamic/trajectories_probing.csv"],
        outputs=[R / "figures/rq3/rq3_dashboard.html"],
        description="3-panel trajectory/drift dashboard."),
    Entrypoint("viz-rq4", "Cruscotto RQ4", "RQ4 dashboard", CAT.VIZ,
        [PY, "-m", "src.viz.plot_rq4_determinization"],
        inputs=[R / "rq4_determinization/determinization.csv"],
        outputs=[R / "figures/rq4/rq4_determinization.html"],
        description="RQ4 determinization dashboard."),
    Entrypoint("viz-supp", "Cruscotto supplementare", "Supplementary dashboard", CAT.VIZ,
        [PY, "-m", "src.viz.plot_ft_geometry_dynamics"],
        inputs=[R / "rq1_emergence/dynamic/rq1_dynamics.csv"],
        outputs=[R / "figures/supplementary_ft_dynamics.html"],
        description="Supplementary FT-geometry dashboard."),
    Entrypoint("viz-pca", "Proiezione PCA/UMAP", "PCA/UMAP scatter", CAT.VIZ,
        [PY, "-m", "src.viz.pca_umap_viz"], optional_args={"--layers": "23", "--reducer": "pca"},
        inputs=[BASE], outputs=[R / "figures/pca"],
        description="PCA/UMAP 2D/4-way category scatter."),
    Entrypoint("gen-fix", "Conio delle fixture", "Generate fixtures", CAT.TESTS,
        [PY, "tests/generate_fixtures.py"], description="Generate CPU test fixtures."),
    Entrypoint("validate", "Vaglio delle config", "Validate configs", CAT.UTILS,
        [PY, "-m", "src.utils.validate_configs"],
        optional_args={"--probing": "configs/config_rq2.yaml", "--lora": "configs/lora_config.yaml"},
        description="Validate config/lora targets + extraction weights dir."),
    Entrypoint("io-smoke", "Prova di scrittura atomica", "IO smoke test", CAT.UTILS,
        [PY, "-m", "src.utils.io_smoke_test"], required_args=["--config"], inputs=[CFG],
        description="Multi-core atomic-IO stress test."),
]
BY_KEY: dict[str, Entrypoint] = {e.key: e for e in REGISTRY}

# ─── Composite rites ────────────────────────────────────────────────────────

# Each step: (key, overrides). overrides may carry "__flags__" (raw store_true flags).
Step = tuple[str, dict]
CKPTS = ["2500", "5000", "7500", "10000"]
RITES: dict[str, list[Step]] = {
    "cammino_completo": (
        [("build-stim", {}), ("merge-stim", {}), ("extract", {}), ("validate", {}),
         ("rq1", {}), ("rq2", {}), ("confound-sign", {}), ("confound-par", {}),
         ("nf4", {}), ("train", {}), ("loop", {}),
         ("gsm8k", {"--tag": "baseline", "--model_path": "EleutherAI/pythia-1.4b",
                    "--loading_strategy": "merged_direct"})]
        + [("gsm8k", {"--tag": f"ckpt_{s}",
                      "--model_path": f"data/processed/checkpoints/checkpoint-{s}",
                      "--loading_strategy": "peft"}) for s in CKPTS]
        + [("gsm8k", {"--tag": "final", "--model_path": "AUTO_FINAL", "--loading_strategy": "peft"}),
           ("rq4", {}), ("rq1-dyn", {}), ("viz-rq1", {}), ("viz-rq2", {}),
           ("viz-rq3", {}), ("viz-rq4", {}), ("viz-supp", {})]
    ),
    "solo_probing": [("rq2", {}), ("confound-sign", {}), ("confound-par", {}), ("viz-rq2", {})],
    "solo_geometria": [("rq1", {}), ("viz-rq1", {})],
    "solo_rq4": [("rq4", {}), ("viz-rq4", {})],
    "solo_viz": [("viz-rq1", {}), ("viz-rq2", {}), ("viz-rq3", {}),
                 ("viz-rq4", {}), ("viz-supp", {}), ("viz-pca", {})],
    "smoke_test": [("__pytest__", {})],
    "dataset_regen": [("regen", {"__flags__": ["--with-extraction", "--with-rq2", "--with-confounds"]})],
}
RITE_DESC = {
    "cammino_completo": t("Pipeline a freddo, completa (GPU, lunghissima).", "Full cold-start pipeline (GPU)."),
    "solo_probing": t("Ri-sonda dai tensori base (CPU).", "Re-probe from base tensors (CPU)."),
    "solo_geometria": t("RQ1 dai tensori base (CPU).", "RQ1 from base tensors (CPU)."),
    "solo_rq4": t("RQ4 da modello + checkpoint (GPU).", "RQ4 from model + checkpoints (GPU)."),
    "solo_viz": t("Ri-disegna tutti i cruscotti (CPU).", "Re-render all dashboards (CPU)."),
    "smoke_test": "pytest tests/ -q (CPU).",
    "dataset_regen": t("Ricostruisce dataset + downstream.", "Rebuild dataset + downstream."),
}

# ─── Drift hooks (RECON §5 — D2/D3/D6/D8/D11) ───────────────────────────────

def _flat_yaml(path: Path) -> dict:
    """Top-level scalar keys via PyYAML, else a flat regex parser."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    out: dict = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_][\w]*):\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).split("#", 1)[0].strip().strip('"\'')
    return out

def _argv_value(argv: list[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default

def drift_note(entry: Entrypoint) -> None:
    """D2/D3 — informational notes before build."""
    if entry.key == "rq2":
        warn("drift D2: " + t("la config monta C=1.0; la spec dichiara 10.0.",
                              "config ships C=1.0; spec asserts 10.0."))
    if entry.key == "train":
        msl = _flat_yaml(REPO_ROOT / RESOLVERS["--lora_config"]).get("max_seq_length")
        if msl is not None and str(msl) != "1024":
            warn(f"drift D3: max_seq_length={msl} — " + t("la spec §7 dice 1024.", "spec §7 says 1024."))
        elif msl is not None:
            info(f"max_seq_length={msl}")

def drift_block_d6(entry: Entrypoint, argv: list[str]) -> bool:
    """D6 — refuse if total_training_steps absent. Loud even under --yes."""
    if entry.key not in {"rq3", "rq4", "rq1-dyn", "gsm8k"}:
        return False
    cfg = _argv_value(argv, "--config", RESOLVERS["--config"])
    if "total_training_steps" not in _flat_yaml(REPO_ROOT / cfg):
        err("drift D6: " + t(
            f"chiave 'total_training_steps' assente in {cfg} — gli script userebbero silenziosamente 2000 ≠ 12343 canonico. Aggiungi la chiave o passa una config esplicita.",
            f"key 'total_training_steps' absent in {cfg} — scripts would silently default to 2000 != canonical 12343. Add the key or pass an explicit config."))
        return True
    return False

def drift_resolve_d8(entry: Entrypoint, overrides: dict, ctx: "Ctx") -> None:
    """D8 — resolve terminal-checkpoint dir name (final_checkpoint vs final_adapter)."""
    if entry.key != "gsm8k":
        return
    if overrides.get("--tag") != "final" and overrides.get("--model_path") != "AUTO_FINAL":
        return
    base = REPO_ROOT / "data" / "processed" / "checkpoints"
    present = [n for n in ("final_checkpoint", "final_adapter") if (base / n).exists()]
    if len(present) == 1:
        overrides["--model_path"] = str(base / present[0])
        info("drift D8: " + t(f"checkpoint terminale = {present[0]}/", f"terminal checkpoint = {present[0]}/"))
    elif len(present) == 2:
        choice = ask(t("D8: trovati ENTRAMBI final_checkpoint e final_adapter. Quale?",
                       "D8: BOTH final_checkpoint and final_adapter exist. Which?"),
                     ["final_checkpoint", "final_adapter"], "final_checkpoint", ctx)
        overrides["--model_path"] = str(base / choice)
    else:
        warn("drift D8: " + t("nessuna directory di checkpoint terminale trovata.",
                              "no terminal checkpoint dir found."))
        if overrides.get("--model_path") == "AUTO_FINAL":
            overrides["--model_path"] = str(base / "final_checkpoint")

def drift_check_d11(entry: Entrypoint, overrides: dict, ctx: "Ctx") -> list[Step]:
    """D11 — gsm8k on a checkpoint needs rq3 rows for that step first."""
    if entry.key != "gsm8k":
        return []
    m = re.match(r"ckpt_(\d+)$", overrides.get("--tag", ""))
    if not m:
        return []
    step = int(m.group(1))
    traj = REPO_ROOT / "results/rq2_probing/dynamic/trajectories_probing.csv"
    if traj.exists() and any(str(r.get("step", "")).strip() == str(step) for r in _read_rows(traj)):
        return []
    warn("drift D11: " + t(f"trajectories_probing.csv non ha righe per step={step}; gsm8k fonderebbe nel vuoto.",
                           f"trajectories_probing.csv has no rows for step={step}; gsm8k would merge onto nothing."))
    if ask(t("Concatenare rq3 prima di gsm8k?", "Chain rq3 before gsm8k?"), ["sì", "no"], "sì", ctx) == "sì":
        return [("rq3", {"--checkpoint_dir": f"data/processed/checkpoints_extracted/checkpoint-{step}"})]
    return []

# ─── Pre-flight ─────────────────────────────────────────────────────────────

@dataclass
class Ctx:
    yes: bool = False
    dry_run: bool = False
    extra: list[str] = field(default_factory=list)

def ask(prompt: str, choices: list[str], default: str, ctx: Ctx) -> str:
    if ctx.yes:
        return default
    opts = "/".join(ch + ("*" if ch == default else "") for ch in choices)
    while True:
        try:
            raw = input(c(f"  {prompt} [{opts}] ", "cyan")).strip().lower()
        except EOFError:
            return default
        if not raw:
            return default
        for ch in choices:
            if ch.lower().startswith(raw) or raw == ch.lower():
                return ch
        print(c("  ?", "dim"))

def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

def _producer_of(missing: Path) -> list[Entrypoint]:
    out, mp = [], (REPO_ROOT / missing).resolve()
    for e in REGISTRY:
        for o in e.outputs:
            op = (REPO_ROOT / o).resolve()
            if op == mp or _within(mp, op) or _within(op, mp):
                out.append(e)
                break
    return out

def gpu_name() -> Optional[str]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=8)
        return (r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else "") or None
    except Exception:
        return None

def _preview(path: Path) -> str:
    p = REPO_ROOT / path
    if not p.exists():
        return ""
    try:
        st = p.stat()
        when = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        if p.is_dir():
            return f"dir · {sum(1 for _ in p.iterdir())} " + t("voci", "items") + f" · {when}"
        head = ""
        if p.suffix == ".csv":
            with p.open(encoding="utf-8") as fh:
                head = " | " + fh.readline().strip()[:70]
        elif p.suffix == ".json":
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                head = " | keys: " + ", ".join(list(d)[:6]) if isinstance(d, dict) else ""
            except Exception:
                head = ""
        return f"{st.st_size} B · {when}{head}"
    except Exception:
        return ""

def preflight(entry: Entrypoint, argv: list[str], ctx: Ctx) -> Optional[str]:
    """Returns 'run', 'skip', or None (abort)."""
    print(c(f"\n  ── preflight · {entry.key} ──", "bold"))
    for inp in entry.inputs:                                          # 1. inputs
        if not (REPO_ROOT / inp).exists():
            prods = [p for p in _producer_of(inp) if p.key != entry.key]
            warn(t(f"input mancante: {inp}", f"missing input: {inp}"))
            if prods:
                info(t("prodotto da: ", "produced by: ") + ", ".join(p.key for p in prods))
            choice = ask(t("[c]oncatena · [a]nnulla · [f]orza", "[c]hain · [a]bort · [f]orce"),
                         ["c", "a", "f"], "f" if ctx.yes else "a", ctx)
            if choice == "a":
                return None
            if choice == "c" and prods:
                producer = prods[0]
                if len(prods) > 1 and not ctx.yes:
                    producer = BY_KEY[ask(t("quale produttore?", "which producer?"),
                                          [p.key for p in prods], prods[0].key, ctx)]
                info(t(f"concateno {producer.key} prima…", f"chaining {producer.key} first…"))
                run_entry(producer, {}, Ctx(ctx.yes, ctx.dry_run, []))
    if entry.gpu_required:                                           # 2. GPU
        g = gpu_name()
        if g:
            info(f"GPU: {g}")
        else:
            warn(t("nessuna GPU rilevata ma l'entry la richiede.", "no GPU but entry requires one."))
            if ask(t("procedere comunque?", "proceed anyway?"), ["sì", "no"], "no", ctx) == "no":
                return None
    decision = "run"                                                # 3. output collision
    for o in entry.outputs:
        if (REPO_ROOT / o).exists():
            info(t(f"output esistente: {o}", f"existing output: {o}"))
            print(c("    " + _preview(o), "dim"))
            choice = ask(t("[r]igenera · [s]alta · [a]nnulla", "[r]erun · [s]kip · [a]bort"),
                         ["r", "s", "a"], "r", ctx)
            if choice == "a":
                return None
            if choice == "s":
                decision = "skip"
    info(t("comando: ", "command: ") + c(" ".join(shlex.quote(a) for a in argv), "cyan"))  # 5. echo
    return decision

# ─── Runner ─────────────────────────────────────────────────────────────────

def _parse_extra(extra: list[str]) -> tuple[dict, list[str]]:
    """Split passthrough into recognized --flag value pairs (so drift hooks see
    --tag/--model_path) and leftover tokens (store_true flags, unknown args)."""
    kv, leftover, i = {}, [], 0
    while i < len(extra):
        tok = extra[i]
        if tok.startswith("--") and i + 1 < len(extra) and not extra[i + 1].startswith("--"):
            kv[tok], i = extra[i + 1], i + 2
        else:
            leftover.append(tok)
            i += 1
    return kv, leftover

def build_argv(entry: Entrypoint, overrides: dict, leftover: list[str]) -> list[str]:
    argv = list(entry.command)
    for flag in entry.required_args:
        if flag in overrides and overrides[flag] != "AUTO_FINAL":
            argv += [flag, overrides[flag]]
        elif flag in RESOLVERS:
            argv += [flag, RESOLVERS[flag]]
        else:
            raise ValueError(t(f"argomento richiesto assente: {flag}", f"required argument missing: {flag}"))
    for flag, dflt in entry.optional_args.items():
        if flag in overrides and overrides[flag]:
            argv += [flag, overrides[flag]]
        elif flag not in overrides and dflt:
            argv += [flag, dflt]
    for f in overrides.get("__flags__", []):
        argv.append(f)
    return argv + leftover

def run_entry(entry: Entrypoint, overrides: dict, ctx: Ctx) -> int:
    kv, leftover = _parse_extra(ctx.extra)
    overrides = {**overrides, **kv}
    clean = Ctx(ctx.yes, ctx.dry_run, [])           # chained sub-runs don't inherit passthrough
    drift_resolve_d8(entry, overrides, ctx)         # D8 fills --model_path before the required check
    for flag in entry.required_args:                # prompt for required flags w/o resolver
        if flag not in overrides and flag not in RESOLVERS:
            if ctx.yes:
                err(t(f"--yes: {flag} richiesto ma non fornito.", f"--yes: {flag} required but absent."))
                return 2
            try:
                overrides[flag] = input(c(f"  {entry.key} richiede {flag}: ", "cyan")).strip()
            except EOFError:
                err(t(f"{flag} richiesto ma stdin chiuso.", f"{flag} required but stdin closed."))
                return 2
    for chained in drift_check_d11(entry, overrides, ctx):
        run_entry(BY_KEY[chained[0]], chained[1], clean)
    try:
        argv = build_argv(entry, overrides, leftover)
    except ValueError as e:
        err(str(e))
        return 2
    drift_note(entry)
    if drift_block_d6(entry, argv):
        return 3
    decision = preflight(entry, argv, ctx)
    if decision is None:
        info(t("annullato.", "aborted."))
        return 0
    if decision == "skip":
        info(t("saltato.", "skipped."))
        return 0
    if ctx.dry_run:
        info(t("dry-run: nessuna esecuzione.", "dry-run: no execution."))
        return 0
    return _execute(entry, argv)

def _execute(entry: Entrypoint, argv: list[str]) -> int:
    say(random.choice(DELPHIC["start"]))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"{ts}_{entry.key}.log"
    tail: list[str] = []
    t0, rc = time.monotonic(), 1
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(argv) + "\n")
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=str(REPO_ROOT))
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                print(c("· ", "dim") + line.rstrip())
                log.write(line)
                tail.append(line.rstrip())
                if len(tail) > 200:
                    tail.pop(0)
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            rc = 130
    dt = time.monotonic() - t0
    rss = ""
    if resource is not None:
        rss = f" · peak RSS {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.0f} MB"
    print(c(f"\n  ── {entry.key} · rc={rc} · {dt:.1f}s{rss} ──", "bold"))
    info(f"log: {log_path.relative_to(REPO_ROOT)}")
    if rc != 0:
        say(random.choice(DELPHIC["end_fail"]))
        err(t("ultime 30 righe:", "last 30 lines:"))
        for line in tail[-30:]:
            print(c("  " + line, "dim"))
    else:
        say(random.choice(DELPHIC["end_ok"]))
        report(entry)
    return rc

# ─── Report ─────────────────────────────────────────────────────────────────

def _read_rows(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []

def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def report(entry: Entrypoint) -> None:
    print(c("  ── " + t("responso", "report") + " ──", "bold"))
    fn = {"rq2": _report_rq2, "rq1": _report_rq1, "rq3": _report_rq3, "rq4": _report_rq4,
          "nf4": _report_nf4, "gsm8k": _report_gsm8k}.get(entry.key)
    try:
        fn() if fn else _report_generic(entry)
    except Exception as e:  # report must never crash the runner
        info(t(f"(report non disponibile: {e})", f"(report unavailable: {e})"))
    say(t("« La Pizia ha parlato. Vedi results/. »", "« The Pythia has spoken. See results/. »"))

def _report_generic(entry: Entrypoint) -> None:
    for o in entry.outputs:
        prev = _preview(o)
        if prev:
            info(f"{o} — {prev}")

def _report_rq2() -> None:
    rows = _read_rows(REPO_ROOT / "results/rq2_probing/accuracy_metrics_corrected.csv")
    for prop in ("sign", "parity"):
        sub = [r for r in rows if r.get("property") == prop and _f(r.get("accuracy")) is not None]
        if sub:
            best = max(sub, key=lambda r: _f(r.get("accuracy")) or 0.0)
            info(f"{prop}: peak acc {_f(best.get('accuracy')):.3f} @ L{best.get('layer')}")

def _report_rq1() -> None:
    iso = _read_rows(REPO_ROOT / "results/rq1_emergence/isotropy_aggregated_balanced.csv")
    deltas = [(_f(r.get("delta_iso")), r.get("layer")) for r in iso if _f(r.get("delta_iso")) is not None]
    if deltas:
        lo = min(deltas, key=lambda x: x[0])
        info(f"min ΔIso {lo[0]:.4f} @ L{lo[1]}")
    cka = _read_rows(REPO_ROOT / "results/rq1_emergence/cka_results_annotated.csv")
    if cka:
        info(t(f"CKA annotato: {len(cka)} righe", f"annotated CKA: {len(cka)} rows"))

def _report_rq3() -> None:
    rows = _read_rows(REPO_ROOT / "results/rq2_probing/dynamic/trajectories_probing.csv")
    rels = [(_f(r.get("geom_delta_math_rel")), r.get("step"), r.get("layer")) for r in rows]
    rels = [x for x in rels if x[0] is not None]
    if rels:
        hi = max(rels, key=lambda x: x[0])
        info(t("deriva relativa max (math) ", "max relative drift (math) ") + f"{hi[0]:.3f} @ step {hi[1]} L{hi[2]}")
    accs = [_f(r.get("probing_acc")) for r in rows if _f(r.get("probing_acc")) is not None]
    if accs:
        info(t("acc sonda: ", "probe acc: ") + f"{min(accs):.3f} → {max(accs):.3f}")

def _report_rq4() -> None:
    rows = _read_rows(REPO_ROOT / "results/rq4_determinization/determinization.csv")
    steps = sorted({int(r["step"]) for r in rows if r.get("step", "").strip().lstrip("-").isdigit()})
    if not steps:
        return
    s0, sf = steps[0], steps[-1]
    for cat in sorted({r.get("category", "") for r in rows}):
        a = next((r for r in rows if r.get("category") == cat and int(r["step"]) == s0), None)
        b = next((r for r in rows if r.get("category") == cat and int(r["step"]) == sf), None)
        if a and b:
            info(f"{cat}: entropy {_f(a.get('entropy_mean')):.3f}→{_f(b.get('entropy_mean')):.3f} · "
                 f"margin {_f(a.get('margin_mean')):.3f}→{_f(b.get('margin_mean')):.3f}")

def _report_nf4() -> None:
    p = REPO_ROOT / "results/nf4_degradation/summary.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    info(f"mean_frobenius_relative = {d.get('mean_frobenius_relative')}")
    info(f"rq3_max_relative_drift = {d.get('rq3_max_relative_drift')}")
    snr = d.get("signal_to_noise_ratio")
    if snr is None:
        info(t("signal_to_noise_ratio = null (caso floor-zero, SNR non calcolabile)",
               "signal_to_noise_ratio = null (zero-floor case, SNR not computable)"))
    else:
        info(f"signal_to_noise_ratio = {snr}×")

def _report_gsm8k() -> None:
    d = REPO_ROOT / "results/gsm8k"
    files = sorted(d.glob("gsm8k_*.json"), key=lambda x: x.stat().st_mtime) if d.exists() else []
    if not files:
        return
    j = json.loads(files[-1].read_text(encoding="utf-8"))
    lo, hi = j.get("ci_lower"), j.get("ci_upper")
    info(f"{files[-1].name}: acc={j.get('accuracy')}" + (f" · CI [{lo}, {hi}]" if lo is not None else ""))

# ─── CLI ────────────────────────────────────────────────────────────────────

def render_table(headers: list[str], rows: list[list[str]]) -> None:
    if _RICH is not None and USE_COLOR:
        tbl = Table(show_header=True, header_style="bold")
        for h in headers:
            tbl.add_column(h)
        for r in rows:
            tbl.add_row(*r)
        _RICH.print(tbl)
        return
    widths = [len(h) for h in headers]
    for r in rows:
        widths = [max(w, len(str(cell))) for w, cell in zip(widths, r)]
    print(c("  ".join(h.ljust(w) for h, w in zip(headers, widths)), "bold"))
    print(c("  ".join("─" * w for w in widths), "dim"))
    for r in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(r, widths)))

def cmd_list(category: Optional[str]) -> None:
    rows = []
    for e in REGISTRY:
        if category and e.category.name.lower() != category.lower():
            continue
        rows.append([e.key, f"{e.category.glyph} {e.category.name}", e.cost,
                     "GPU" if e.gpu_required else "—", e.description])
    if not rows:
        warn(t(f"nessuna entry per categoria '{category}'", f"no entry for category '{category}'"))
        return
    render_table([t("chiave", "key"), t("categoria", "category"), t("costo", "cost"),
                  "gpu", t("descrizione", "description")], rows)
    print(c(f"\n  {len(rows)} " + t("entrypoint · riti: ", "entrypoints · rites: ")
            + ", ".join(RITES), "dim"))

def banner() -> None:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, cwd=str(REPO_ROOT), timeout=5).stdout.strip() or "—"
    except Exception:
        h = "—"
    print(c(TRIPOD, "cyan"))
    print(c(f"   Banco della Pizia · dev @ {h} · {os.environ.get('CONDA_DEFAULT_ENV', '—')} "
            f"· GPU: {gpu_name() or '—'}", "bold"))
    say(DELPHIC["banner"])

def run_sequence(name: str, ctx: Ctx) -> int:
    if name not in RITES:
        err(t(f"rito sconosciuto: {name}", f"unknown rite: {name}"))
        return 2
    say(t(f"« Rito: {name}. »", f"« Rite: {name}. »"))
    rc = 0
    for key, overrides in RITES[name]:
        if key == "__pytest__":
            rc = subprocess.call([PY, "-m", "pytest", "tests/", "-q"], cwd=str(REPO_ROOT))
        else:
            rc = run_entry(BY_KEY[key], overrides, ctx)
        if rc != 0:
            err(t(f"rito interrotto a {key} (rc={rc}).", f"rite halted at {key} (rc={rc})."))
            return rc
    return rc

# ─── Interactive menu ───────────────────────────────────────────────────────

def menu(ctx: Ctx) -> None:
    cats = list(Category)
    while True:
        print(c("\n  " + t("Categorie:", "Categories:"), "bold"))
        for i, cat in enumerate(cats, 1):
            n = sum(1 for e in REGISTRY if e.category == cat)
            print(f"   {i:2} {cat.glyph}  {cat.name:<11} {c(f'({n})', 'dim')}")
        print(f"   {c('R', 'magenta')}  " + t("riti compositi", "composite rites"))
        print(f"   {c('L', 'magenta')}  " + t("lista completa", "full list"))
        print(f"   {c('Q', 'magenta')}  " + t("esci", "quit"))
        sel = input(c("  > ", "cyan")).strip()
        if not sel:
            continue
        if sel.lower() == "q":
            return
        if sel.lower() == "l":
            cmd_list(None)
        elif sel.lower() == "r":
            _rite_menu(ctx)
        elif sel in BY_KEY:
            run_entry(BY_KEY[sel], {}, ctx)
        elif sel.isdigit() and 1 <= int(sel) <= len(cats):
            _category_menu(cats[int(sel) - 1], ctx)

def _category_menu(cat: Category, ctx: Ctx) -> None:
    entries = [e for e in REGISTRY if e.category == cat]
    while True:
        print(c(f"\n  {cat.glyph} {cat.label}", "bold"))
        for i, e in enumerate(entries, 1):
            gpu = c(" GPU", "yellow") if e.gpu_required else ""
            print(f"   {i:2} {c(e.key, 'cyan'):<22} {e.name}  {c(e.cost, 'dim')}{gpu}")
        print(f"   {c('B', 'magenta')}  " + t("indietro", "back"))
        sel = input(c("  > ", "cyan")).strip()
        if sel.lower() == "b" or not sel:
            return
        if sel.isdigit() and 1 <= int(sel) <= len(entries):
            run_entry(entries[int(sel) - 1], {}, ctx)

def _rite_menu(ctx: Ctx) -> None:
    names = list(RITES)
    print(c("\n  " + t("Riti compositi:", "Composite rites:"), "bold"))
    for i, n in enumerate(names, 1):
        print(f"   {i:2} {c(n, 'cyan'):<28} {c(RITE_DESC.get(n, ''), 'dim')}")
    print(f"   {c('B', 'magenta')}  " + t("indietro", "back"))
    sel = input(c("  > ", "cyan")).strip()
    if sel.isdigit() and 1 <= int(sel) <= len(names):
        run_sequence(names[int(sel) - 1], ctx)

# ─── main ───────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    global LANG, USE_COLOR
    parser = argparse.ArgumentParser(description="Il Banco della Pizia — thesis orchestrator")
    parser.add_argument("--list", action="store_true", help="dump REGISTRY as a table")
    parser.add_argument("--category", help="filter --list by category name")
    parser.add_argument("--run", metavar="KEY", help="run one entrypoint")
    parser.add_argument("--sequence", metavar="NAME", help="run a composite rite")
    parser.add_argument("--lang", choices=["it", "en"], default="it")
    parser.add_argument("--dry-run", action="store_true", help="pre-flight only")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--yes", action="store_true", help="accept confirmations (not D6)")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="args after -- forwarded")
    args = parser.parse_args(argv)

    LANG = args.lang
    USE_COLOR = not args.no_color and (args.list or sys.stdout.isatty())
    extra = list(args.passthrough)
    if extra and extra[0] == "--":
        extra = extra[1:]
    ctx = Ctx(yes=args.yes, dry_run=args.dry_run, extra=extra)

    if args.list:
        cmd_list(args.category)
        return 0
    if args.run:
        if args.run not in BY_KEY:
            err(t(f"chiave sconosciuta: {args.run}", f"unknown key: {args.run}"))
            return 2
        return run_entry(BY_KEY[args.run], {}, ctx)
    if args.sequence:
        return run_sequence(args.sequence, ctx)
    banner()
    try:
        menu(ctx)
    except (KeyboardInterrupt, EOFError):
        print()
        say(t("« Il pellegrino si congeda. »", "« The pilgrim departs. »"))
    return 0

if __name__ == "__main__":
    sys.exit(main())
