"""Smoke tests for scripts/oracolo.py — Il Banco della Pizia orchestrator.

Pins the orchestrator's registry (28 entrypoints) and its drift hooks (D3, D6)
against silent breakage. When someone adds, renames, deletes, or breaks an
entrypoint or a drift hook, these tests fail loudly.

Design:
  - The orchestrator is a CLI; we drive it via subprocess.run, never import it.
  - Everything runs through --dry-run: no underlying pipeline script executes,
    no GPU, no network.
  - --config configs/config_rq2.yaml is passed so the D6 guard (missing
    total_training_steps) does not fire on tests that should pass.

Hardcoded KNOWN_KEYS / COMPOSITE_RITES / ALLOWED_RC are the intentional
tripwire: when the registry legitimately changes, this file must be edited in
the same commit. That is the safeguard, not a nuisance.

Two rites chain fewer than 2 registry keys, so the "≥2 chained keys" sanity is
relaxed for them (see _MIN_CHAINED_KEYS):
  - smoke_test  delegates to `pytest tests/` (a non-registry pseudo-step); under
    --dry-run it echoes the argv without executing pytest, so it is safe to run
    here and enumerates 0 registry keys.
  - dataset_regen is a single-step orchestrator rite (one registry key: regen).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# ─── Module-level: skip if oracolo absent ───────────────────────────────────
ORACOLO = Path(__file__).resolve().parents[1] / "scripts" / "oracolo.py"
if not ORACOLO.exists():
    pytest.skip("scripts/oracolo.py not built", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "config_rq2.yaml"

KNOWN_KEYS = (
    "extract", "rq1", "rq2", "confound-sign", "confound-par", "train",
    "loop", "rq3", "nf4", "gsm8k", "rq4", "rq1-dyn", "validate", "io-smoke",
    "build-stim", "merge-stim", "regen", "ds-test", "cka-main", "iso-main",
    "viz-rq1", "viz-rq2", "viz-rq3", "viz-rq4", "viz-supp", "viz-pca",
    "gen-fix", "chk-iface",
)
assert len(KNOWN_KEYS) == 28

COMPOSITE_RITES = (
    "cammino_completo", "solo_probing", "solo_geometria", "solo_rq4",
    "solo_viz", "smoke_test", "dataset_regen",
)

ALLOWED_RC = {0, 2, 3}  # 0=ok, 2=clean missing-arg refusal, 3=D6

# Rites that legitimately chain fewer than 2 registry keys (relaxes the sanity):
# smoke_test runs a non-registry pytest step (0 keys); dataset_regen is single-step.
_MIN_CHAINED_KEYS = {"dataset_regen": 1, "smoke_test": 0}

_PREFLIGHT_RE = re.compile(r"preflight · (\S+)")
_TRACEBACK = "Traceback (most recent call last)"


def run_oracolo(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ORACOLO), *args],
                          capture_output=True, text=True, timeout=20, cwd=str(REPO))


def _out(r: subprocess.CompletedProcess) -> str:
    return (r.stdout or "") + "\n" + (r.stderr or "")


def _preflight_keys(out: str) -> set[str]:
    """Distinct registry keys that actually reached a pre-flight in a run."""
    return {m for m in _PREFLIGHT_RE.findall(out) if m in KNOWN_KEYS}


def _lora_max_seq_length() -> str:
    """Read the live max_seq_length from lora_config.yaml (column-0 key, no yaml lib)."""
    for line in (REPO / "configs" / "lora_config.yaml").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("max_seq_length"):
            return line.split(":", 1)[1].split("#", 1)[0].strip()
    raise AssertionError("max_seq_length not found in configs/lora_config.yaml")


# ─── Registry presence ──────────────────────────────────────────────────────

def test_list_contains_all_28_keys():
    r = run_oracolo("--list", "--no-color", "--lang", "en")
    assert r.returncode == 0, _out(r)
    out = _out(r)
    for k in KNOWN_KEYS:
        assert k in out, f"registry key missing from --list: {k}"
    low = out.lower()
    assert "rites" in low or "riti" in low, "no composite-rites section in --list"
    assert "cammino_completo" in out, "expected rite cammino_completo not listed"


@pytest.mark.parametrize("key", KNOWN_KEYS)
def test_dry_run_every_entrypoint(key):
    r = run_oracolo("--dry-run", "--run", key, "--yes", "--no-color", "--lang", "en",
                    "--", "--config", str(CFG))
    out = _out(r)
    assert r.returncode in ALLOWED_RC, f"{key}: rc={r.returncode}\n{out}"
    assert _TRACEBACK not in out, f"{key}: orchestrator raised a traceback\n{out}"


# ─── Composite rites ────────────────────────────────────────────────────────

@pytest.mark.parametrize("rite", COMPOSITE_RITES)
def test_composite_dry_run(rite):
    r = run_oracolo("--dry-run", "--sequence", rite, "--yes", "--no-color", "--lang", "en")
    out = _out(r)
    assert r.returncode in ALLOWED_RC, f"{rite}: rc={r.returncode}\n{out}"
    assert _TRACEBACK not in out, f"{rite}: orchestrator raised a traceback\n{out}"
    keys = _preflight_keys(out)
    minimum = _MIN_CHAINED_KEYS.get(rite, 2)
    assert len(keys) >= minimum, f"{rite}: chained keys {sorted(keys)} < {minimum}"


# ─── Error handling ─────────────────────────────────────────────────────────

def test_unknown_key_fails_cleanly():
    r = run_oracolo("--run", "definitely_not_a_key", "--yes")
    out = _out(r)
    assert r.returncode != 0, "unknown key should not exit 0"
    assert _TRACEBACK not in out, f"unknown key raised a traceback\n{out}"
    low = out.lower()
    assert "definitely_not_a_key" in out or "unknown" in low or "sconosciut" in low


# ─── Drift hooks ────────────────────────────────────────────────────────────

def test_drift_D6_fires(tmp_path):
    # Copy config_rq2.yaml, strip the total_training_steps line (plain text, no yaml lib).
    src = (REPO / "configs" / "config_rq2.yaml").read_text(encoding="utf-8")
    stripped = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("total_training_steps"))
    cfg = tmp_path / "no_total_steps.yaml"
    cfg.write_text(stripped, encoding="utf-8")

    r = run_oracolo("--dry-run", "--run", "rq4", "--yes", "--", "--config", str(cfg))
    out = _out(r)
    assert r.returncode == 3, f"D6 must refuse with rc=3, got {r.returncode}\n{out}"
    assert "d6" in out.lower(), f"D6 message absent\n{out}"


def test_drift_D3_surfaces_max_seq_length():
    expected = _lora_max_seq_length()  # read live, do not hardcode 512
    r = run_oracolo("--dry-run", "--run", "train", "--yes", "--no-color", "--lang", "en",
                    "--", "--config", str(CFG), "--lora_config", "configs/lora_config.yaml")
    out = _out(r)
    assert r.returncode in ALLOWED_RC, f"train: rc={r.returncode}\n{out}"
    assert _TRACEBACK not in out, out
    assert "max_seq_length" in out, f"D3 did not surface max_seq_length\n{out}"
    assert expected in out, f"D3 did not surface the live value {expected!r}\n{out}"


def test_dry_run_smoke_test_does_not_execute_pytest():
    # python scripts/oracolo.py --dry-run --sequence smoke_test --yes --no-color --lang en
    result = subprocess.run(
        [sys.executable, str(ORACOLO), "--dry-run",
         "--sequence", "smoke_test", "--yes", "--no-color",
         "--lang", "en"],
        capture_output=True, text=True, timeout=10, cwd=str(REPO)
    )
    out = (result.stdout or "") + "\n" + (result.stderr or "")
    # Contract: dry-run echoes the argv but does NOT execute pytest. Pytest, when
    # actually run, prints "collected N items" (or "collecting ..."). Its absence
    # proves no execution.
    assert "collected" not in out.lower(), \
        "pytest appears to have actually run under --dry-run"
    assert "collecting" not in out.lower()
    # And the argv echo must be present (proves preflight reached the would-run point).
    assert "pytest" in out  # the argv mentions pytest
    assert result.returncode == 0
