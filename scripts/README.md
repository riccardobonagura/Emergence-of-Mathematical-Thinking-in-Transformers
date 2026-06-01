# Il Banco della Pizia — `scripts/oracolo.py`

Single-file interactive orchestrator for the thesis pipeline. The pilgrim
consults the Oracle of Delphi about Pythia's inner geometry. Every entrypoint
catalogued in `docs/RECON.md` §2 (28 of them) is exposed through a menu, run as
a subprocess, parsed, and reported. The hardcoded `REGISTRY` is the contract —
the script does **not** re-discover entrypoints at runtime.

Stdlib only; `rich` and `pandas` are used if importable, never required.

## Invocation

```bash
python scripts/oracolo.py                       # interactive menu (Italian)
python scripts/oracolo.py --list                # dump the registry as a table
python scripts/oracolo.py --list --category rq2 # filter the table
python scripts/oracolo.py --run KEY [-- ...]    # run one entry; args after -- are forwarded
python scripts/oracolo.py --sequence NAME       # run a composite rite
python scripts/oracolo.py --dry-run --run KEY   # pre-flight only, no execution
python scripts/oracolo.py --lang en             # English UI
python scripts/oracolo.py --no-color            # disable styling
python scripts/oracolo.py --yes                 # accept confirmations (CI); D6 stays loud
```

## Categories (menu order)

`SETUP 🛠 · DATASET 𝝳 · EXTRACTION 🜍 · RQ1 △ · RQ2 ⊕ · FINETUNING 🜚 ·
RQ3 ⇌ · RQ4 ⚖ · EVAL 𝛴 · VIZ ◐ · TESTS ✓ · UTILS ·`

## Composite rites (`--sequence` or menu `R`)

`cammino_completo` (full cold-start, GPU) · `solo_probing` · `solo_geometria` ·
`solo_rq4` · `solo_viz` · `smoke_test` (pytest) · `dataset_regen`.

## Encoded drifts (RECON §5, handled in code)

- **D2** — RQ2 prints "config ships C=1.0; spec asserts 10.0".
- **D3** — `train` reads `lora_config.yaml` and flags `max_seq_length` ≠ 1024.
- **D6** — `rq3/rq4/rq1-dyn/gsm8k` refuse if `total_training_steps` is absent
  (silent default 2000 ≠ canonical 12343). **Not overridable by `--yes`.**
- **D8** — `gsm8k --tag final` resolves `final_checkpoint/` vs `final_adapter/`.
- **D11** — `gsm8k ckpt_N` chains `rq3` first if the trajectory CSV lacks that step.

Note: `build_control` is a library, not an entry (D1) — it is not registered.

## Pre-flight (every run)

Missing inputs → offer to chain the producer / abort / force · GPU probe ·
output-collision preview (mtime/size/head) · drift hooks · exact `argv` echo.
Live output is teed to `logs/oracolo/{timestamp}_{key}.log` (never overwritten);
non-zero exit prints the last 30 lines.

## Adding an entry to the registry

Append an `Entrypoint(...)` to `REGISTRY`. Required fields:

```python
Entrypoint(
    "my-key", "Titolo IT", "Title EN", Category.RQ2,
    [PY, "-m", "src.module"],          # command (always start from sys.executable = PY)
    required_args=["--config"],         # flags that MUST be present
    optional_args={"--flag": "default"},# name -> default (appended if non-empty)
    inputs=[P("data/...")],             # checked in pre-flight; reverse-indexed to producers
    outputs=[P("results/...")],         # collision preview + report source
    gpu_required=True, cost="medium",   # cost ∈ {fast, medium, long, very_long}
    description="one line: what, not how",
)
```

If the flag needs a repo-wide default value, add it to `RESOLVERS`. To add a
bespoke post-run summary, extend `report()` with a `_report_<key>()` branch.

> Caveat: `chk-iface` (`tests/check_interface.py`) actually loads Pythia-1.4B
> via TransformerLens — it is **not** CPU-only/fast, so it is registered with
> `gpu_required=True, cost="medium"`. Validate it with `--dry-run` unless you
> intend to wake the model.
