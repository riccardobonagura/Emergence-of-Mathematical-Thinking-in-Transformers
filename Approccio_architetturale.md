L'obiettivo di questa sessione è **progettare la v2 della codebase** — non implementarla, solo progettarla — con focus su tre dimensioni:

1. **Leggibilità umana e AI** — un agente che legge un singolo file deve capire l'intero dominio verticale senza navigare tra 7 moduli separati
2. **Rigore dei contratti** — TypedDict o dataclass espliciti tra orchestratori e worker, nessun dict implicito passato tra moduli
3. **Navigabilità** — riduzione da 30+ file a ~9 file monolitici verticali, con sezioni interne che sostituiscono i file separati. (NON ADESSO)

---

### STATO ATTUALE — STRUTTURA CODEBASE

```
src/
  config/          categories.py, models.py
  dataset/         build_stimuli.py, build_control.py, merge_stimuli.py, test_dataset.py
  extraction/      extract_states.py, checkpoint_loop.py
  metrics/         cka.py, isotropy.py
  probing/         seeds.py, directions.py, stats.py, pipeline.py,
                   probing_dataset.py, engine.py, io_utils.py
  finetuning/      train_qlora.py
  eval/            eval_gsm8k.py, nf4_degradation.py
  utils/           validate_configs.py
  viz/             plot_rq1_emergence.py, probing_viz.py, pca_umap_viz.py
run_rq1.py, run_rq2.py, run_rq3.py
configs/, tests/, data/, results/
```

---

### OBIETTIVO v2 — STRUTTURA TARGET

```
src/
  config.py        # categories + model registry (invariato, già compatto)
  data.py          # build_stimuli + build_control + merge + validation
  extraction.py    # extract_states + checkpoint_loop
  geometry.py      # isotropy + CKA + PCA/UMAP + viz RQ1
  probing.py       # seeds + directions + stats + pipeline + engine +
                   # probing_dataset + io_utils + viz RQ2
  training.py      # train_qlora + eval_gsm8k + nf4_degradation + viz RQ3

run_rq1.py         # orchestratore RQ1 — thin wrapper su geometry.py
run_rq2.py         # orchestratore RQ2 — thin wrapper su probing.py
run_rq3.py         # orchestratore RQ3 — thin wrapper su training.py + probing.py
run.py             # orchestratore globale interattivo con stato persistito
```

---

### I TRE TASK ARCH

**ARCH-01 — Refactor verticale monolitico** Collassare i 30+ file in 6 file src/ monolitici. Ogni file è autocontenuto verticalmente — sezioni interne (`# ══ SECTION 1 — SEEDS ══`) sostituiscono i moduli separati. L'obiettivo è che leggere `probing.py` dall'inizio alla fine dia la comprensione completa del probing senza saltare tra file.

**ARCH-02 — Orchestratore globale `run.py`** CLI interattiva con stato persistito in `run_state.json`. Conosce i prerequisiti di ogni fase, blocca esecuzione fuori ordine, mostra lo stato corrente:

```
[1] Generate dataset     ✓ DONE
[2] Extract states       ✓ DONE
[3] RQ1 Geometry         ✓ DONE
[4] RQ2 Probing          ✓ DONE
[5] Fine-tuning          ✓ DONE
[6] GSM8K eval           🔄 IN PROGRESS
[7] Checkpoint loop      ⏳ WAITING
[8] Visualize all        ⏳ WAITING
```

**ARCH-03 — Contratti espliciti** Definire TypedDict o dataclass per ogni handoff tra orchestratori e worker:

```python
@dataclass
class ExtractionResult:
    tensor_dir: Path
    n_layers: int
    d_model: int
    n_stimuli: int

@dataclass  
class ProbingResult:
    layer: int
    property: str
    accuracy: float
    ci_lower: float
    ci_upper: float
    weights: np.ndarray
    bias: float
```

---

### DECISIONI ARCHITETTURALI GIÀ PRESE

- **Monolitico verticale** è la direzione scelta — ispirato alla tesi di Matt Pocock che gli agenti AI tendono a micro-modularizzare eccessivamente. L'utente vuole testare il modello opposto.
- **Nessun framework** (Lightning, Hydra, Kedro) — overhead superiore al vantaggio per un progetto di ricerca single-GPU
- **Sezioni interne** invece di file separati: `# ══════ SECTION N — NOME ══════` come divisori visivi
- **`run.py` con stato persistito** — non un Makefile, non uno script bash, ma una CLI Python con memoria