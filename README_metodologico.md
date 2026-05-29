# Geometric Dynamics in Transformer Internal Representations
**Tesi di Laurea Triennale · Bonagura N46007216 · Università degli Studi di Napoli Federico II**
**Corso di Laurea in Informatica · A.A. 2024/2025**

---

## Posizionamento epistemologico

Questa tesi si colloca nel campo della **Mechanistic Interpretability**, il cui obiettivo è comprendere *come* un modello linguistico produce un output, non solo *quale* output produce. L'approccio è **correlativo**, non meccanicistico: i risultati identificano *dove* determinate informazioni sono accessibili nelle rappresentazioni interne, senza tracciare i circuiti computazionali causali che le producono. Questo è il passo che precede il reverse engineering — localizzare prima di spiegare.

Il modello oggetto di studio è **Pythia-1.4B** (EleutherAI), scelto per la sua trasparenza architetturale e per essere progettato specificamente per la ricerca in interpretabilità. I risultati sono validi per questo modello, questo tokenizer, questo dominio aritmetico. Generalizzazioni a modelli diversi richiedono replica esplicita.

---

## Research Questions

### RQ1 — Differenziazione Geometrica per Layer

> *In Pythia-1.4B, in quali layer le rappresentazioni interne di stimoli aritmetici mostrano la massima divergenza geometrica rispetto a stimoli linguistici di controllo, misurata tramite isotropy differenziale e CKA inter-categoria?*

**Razionale.** Se il modello ha sviluppato rappresentazioni specializzate per il ragionamento aritmetico, ci aspettiamo che lo spazio delle rappresentazioni matematiche diverga geometricamente da quello del linguaggio generico in modo misurabile e localizzabile per layer. L'isotropy differenziale (ΔIso) misura la dispersione relativa delle rappresentazioni; la CKA inter-categoria misura la similarità strutturale tra i due sottospazi. I due strumenti sono complementari: il primo cattura proprietà globali della distribuzione, il secondo la struttura relazionale.

**Scope dichiarato.** I risultati descrivono la divergenza geometrica tra stimoli aritmetici nel formato `{a} OP {b} =` e stimoli di controllo linguistici. Non è possibile concludere che la struttura osservata sia causalmente responsabile delle capacità aritmetiche del modello.

**Strumento primario per RQ1:** CKA inter-categoria `CKA(H_math^l, H_ctrl^l)` per layer l.
**Strumento secondario:** ΔIso(l) = ISO(math, l) − ISO(ctrl, l), dove ISO = media delle similarità coseno tra coppie di rappresentazioni. Nota: ISO alto → anisotropia (vettori collineari); ISO basso → isotropia (vettori distribuiti). ΔIso < 0 indica che le rappresentazioni matematiche sono *più isotrope* di quelle di controllo.

---

### RQ2 — Decodificabilità Lineare di Proprietà Aritmetiche

> *Quali proprietà del risultato aritmetico — nello specifico segno e parità — sono linearmente decodificabili dagli hidden state di Pythia-1.4B, e in quali layer questa decodificabilità emerge e raggiunge il picco?*

**Razionale.** La decodificabilità lineare è una condizione necessaria (non sufficiente) per affermare che un'informazione è rappresentata in modo accessibile nel sottospazio lineare delle attivazioni. Una sonda lineare (regressione logistica) che raggiunge alta accuracy indica che esiste una direzione nel hidden space lungo cui l'informazione è organizzata. Segno e parità sono proprietà cognitivamente distinte: il segno dipende dalla struttura degli operandi di input, la parità richiede computazione aritmetica. Questa differenza è metodologicamente rilevante.

**Scope dichiarato.** La decodificabilità lineare non implica che il modello *usi* questa informazione per generare il suo output (Hewitt & Liang, 2019). I risultati sono validi per stimoli nel dominio [10,50], operatori sottrazione (CAT-SIGN) e addizione/sottrazione (CAT-PARITY), 3 template sintatticamente simili. La generalizzazione a operandi fuori da questo range, a template diversi o a operatori diversi richiede verifica esplicita.

**Validazione statistica.** Ogni layer: bootstrap CI 95% (N=2000 campionamenti), permutation test con 1000 shuffle delle label di training su split fisso (p-value riportato). Selectivity = accuracy(task) − accuracy(control task) riportata per escludere memorizzazione di artefatti superficiali. Correzione Benjamini-Hochberg applicata per 48 test simultanei (24 layer × 2 proprietà).

**Extraction point.** Le rappresentazioni sono estratte al token "=" (ultimo token della sequenza), *prima* che il modello generi il risultato. Questo significa che la sonda misura la rappresentazione dell'*aspettativa* del risultato, non del risultato computato. Questa distinzione è ontologicamente rilevante e dichiarata come limite interpretativo.

---

### RQ3 — Dinamica Geometrica del Fine-Tuning

> *Il fine-tuning QLoRA su MetaMath produce nelle rappresentazioni residue di Pythia-1.4B un drift geometrico superiore alla degradazione baseline NF4? In quali layer le proiezioni di attenzione mostrano la maggiore riorganizzazione, e questo drift correla con il miglioramento su GSM8K?*

**Razionale.** Se l'apprendimento matematico produce riorganizzazione geometrica sistematica, il drift delle rappresentazioni dovrebbe essere non uniforme tra i layer e correlato con il miglioramento funzionale misurato su GSM8K. La baseline NF4 controlla per la degradazione introdotta dalla quantizzazione a 4 bit, separando il segnale di apprendimento dal rumore di quantizzazione.

**Scope dichiarato e confound strutturale.** Il fine-tuning QLoRA è applicato esclusivamente alle proiezioni di attenzione (`query_key_value`). Le MLP sono congelate. La letteratura (ROME/MEMIT) localizza lo storage di conoscenza fattuale prevalentemente nelle MLP intermedie. Di conseguenza, le inferenze di RQ3 sono valide per la riorganizzazione delle proiezioni di attenzione, non per l'architettura completa del modello. Questo è un limite strutturale dichiarato, non un errore metodologico.

**Metrica drift.** Frobenius distance relativa: `‖H_ckpt − H_base‖_F / ‖H_base‖_F`, calcolata separatamente su stimoli matematici e di controllo. La metrica non decompone la trasformazione in componenti rotazionali, scalari e traslazionali — solo la distanza totale è misurata.

**Valutazione funzionale.** GSM8K benchmark (Cobbe et al., 2021), 0-shot, flexible extract. La letteratura riporta risultati standard a 5-shot: la differenza sistematica stimata è 5–15% inferiore per la condizione 0-shot. Tutti i confronti sono interni alla stessa configurazione di valutazione.

---

## Dataset

Il dataset è costruito secondo il paradigma delle **coppie minime contrastive**, ispirato a BLiMP (Warstadt et al., 2019): ogni coppia differisce esattamente per la proprietà target, mantenendo invariate tutte le altre caratteristiche superficiali.

| Categoria | N stimoli | Proprietà | Operatori |
|---|---|---|---|
| CAT-SIGN | 1000 (500 coppie) | Segno del risultato | Sottrazione, a,b ∈ [10,50] |
| CAT-PARITY | 1000 (500 coppie) | Parità del risultato | Addizione + sottrazione |
| CTRL-NEU | 500 | — | Prosa inglese, no numeri |
| CTRL-NUM | 500 | — | Contesto numerico non aritmetico |

**Vincoli tecnici dichiarati.**
- Operandi in [10,50]: ogni operando è rappresentato da un singolo token nel tokenizer GPT-NeoX. Operandi negativi o con più di 2 cifre richiedono token multipli e rompono l'invariante di estrazione.
- Template: 3 varianti sintatticamente simili per categoria. I risultati sono template-specific per costruzione.
- Pool coverage: ~60% dello spazio delle coppie possibili. Il test set non è out-of-distribution rispetto al training set della sonda.
- Bilanciamento 50/50: artificiale per design, standard nel probing (BLiMP). L'accuracy è misurata in condizioni ottimali; il prior reale delle classi non è rappresentato.

---

## Architettura sperimentale

```
Dataset (3000 stimoli)
    │
    ▼
Estrazione hidden states — Pythia-1.4B
    │  24 layer × 3000 stimoli × 2048 dim → 24 tensori [3000, 2048]
    │  Strategia: last_token ("="), left-padding, FP16
    │
    ├──▶ RQ1: Isotropy differenziale + CKA inter/evolutionary
    │
    ├──▶ RQ2: Linear Probing (sign, parity)
    │         LogisticRegression, lbfgs, C=1.0
    │         Split 80/20 stratificato, seed deterministico
    │         Validazione: bootstrap CI, permutation test, selectivity, BH correction
    │         Confound T02: cosine(w_sign, w_magnitude) + Pearson(operand_a, sign_label)
    │
    └──▶ RQ3: QLoRA fine-tuning su MetaMathQA
              25 checkpoint → merged weights → estrazione → frozen probe accuracy
              Frobenius drift per layer (math vs ctrl separati)
              GSM8K 0-shot a ogni checkpoint
```

---

## Riferimenti metodologici primari

- Belinkov (2022) — *Probing Classifiers: Promises, Shortcomings, and Advances*
- Hewitt & Liang (2019) — *Designing and Interpreting Probes with Control Tasks*
- Ethayarajh (2019) — *How Contextual are Contextualized Word Representations?*
- Kornblith et al. (2019) — *Similarity of Neural Network Representations Revisited* (CKA)
- Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs*
- Meng et al. (2022) — *Locating and Editing Factual Associations in GPT* (ROME)
- Quirke & Barez (2024) — *Understanding Addition in Transformers*
- Nanda et al. (2023) — *Progress measures for grokking via mechanistic interpretability*
- Cobbe et al. (2021) — *Training Verifiers to Solve Math Word Problems* (GSM8K)
- Benjamini & Hochberg (1995) — *Controlling the False Discovery Rate*
- Warstadt et al. (2019) — *BLiMP: The Benchmark of Linguistic Minimal Pairs*
- Yu et al. (2023) — *MetaMath: Bootstrap Your Own Mathematical Questions*

---

## Limiti dichiarati

| Limite | Tipo | Riferimento |
|---|---|---|
| Singolo modello (Pythia-1.4B) | Scope | R-I-02 |
| Singolo template aritmetico | Scope | E-P-04, Zhang et al. 2025 |
| Approccio correlativo, non causale | Epistemologico | E-G-04 |
| Estrazione all'aspettativa, non al risultato | Metodologico | E-P-02 |
| QLoRA su QKV only — MLP congelate | Strutturale | RQ3 scope |
| Dominio [10,50] — no OOD | Generalizzabilità | E-P-04 |
| 0-shot GSM8K vs 5-shot standard | Comparabilità | E-F-01 |
| MetaMath/GSM8K distributional overlap | Contaminazione | Yu et al. 2023 |
| Pool coverage ~60% — test set non OOD | Generalizzabilità | — |
