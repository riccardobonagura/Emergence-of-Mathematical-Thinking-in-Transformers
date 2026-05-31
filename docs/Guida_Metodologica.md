# Geometric Dynamics in Transformer Internal Representations — Metodologia unificata

**Tesi di Laurea Triennale · Bonagura N46007216 · Università degli Studi di Napoli Federico II**
**Corso di Laurea in Informatica · A.A. 2025/2026 · Modello: Pythia-1.4B (EleutherAI)**

> Documento unico che assorbe `README_metodologico.md` (front matter della tesi) e `Guida_Metodologica.md`
> (apparato di peer-review). Dove le due fonti divergevano tra loro o contraddicevano la ground truth del
> progetto (spec `config_rq2.yaml`, Ground-Truth Map, architettura GPT-NeoX), il conflitto è stato risolto e
> annotato. Le correzioni e le verifiche di citazione sono tracciate nel **§10 — Registro di riconciliazione**.

---

## 1. Posizionamento epistemologico

Questa tesi si colloca nella **Mechanistic Interpretability**: l'obiettivo è comprendere *come* un modello
linguistico produce un output, non solo *quale*. L'approccio è **correlativo**, non meccanicistico: i risultati
identificano *dove* certe informazioni sono linearmente accessibili nelle rappresentazioni interne, senza
tracciare i circuiti causali che le producono. È il passo che precede il reverse engineering — localizzare
prima di spiegare.

Il modello è **Pythia-1.4B** (EleutherAI), scelto per la trasparenza architetturale e perché progettato per la
ricerca in interpretabilità. I risultati valgono per questo modello, questo tokenizer, questo dominio aritmetico.
Generalizzazioni richiedono replica esplicita.

**Vincoli architetturali da non fraintendere (GPT-NeoX):** 24 layer, d_model 2048, rotary su ~25% delle
dimensioni di testa, **residuo parallelo** (attention e MLP leggono lo stesso input LayerNorm e scrivono
indipendentemente — sono fratelli, non sequenziali), embedding non legati, `pad_id == eos == bos == 0`.
`layer_XX.pt == blocks.X.hook_resid_post`, cioè l'**output del blocco X**, *non* l'embedding (l'embedding vero è
`hook_embed`). Tutta la geometria (isotropy, CKA) è calcolata sul manifold **LayerNorm-folded** (`fold_ln=True`,
fp16); ogni claim geometrico va riferito a quel manifold.

---

## 2. Invarianti di contesto della ricerca (Livello 0)

| ID | Invariante |
|---|---|
| R-I-01 | Tesi triennale, non paper peer-reviewed — standard di rigore accademico, scope limitato |
| R-I-02 | Modello singolo (Pythia-1.4B), dataset sintetico controllato, task aritmetico semplice |
| R-I-03 | Nessun claim di generalizzazione oltre Pythia-1.4B e aritmetica 2-digit è difendibile |
| R-I-04 | Tutti i risultati sono correlativi — nessuna causalità è dimostrabile con questa metodologia |

---

## 3. Research Questions (con razionale e scope dichiarato)

### RQ1 — Differenziazione geometrica per layer

> *In Pythia-1.4B, in quali layer le rappresentazioni interne di stimoli aritmetici mostrano la massima
> divergenza geometrica rispetto a controlli linguistici, misurata tramite isotropy differenziale e CKA
> inter-categoria?*

**Razionale.** Se il modello avesse sviluppato rappresentazioni specializzate per l'aritmetica, lo spazio delle
rappresentazioni matematiche dovrebbe divergere in modo misurabile e localizzabile per layer. ΔIso cattura la
dispersione globale; la CKA inter-categoria cattura la struttura relazionale tra i due sottospazi.

**Strumenti.** Primario: CKA inter-categoria `CKA(H_math^l, H_ctrl^l)`. Secondario: ΔIso(l) = ISO(math,l) −
ISO(ctrl,l), con ISO = media delle similarità coseno tra coppie. ISO alto → anisotropia; ISO basso → isotropia.
ΔIso < 0 = rappresentazioni matematiche *più isotrope* dei controlli.

**Robustezza CKA obbligatoria (assorbe la Ground-Truth Map).** La CKA lineare sovrappesa le direzioni
principali ad alta varianza (Davari et al. 2022; Cloos et al. 2024). Poiché gli stimoli matematici terminano in
"=" e i controlli in parola/".", la **asimmetria posizionale** è una direzione ad alta varianza ed è la
spiegazione alternativa di default per qualsiasi divergenza inter-categoria. Prima di accettare un claim "la
matematica diverge geometricamente dal linguaggio" servono tutti e quattro: (a) **CKA debiased**, (b) baseline
*matched-terminal* `CKA(CTRL-NEU, CTRL-NUM)` e baseline *within-math across-template*, (c) **leave-k-out
influence check**, (d) **cross-check con distanza di Procrustes ortogonale**. Finché non li supera tutti, la
divergenza va trattata come **upper bound** sulla divergenza content-driven, non come finding.

**Scope.** Descrive la divergenza tra stimoli `{a} OP {b} =` e controlli linguistici. Non si conclude che la
struttura osservata sia causalmente responsabile delle capacità aritmetiche.

### RQ2 — Decodificabilità lineare di proprietà aritmetiche

> *Quali proprietà del risultato — segno e parità — sono linearmente decodificabili dagli hidden state di
> Pythia-1.4B, e in quali layer la decodificabilità emerge e raggiunge il picco?*

**Razionale.** La decodificabilità lineare è condizione *necessaria, non sufficiente* perché un'informazione sia
accessibile nel sottospazio lineare delle attivazioni (Belinkov 2022). Segno e parità sono distinti: il segno
dipende dalla struttura degli operandi di input; la parità richiede combinazione aritmetica dei due operandi.

**Lettura interpretativa corretta (assorbe la confound posture).** Il pattern naïf "il segno emerge prima della
parità" va **invertito in interpretazione**: la proprietà che emerge presto (segno) è quella la cui
decodificabilità è *meno* affidabile come evidenza di computazione, perché ai layer iniziali è presente
soprattutto struttura a livello di operando, non un confronto computato. La parità che emerge tardi (~layer 13)
è *più* compatibile con una combinazione genuina di entrambi gli operandi. **Attenzione:** la parità *precoce*
NON va auto-segnalata come leak come si fa per il segno — `parità(risultato)` è essenzialmente la componente di
Fourier a periodo 2 di un numero, e i numeri sono rappresentati su un manifold di Fourier/elicoidale fin dal
layer 0 (Kantamneni & Tegmark 2025). La domanda corretta per la parità precoce è se la direzione di parità si
allinea con la feature T=2, non se è un artefatto.

**Validazione statistica.** Per ogni layer: bootstrap CI 95% (**N = 1000**), permutation test (**N = 1000**,
shuffle delle label sul *solo* train via CV interna, niente leakage sul test), p-value riportato. Correzione
**Benjamini-Hochberg** su **48 test** (24 layer × 2 proprietà). Regolarizzazione probe: **LogisticRegression
lbfgs, C = 10.0** (selezionata da C-sweep), `max_iter=1000`. Split pair-aware: i membri di una coppia
contrastiva (a−b / b−a) restano nella stessa partizione → nessun leakage minimal-pair; test indices congelati
*prima* di ogni training.

**Selectivity — chiarimento critico.** La **selectivity di Hewitt-Liang** (= accuracy(task) − accuracy(control
task), dove il control task usa *gli stessi input con label casuali e una probe riaddestrata*) **non è
calcolata** in questo progetto. Ciò che viene riportato è `ctrl_positive_pred_rate` (frazione di input di
controllo predetti positivi: ≈0.5 = la direzione non si attiva spuriamente su prosa non-aritmetica) — una
**metrica di sanity, non selectivity**. La vera evidenza di validità della probe vive nei moduli confound
**N-01 / N-02** (§6). Nessun claim della tesi deve appoggiarsi su `ctrl_positive_pred_rate` come se fosse
selectivity.

**Extraction point.** Le rappresentazioni sono estratte al token "=" (terminale), *prima* che il modello generi
il risultato. La probe misura quindi la rappresentazione dell'*aspettativa* del risultato, non del risultato
computato — limite interpretativo dichiarato (E-P-02). La giustificazione corretta dell'estrazione al "=" è la
convergenza dell'informazione query-rilevante sul token finale (Nikankin et al. 2024) e il flusso
operando→risultato (Stolfo et al. 2023), **non** ROME (che riguarda factual recall, §9).

**Scope.** Valido per dominio [10,50], sottrazione (CAT-SIGN) e addizione/sottrazione (CAT-PARITY), 3 template
sintatticamente simili. Generalizzazione fuori range / template / operatori richiede verifica esplicita.

### RQ3 — Dinamica geometrica del fine-tuning

> *Il fine-tuning QLoRA su MetaMath produce nelle rappresentazioni residue un drift geometrico superiore alla
> degradazione baseline NF4? In quali layer le proiezioni di attenzione si riorganizzano di più, e questo drift
> correla con il miglioramento su GSM8K?*

**Razionale.** Se l'apprendimento matematico produce riorganizzazione sistematica, il drift dovrebbe essere
non-uniforme tra layer e correlato col miglioramento funzionale (GSM8K). La baseline NF4 separa il segnale di
apprendimento dal rumore di quantizzazione.

**Scope e confound strutturale.** QLoRA è applicato *solo* alle proiezioni `query_key_value`; le MLP sono
congelate. Poiché il residuo è parallelo, congelare la MLP mentre si adatta l'attention è una **separazione
pulita di due percorsi di scrittura indipendenti**, non un taglio parziale di un blocco sequenziale. ROME/MEMIT
localizzano lo *storage fattuale* prevalentemente nelle MLP intermedie: le inferenze di RQ3 sono quindi valide
per la riorganizzazione del pathway di attenzione/trasporto, non per la computazione aritmetica né per lo
storage fattuale. **Soffitto atteso:** QLoRA richiede adapter su *tutti* i layer lineari per eguagliare il full
fine-tuning (Dettmers et al. 2023); il setup QKV-only è atteso *capping* dei guadagni GSM8K — la correlazione
drift↔GSM8K vive in un regime ristretto, da documentare come limite (opzione: contrasto con QLoRA targettato
sulle MLP per separare trasporto da computazione).

**Metriche drift (due formulazioni).** (1) Normalizzata per dimensione: `‖H_ckpt − H_base‖_F / (N·d)`.
(2) Relativa: `‖H_ckpt − H_base‖_F / ‖H_base‖_F` (scala-invariante). Entrambe calcolate separatamente su math e
ctrl. La **relativa** è la metrica di confronto con T16: è scala-invariante, quindi valida nonostante T16 usi
hook HF nativi (senza fold_ln) mentre RQ3 usa TransformerLens (`fold_ln=True`). TransformerLens non può wrappare
modelli NF4 (il fold richiede accesso diretto ai pesi, incompatibile coi tensori quantizzati BitsAndBytes). Le
metriche misurano solo la distanza totale, non la decompongono in rotazione/scala/traslazione.

**Soglie T16.** Le soglie `<3% trascurabile / <5% minore / >5% significativo` sono l'**operazionalizzazione
propria del progetto**, *non* numeri forniti da Dettmers et al. (che riportano recupero di *task performance*,
non soglie di distanza-Frobenius di rappresentazione). Vanno citate come tali. Un drift "grande" è
"gradient explosion" solo se lo dice la curva di loss del training.

**Valutazione funzionale.** GSM8K (Cobbe et al. 2021), **0-shot**, flexible extract. La letteratura riporta
standard a 5-shot: la differenza sistematica stimata è 5–15% inferiore in 0-shot. Per un base model da 1.4B il
risultato 0-shot atteso è prossimo a ~0% (baseline 0.0 corretto e atteso). Tutti i confronti sono interni alla
stessa configurazione.

---

## 4. Dataset — `data/processed/dataset_master_v5.jsonl`

Paradigma **coppie minime contrastive** (BLiMP-inspired): ogni coppia differisce esattamente per la proprietà
target, invarianti tutte le altre caratteristiche superficiali. 3000 stimoli, 4 categorie.

| Categoria | N stimoli | Proprietà | Operatori / operandi |
|---|---|---|---|
| CAT-SIGN | 1000 (500 coppie) | Segno del risultato | Sottrazione, a,b ∈ [10,50] |
| CAT-PARITY | 1000 (500 coppie) | Parità del risultato | Addizione + sottrazione, a,b ∈ [10,50] |
| CTRL-NEU | 500 | — | Prosa inglese, no numeri |
| CTRL-NUM | 500 | — | Contesto numerico non aritmetico |

I controlli usano label sentinella −1 per sign/parity. Tutti gli operandi sono single-token sotto il tokenizer
GPT-NeoX.

**Vincoli tecnici dichiarati.**
- **Operandi [10,50]:** ciascun operando è un singolo token; operandi negativi o >2 cifre richiedono token
  multipli e rompono l'invariante di estrazione.
- **Template:** 3 varianti per categoria → i risultati sono template-specific per costruzione.
- **Pool coverage ~60%** dello spazio delle coppie possibili: il test set della sonda non è OOD rispetto al
  training set della sonda.
- **Bilanciamento 50/50:** artificiale per design (standard nel probing); accuracy misurata in condizioni
  ottimali, il prior reale delle classi non è rappresentato.
- **500 coppie vs 1000 BLiMP:** sotto lo standard, giustificato da vincoli computazionali (BLiMP fornisce solo
  il *razionale minimal-pair*, **non** soglie di accuracy).

---

## 5. Architettura sperimentale

```
Dataset (3000 stimoli)
    │
    ▼
Estrazione hidden states — Pythia-1.4B (TransformerLens, fold_ln=True, fp16)
    │  24 layer × 3000 × 2048 → 24 tensori [3000, 2048]
    │  Strategia: gathered_terminal ("="), right-padding via to_tokens(prepend_bos=True)
    │  Terminale via _last_token_indices (scan da destra), NON [:, -1, :]
    │  layer_XX.pt = blocks.X.hook_resid_post  (output del blocco, non embedding)
    │
    ├──▶ RQ1: ΔIso + CKA inter/evolutionary + apparato robustezza (debiased/Procrustes/influence/baseline)
    │
    ├──▶ RQ2: Linear probing (sign, parity)
    │         LogisticRegression lbfgs, C=10.0, max_iter=1000
    │         Split 80/20 pair-aware, seed deterministico (get_seed)
    │         Validazione: bootstrap CI (1000), permutation test (1000, train-only CV), BH su 48
    │         Confound N-01 (sign): cosine(w_sign, w_op1) + LinReg R²(op1) + Pearson(logit, op1)
    │         Confound N-02 (parity): op2 value/parity decodability + Pearson(logit, op2 parity)
    │
    └──▶ RQ3: QLoRA su MetaMathQA (NF4, r=16, α=32, QKV-only, MLP congelate)
              ~25 checkpoint (save ogni 500 step, ~12k step) → merge → estrazione → frozen probe accuracy
              Frobenius drift per layer (math vs ctrl separati, normalizzata + relativa)
              GSM8K 0-shot a ogni checkpoint; baseline NF4 = T16 (nf4_degradation.py)
```

---

## 6. Validazione confound — `run_confound_checks.py` / `run_parity_confound_checks.py`

Script diagnostici standalone (non chiamati dagli orchestratori), eseguiti dopo RQ2. Sono la **vera evidenza di
validità della probe** (non `ctrl_positive_pred_rate`).

**N-01 (sign).** Il segno decodifica un confronto astratto o la magnitudine di operand1? In-data `Pearson(op1,
sign) ≈ 0.58` e op1>op2 concorda col segno il 100% delle volte per costruzione: la decodificabilità *precoce*
del segno va trattata come **shortcut di magnitudine finché** il controllo su operand1 non la scagiona con
significatività. V1: R² LinReg di op1 dagli hidden state (perm-gated). V2: cosine(w_sign, w_op1). V3:
Pearson(logit sign frozen, op1). BH sui p-value di R²(op1) tra layer.

**N-02 (parity).** La parità decodifica la parità del risultato o quella di operand2? V1: decodificabilità del
valore di op2 (R², permutato). V2: decodificabilità della parità di op2 (accuracy + cosine). V3: Pearson(logit
parity frozen, parità op2). Diagnostica di protezione del dataset: bilanciamento parità del primo operando +
`corr(parità_risultato, parità_op2)` ground-truth.

Entrambi i moduli includono l'assertion index-space: ordine `metadata.stimuli_ids` == ordine righe JSONL.

---

## 7. Tabella dei principi epistemologici

### Livello 1 — Principi epistemologici trasversali

| ID | Principio | Fonte | Regola | Segnale di violazione |
|---|---|---|---|---|
| **E-O-01** | Distinguere correlazione da causazione | Belinkov (2022) §4 | Ogni "il modello *usa* X" → "X è *linearmente decodificabile* da Y" | "il modello calcola il segno al layer 3" invece di "il segno è decodificabile al layer 3" |
| **E-O-02** | Construct validity | Cronbach & Meehl (1955); adattato da Hewitt & Liang (2019) | Una sonda che predice X prova che X è nell'hidden state — non che X *guidi* il comportamento | Accuracy alta ≠ informazione computazionalmente rilevante |
| **E-O-03** | Controllo dei confound | Campbell & Stanley (1963) | Ogni differenza tra condizioni deve avere un'alternativa confound esplicita e quantificata | N-01/N-02 documentati ma *non* discussi in termini di impatto quantitativo |
| **E-O-04** | Standard di replicazione | Gundersen & Kjensmo (2018) | Seed fissi, dataset versionato, codice pubblico — necessario non sufficiente | Risultati che variano >2% su seed diversi |

### Livello 2 — ML Research Standards

| ID | Principio | Fonte | Regola | Segnale di violazione |
|---|---|---|---|---|
| **E-M-01** | Baseline comparison | Standard ML | Ogni metrica confrontata con: (a) chance 0.5, (b) majority class, (c) null da permutazione, (d) probe di controllo sull'operando. *(NB: Quirke & Barez NON è un baseline valido — vedi §9.)* | Accuracy su sign senza confronto con majority/permutation null/operand control |
| **E-M-02** | Significatività statistica | Cohen (1988); permutation testing | Riportare CI e p-value; per probing il permutation test con label shuffle è il gold standard | Accuracy senza CI o senza p-value |
| **E-M-03** | Effect size > significatività | Cohen (1988); Wasserstein et al. (2019) | Riportare magnitudine, non solo significatività (0.52 con p<0.05 è irrilevante) | Peak accuracy senza differenza dal baseline random |
| **E-M-04** | Probe selectivity | Hewitt & Liang (2019) §3 | Selectivity vera = same-input + *label casuali* + probe riaddestrata. **NON calcolata qui**: `ctrl_positive_pred_rate` è sanity, non selectivity; l'evidenza vera è N-01/N-02 | Claim che si appoggia su `ctrl_positive_pred_rate` come selectivity |
| **E-M-05** | Multiple comparison correction | Benjamini & Hochberg (1995) | 48 test (24 layer × 2 prop) → correzione FDR | BH non applicata |

### Livello 3 — Probing Methodology Specific

| ID | Principio | Fonte | Regola | Segnale di violazione |
|---|---|---|---|---|
| **E-P-01** | Complessità della probe come variabile | Hewitt & Liang (2019); Pimentel et al. (2020) | La scelta lineare (logistic reg.) è una decisione forte; una MLP probe molto migliore indicherebbe non-linearità | MLP probe >> linear probe |
| **E-P-02** | Giustificazione dell'extraction point | Nikankin et al. (2024); Stolfo et al. (2023) | L'estrazione al "=" si giustifica con la convergenza sul token finale (Nikankin) e il flusso operando→risultato (Stolfo). ROME solo per contrasto (factual recall, mapping non diretto) | Estrazione al "=" giustificata *positivamente* via ROME |
| **E-P-03** | Contaminazione train/test | Standard ML | I pesi della probe non devono vedere il test; test indices salvati prima del training ✓ | Mancata verifica che MetaMath non contenga esempi identici |
| **E-P-04** | Generalizzazione OOD | Zhang et al. (2025) | Una probe su un template non generalizza per forza; claim qualificati come "su questo template" | Risultati template-specific presentati come generali |

### Livello 4 — Geometric/Representational Analysis Specific

| ID | Principio | Fonte | Regola | Segnale di violazione |
|---|---|---|---|---|
| **E-G-01** | Interpretazione dell'isotropy | Ethayarajh (2019); Mu & Viswanath (2018) | L'anisotropia è lo stato *normale* degli hidden state, non una struttura semantica né un collasso. ΔIso è differenza relativa, mai misura assoluta; riportare con floor random-Gaussian norm-matched e CI | "matematica più strutturata perché più anisotropa"; ΔIso<0 letto come "feature extraction specializzata" |
| **E-G-02** | Interpretazione della CKA | Kornblith et al. (2019); Davari et al. (2022); Cloos et al. (2024) | CKA misura similarità, non qualità. CKA lineare sovrappesa direzioni ad alta varianza → usare **debiased CKA** + Procrustes + leave-k-out + baseline matched-terminal/within-math. La asimmetria posizionale "=" è l'alternativa di default | CKA descritta come "divergenza" senza i quattro controlli e senza baseline di varianza |
| **E-G-03** | Definizione del layer di emergenza | Shai et al. (2024); Nikankin et al. (2024); Olsson et al. (2022) | "Emergence layer" (def. operativa: primo layer con accuracy >0.7) va esplicitato; in tensione col profilo distribuito ("bag of heuristics"; belief state su più layer). **Preferire il profilo di decodificabilità a un punto singolo** | Emergence layer come punto singolo senza def. operativa né profilo |
| **E-G-04** | Cambiamento geometrico ≠ cambiamento di capacità | RQ3 limitation | Il drift Frobenius misura cambiamento dei pesi, non delle capacità; correlazione drift↔ΔGSM8K richiede cautela (terze variabili: distribuzione MetaMath vs GSM8K) | Linguaggio causale tra drift e performance |

### Livello 5 — Fine-Tuning & Evaluation Standards

| ID | Principio | Fonte | Regola | Segnale di violazione |
|---|---|---|---|---|
| **E-F-01** | Coerenza few-shot | Brown et al. (2020); Cobbe et al. (2021) | GSM8K è tipicamente 5-shot; qui 0-shot → numeri sistematicamente più bassi (~5–15%), da dichiarare | Confronto con la letteratura GSM8K senza dichiarare 0-shot vs 5-shot |
| **E-F-02** | Sufficienza delle epoch | Training dynamics | 1 epoch su MetaMath a batch effettivo 32 (8×4) ≈ 12k step; ~3.1M parametri trainable (LoRA r=16, QKV-only). Confrontare il training loss finale con la letteratura | Training loss finale dichiarato buono senza reference |
| **E-F-03** | Trasparenza sulla quantizzazione | Dettmers et al. (2023) | NF4 introduce degradazione; T16 la misura e va citato coi risultati RQ3 | Risultati RQ3 senza disclosure del baseline NF4 |

---

## 8. Reference Table — claim chiave per paper

*Per ogni paper: claim rilevante + implicazione per la tesi. Le voci marcate **[verificato]** sono state
controllate via web il 31/05/2026 contro fonte primaria. (Nessuna voce resta non verificata.)*

| Paper | Claim rilevante | Implicazione |
|---|---|---|
| Belinkov (2022) *Probing Classifiers* | "High probe accuracy does not entail the property is used by the model" | Ogni claim RQ2 qualificato come "linearmente decodificabile" |
| Hewitt & Liang (2019) *Control Tasks* | Selectivity = acc(task) − acc(control task con label casuali) | La selectivity vera **non è calcolata**; l'evidenza è N-01/N-02, non `ctrl_positive_pred_rate` |
| Kantamneni & Tegmark (2025) *Language Models Use Trigonometry to Do Addition* (arXiv 2502.00873, MIT) **[verificato]** | I numeri vivono su un'elica generalizzata (Fourier + lineare) dal layer 0; rappresentazione causalmente implicata per addizione **e** sottrazione. **L'elica ha periodi T=[2,5,10,100] + componente lineare (confermato dal paper)** | **Lente primaria per RQ2.** La parità ≈ componente Fourier T=2 (esiste esplicitamente nel paper): testare se la direzione di parità è nello span del T=2; il segno richiede la fase relativa di *entrambi* gli operandi |
| Nikankin, Reusch, Mueller & Belinkov (2024) *Arithmetic Without Algorithms* (arXiv 2410.21272, Technion; ICLR 2025) **[verificato]** | Aritmetica = euristiche distribuite ("bag of heuristics"), non un algoritmo pulito; la risposta è estraibile con alta accuracy solo dalla posizione finale, dopo i layer tardivi | Giustifica estrazione al "=" e la scelta QKV (attention = trasporto); spinge verso il **profilo** di decodificabilità, non un punto |
| Stolfo, **Belinkov** & Sachan (2023) *A Mechanistic Interpretation of Arithmetic Reasoning … via Causal Mediation Analysis* (arXiv 2305.15054; EMNLP 2023) **[verificato — autori corretti]** | Evidenza di mediazione causale: l'informazione query-rilevante è trasmessa dai layer iniziali di mid-sequence al token finale via attention | Corrobora l'estrazione al "="; template per claim causali (ablate L, re-probe L+k). *(Correzione: NON "Stolfo, Jin, Shu & Sachan" — vedi §10 #14)* |
| Kornblith et al. (2019) *CKA* | CKA lineare invariante a trasf. ortogonali e scaling isotropo, **non** anisotropo | `fold_ln` ripiega γ per-feature nei pesi → CKA può shiftare; misurare l'effetto fold_ln (True vs False) |
| Davari, Horoi, Natik, Lajoie, Wolf & Belilovsky (2022) *Reliability of CKA as a Similarity Measure* (arXiv 2210.16156; ICLR 2023) **[verificato]** + Cloos, Siegel, Brincat, Miller & Cueva (2024) *Differentiable optimization of similarity scores between models and brains* (ICLR 2024 Workshop on Representational Alignment) **[verificato]** | CKA lineare sovrappesa direzioni ad alta varianza, sensibile a outlier, manipolabile (i punteggi di similarità si possono ottimizzare verso valori arbitrari) senza cambiamento funzionale, nessuna soglia universale | Usare debiased CKA + leave-k-out + Procrustes; riportare più metriche; la posizione "=" è il sospetto principale ad alta varianza |
| Ethayarajh (2019) *How Contextual…* | Embedding contestuali anisotropi di default | ΔIso<0 atteso, non collasso; mai letto come "feature extraction" (E-G-01) |
| Shai, Marzen, Teixeira, Oldenziel & Riechers (2024) *Belief State Geometry* (NeurIPS 2024, arXiv 2405.15943) **[verificato]** | Le belief state sono distribuite su più layer — non c'è un "layer della conoscenza" singolo | Giustifica `all_layers`; tensione col concetto di emergence_layer singolo *(correzione: NON "Lindsey et al.")* |
| Quirke & Barez (2023) *Understanding Addition in Transformers* (arXiv 2310.13121) **[verificato]** | Transformer **a un layer** addestrato *from scratch*; "double staircase" che cammina sulle posizioni di **digit-token** | Gli operandi della tesi sono **single-token** → nessuno staircase. **Solo lassamente analogo, MAI un confronto per i layer di emergenza.** *(correzione: non ICLR 2024)* |
| Meng et al. (2022) *ROME* | Storage fattuale all'ultimo subject token, MLP intermedie | **Solo per contrasto**: spiega perché QLoRA MLP-frozen è scoped fuori dagli edit di storage fattuale. NON giustifica l'estrazione al "=" (categoria diversa) |
| Tenney et al. (2019) *BERT rediscovers the NLP pipeline* | Proprietà linguistiche diverse emergono a layer diversi in modo consistente | Pattern compatibile con RQ2 (sign nei layer iniziali, parity negli intermedi) |
| Olsson et al. (2022) *Induction Heads* | Componenti specifici mediano comportamenti specifici | Il nostro approccio è correlativo, non mechanistic — limite da discutere |
| Dettmers et al. (2023) *QLoRA* | NF4 recupera la **task performance** a 16-bit; eguagliare il full FT richiede adapter su **tutti** i layer lineari (QKV-only sotto-performa). **Non** fornisce soglie di Frobenius di rappresentazione | Le soglie T16 `<3%/<5%` sono **nostre**, non di Dettmers; il setup QKV-only è atteso *capping* dei guadagni → regime ristretto per la correlazione drift↔GSM8K |
| Biderman et al. (2023) *Pythia* (arXiv 2304.01373; ICML 2023) **[verificato]** | 154 checkpoint pubblici per ciascuno dei 16 modelli, dataloader ricostruibili | Risorsa inutilizzata: sondare la traiettoria di pretraining (quando emergono sign/parity) è l'esperimento Pythia canonico (collega a grokking, Nanda et al. 2023) |
| Pimentel et al. (2020) *Information-theoretic probing* | MDL più robusto dell'accuracy | Non implementato — limite (opzionale: online-code MDL) |
| Cobbe et al. (2021) *GSM8K* | Anche GPT-3 175B fatica su GSM8K; fine-tuned 175B + verifier ≈ 55%. *(Il 58.1% spesso citato è CoT prompting, Wei et al. 2022 / scala PaLM — NON GPT-3 175B 5-shot.)* | Per un base model 1.4B in 0-shot il risultato atteso è ~0% (baseline 0.0 corretto) |
| Warstadt et al. (2019) *BLiMP* | 1000 coppie per sub-dataset come standard | Noi 500 coppie: sotto lo standard (vincoli computazionali). **BLiMP fornisce solo il razionale minimal-pair, NON soglie di accuracy** |
| Yu et al. (2023) *MetaMath* | Bootstrap di domande matematiche da GSM8K/MATH | Possibile overlap distribuzionale MetaMath↔GSM8K → confound da dichiarare in RQ3 |

---

## 9. Reference-class da NON usare come benchmark

- **ROME (Meng 2022) / MEMIT (Meng 2023):** localizzano il *factual recall* nelle MLP intermedie. Non
  applicabile: aritmetica ≠ fatti; estrazione al "=" (operatore), non all'ultimo subject token; RQ3 congela le
  MLP che ROME implica. Non confrontare `l*` con un range "8–16 MLP" — sarebbe un errore di categoria. Citare
  ROME solo per contrasto.
- **Quirke & Barez (2023):** transformer a un layer, operandi multi-digit-token. La tesi ha operandi single-token
  → nessuno staircase. Lassamente analogo, mai un punto di confronto per i layer di emergenza. *(Nota: non
  esiste un paper Quirke & Barez su musica/chiavi; se compare una citazione del genere è fabbricata.)*
- **BLiMP (Warstadt 2020):** ispira il design minimal-pair ma non fornisce soglie di accuracy di probing.

---

## 10. Registro di riconciliazione (conflitti risolti tra le due fonti)

| # | Voce | README diceva | Guida / ground truth diceva | Risoluzione |
|---|---|---|---|---|
| 1 | Regolarizzazione probe | `C = 1.0` | spec `config_rq2.yaml`: **C = 10.0** | Corretto a **10.0** (SSOT = spec) |
| 2 | Campioni bootstrap | `N = 2000` | spec: **1000** | Corretto a **1000** |
| 3 | Selectivity | "riportata" come acc(task)−acc(ctrl) | Ground truth: **non calcolata**; esiste `ctrl_positive_pred_rate` (sanity) + N-01/N-02 | Riformulato: selectivity H&L non calcolata; sanity ≠ selectivity |
| 4 | Nome confound sign | "Confound **T02**" | spec: **N-01** | Standardizzato a **N-01** (T02 = alias) |
| 5 | Layer 0 | Guida: "layer 0 = layer di **embedding**" | `layer_00` = `blocks.0.hook_resid_post` = **output del blocco 0** | Corretto: above-chance al layer_00 è **atteso**, non bug |
| 6 | Quirke & Barez | usato come baseline / "ICLR 2024" / staircase compatibile | **[verificato]** arXiv 2310.13121, 1-layer toy, digit-token | Rimosso come baseline; declassato a "lassamente analogo" |
| 7 | Belief state geometry | (assente in README) | Guida: "**Lindsey et al. (2024)**" | **[verificato]** → **Shai et al. (NeurIPS 2024)**, arXiv 2405.15943 |
| 8 | GSM8K GPT-3 | (assente) | Guida: "GPT-3 175B 5-shot = **58.1%**" | **[verificato]** conflazione: 58.1% è CoT prompting (Wei 2022); corretto |
| 9 | Soglie Frobenius | (assente) | Guida: Dettmers "Frobenius rel. <3%" | Dettmers non fornisce soglie di rappresentazione; soglie T16 = **nostre** |
| 10 | Estrazione al "=" | giustificata via ROME | ROME = factual recall (categoria diversa) | Giustificazione → Nikankin/Stolfo; ROME solo per contrasto |
| 11 | Robustezza CKA | (assente in entrambi) | Ground-Truth Map: debiased + Procrustes + leave-k-out + matched-terminal | Aggiunto a RQ1 ed E-G-02 |
| 12 | Asimmetria posizionale | (assente in entrambi) | spec + Ground-Truth Map | Aggiunta come confound primario CKA e come limite dichiarato |
| 13 | Lente arithmetic regime-matched | (assente in entrambi) | Kantamneni & Tegmark, Nikankin, Stolfo | Aggiunti alla reference table |
| 14 | **Autori di Stolfo et al. 2023** | (assente) | Ground-Truth Map: "Stolfo, **Jin, Shu** & Sachan" | **[verificato 31/05/2026]** Errore nella mappa. arXiv 2305.15054 (EMNLP 2023) è **Stolfo, Belinkov & Sachan**. "Jin/Shu" appartengono a un *altro* paper Stolfo 2023 (*A causal framework to quantify the robustness…*, ACL 2023: Stolfo, Jin, Shridhar, Schölkopf, Sachan — e "Shu" è "Shridhar" mutilato). Corretto in §8 |
| 15 | Verifica delle citazioni importate dalla mappa | marcate "da verificare" | — | **[fatto 31/05/2026]** Kantamneni & Tegmark (2502.00873), Nikankin (2410.21272), Davari (2210.16156), Biderman (2304.01373) confermate accurate; Stolfo (2305.15054) con autori corretti (#14); Cloos et al. 2024 (Cloos, Siegel, Brincat, Miller & Cueva, ICLR 2024 Workshop on Representational Alignment) confermata. **Nessuna citazione resta non verificata** |

---

## 11. Limiti dichiarati (unione delle due fonti + spec)

| Limite | Tipo | Riferimento |
|---|---|---|
| Singolo modello (Pythia-1.4B) | Scope | R-I-02 |
| Singolo dominio/3 template, [10,50] | Scope / Generalizzabilità | E-P-04; Zhang et al. 2025 |
| Approccio correlativo, non causale | Epistemologico | E-O-01, E-G-04 |
| Estrazione all'aspettativa, non al risultato | Metodologico | E-P-02 |
| QLoRA QKV-only, MLP congelate | Strutturale | RQ3 scope |
| 0-shot GSM8K vs 5-shot standard (~5–15% più basso) | Comparabilità | E-F-01 |
| Overlap distribuzionale MetaMath/GSM8K | Contaminazione | Yu et al. 2023 |
| Pool coverage ~60% — test set non OOD | Generalizzabilità | Dataset §4 |
| 500 coppie vs 1000 BLiMP | Sample size | Vincoli computazionali |
| **Asimmetria posizionale: math finisce in "=", ctrl in parola/"."** | RQ1 caveat | commento in `extract_states.py`; confound primario CKA |
| T16 (NF4) usa hook HF nativi, non TransformerLens | Comparabilità | la Frobenius relativa risolve la scala |
| Selectivity H&L non calcolata (solo sanity + N-01/N-02) | Metodologico | E-M-04 |

---

## 12. Handoff — peer review metodologico

**Ruolo:** reviewer anonimo di conferenza NLP (ACL/EMNLP/ICLR). Target di rigore: tesi triennale di eccellenza,
non paper peer-reviewed. Ogni deviazione va identificata, classificata come *limite difendibile* o *flaw*, e
mitigata.

**Accesso:** codebase navigabile; file di progetto; web search per verifica di claim; questo documento come
guida.

### Domande aperte prioritarie (corrette)

**P0 — potenzialmente invalidanti**
1. **Decodificabilità del segno ai layer iniziali** (incluso `layer_00`): shortcut di magnitudine vs confronto
   genuino? *NB: `layer_00` è l'output del blocco 0 — l'attention ha già mischiato i due operandi, quindi
   above-chance è atteso e NON prova "info nell'input".* Da risolvere con N-01 (R² op1, cosine, logit-Pearson)
   prima di qualsiasi claim. Riferimento: Tenney et al. (2019) per il pattern per-layer.
2. **Confound N-01 con misura diretta:** la cosine dei pesi è indiretta. Esiste già il test diretto
   `Pearson(logit sign frozen, op1)` e R² di op1 — vanno *riportati con BH*, non solo la cosine.
3. **Robustezza della divergenza CKA inter-categoria:** sopravvive a debiased + matched-terminal +
   within-template + leave-k-out + Procrustes? Senza, è solo asimmetria posizionale.

**P1 — limiti da documentare con precisione**
4. **Correzione multipla** già applicata (BH su 48). Verificare quanti risultati sopravvivono post-BH.
5. **0-shot vs 5-shot GSM8K:** stima della differenza sistematica (~5–15%) dichiarata.
6. **Soffitto QKV-only** (Dettmers): la correlazione drift↔GSM8K vive in regime ristretto; documentare, eventuale
   contrasto MLP-targeted.

**P2 — qualificazioni linguistiche**
7. Nessun claim causale per risultati correlativi (E-O-01).
8. "Emergence layer" definito operativamente (soglia 0.7) **e** affiancato dal profilo di decodificabilità.

### Output atteso

1. **Executive summary** — il claim principale è difendibile? Sì / No / Con qualificazioni.
2. **Per ogni RQ** — forza delle evidenze, limiti critici, claim da riformulare.
3. **Tabella dei fix** ordinata P0/P1/P2 con la mitigazione più economica (single-GPU, single-model, scope
   triennale).
4. **Suggested additions** — analisi aggiuntive (anche semplici) che rafforzerebbero la tesi (es.: profilo di
   decodificabilità sui checkpoint di pretraining Pythia; allineamento direzione-parità con T=2 Fourier;
   contrasto QLoRA MLP-targeted).