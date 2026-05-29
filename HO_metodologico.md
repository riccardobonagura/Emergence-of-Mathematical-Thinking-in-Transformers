**Approccio ottimale — combinazione di tre strumenti:**

1. Una **tabella dei principi epistemologici** (come quella di design) — guida permanente a basso costo token
2. Una **reference table di claim chiave** — per ogni paper rilevante, 1-2 frasi con il claim specifico che impatta la nostra metodologia (~50 token per paper vs 7,000)
3. **Web search** per verifica di numeri specifici e paper non in training data

---

## Tabella dei principi epistemologici — ML Research Standards

### Livello 0 — Research Context Invariants

|ID|Invariante|
|---|---|
|R-I-01|Tesi triennale, non paper peer-reviewed — standard di rigore accademico ma scope limitato|
|R-I-02|Modello singolo (Pythia-1.4B), dataset sintetico controllato, task aritmetico semplice|
|R-I-03|Nessun claim di generalizzazione oltre Pythia-1.4B e aritmetica 2-digit è difendibile|
|R-I-04|Tutti i risultati sono correlativi — nessuna causalità è dimostrabile con questa metodologia|

---

### Livello 1 — Overarching Epistemological Principles

|ID|Principio|Fonte|Regola|Segnale di violazione|
|---|---|---|---|---|
|**E-O-01**|**Distinguish correlation from causation**|Philosophy of science standard; Belinkov (2022) §4|Ogni claim di "il modello _usa_ X" deve essere riformulato come "X è _linearmente decodificabile_ da Y"|Frasi tipo "il modello calcola il segno al layer 3" invece di "il segno è decodificabile al layer 3"|
|**E-O-02**|**Construct validity**|Cronbach & Meehl (1955); adattato a NLP da Hewitt & Liang (2019)|Una sonda che predice X prova che X è nell'hidden state — non prova che X _guida_ il comportamento|Accuracy alta ≠ informazione computazionalmente rilevante|
|**E-O-03**|**Confound control**|Campbell & Stanley (1963) design sperimentale classico|Ogni differenza osservata tra condizioni deve avere un'alternativa confound esplicitamente discussa e quanto possibile controllata|N-01 e N-02 documentati — ma sono discussi in termini di impatto quantitativo?|
|**E-O-04**|**Replication standard**|Reproducibility crisis literature; Gundersen & Kjensmo (2018) ML reproducibility|Seeds fissi, dataset versionato, codice pubblico — condizione necessaria ma non sufficiente per replicabilità|Se i risultati variano di >2% su seed diversi, la robustezza è dubbia|

---

### Livello 2 — ML Research Standards

|ID|Principio|Fonte|Regola|Segnale di violazione|
|---|---|---|---|---|
|**E-M-01**|**Baseline comparison**|Standard ML evaluation|Ogni metrica deve essere confrontata con almeno: (a) random baseline, (b) majority class baseline, (c) risultati pubblicati su task comparabile|Accuracy 1.0 su sign — è confrontata con majority baseline? Con risultati di Quirke & Barez?|
|**E-M-02**|**Statistical significance**|Cohen (1988); permutation testing standard|I risultati devono riportare CI e p-value. Per probing: permutation test con label shuffle è il gold standard|Accuracy senza CI è un numero vuoto — abbiamo bootstrap CI ma è documentato il p-value?|
|**E-M-03**|**Effect size over significance**|Cohen (1988); Wasserstein et al. (2019)|Riportare magnitude dell'effetto, non solo se è significativo. Un'accuracy di 0.52 con p<0.05 è irrilevante.|Peak accuracy senza discussione della differenza rispetto al baseline random|
|**E-M-04**|**Probe selectivity**|Hewitt & Liang (2019) §3|Una sonda dovrebbe avere accuracy alta sul task target e bassa su control tasks. Senza control task la sonda potrebbe memorizzare artefatti del dataset.|Sign probe accuracy = 1.0 al layer 0 — questo è plausibile o è un artefatto?|
|**E-M-05**|**Multiple comparison correction**|Benjamini & Hochberg (1995)|Se si testa la stessa ipotesi su 24 layer, il p-value threshold va corretto (Bonferroni o FDR)|24 layer × 2 properties = 48 test — correction applicata?|

---

### Livello 3 — Probing Methodology Specific

|ID|Principio|Fonte|Regola|Segnale di violazione|
|---|---|---|---|---|
|**E-P-01**|**Probe complexity as variable**|Hewitt & Liang (2019); Pimentel et al. (2020)|La scelta di logistic regression (lineare) è una decisione metodologica forte che deve essere giustificata. Una MLP probe che ottiene accuracy molto superiore suggerisce che l'informazione non è linearmente accessibile.|Se MLP probe >> linear probe: linearità non è la struttura giusta|
|**E-P-02**|**Extraction point justification**|Meng et al. (2022) ROME §3|L'estrazione al token "=" deve essere giustificata con reference alla letteratura, non solo a ROME (che è su factual recall, non arithmetic). Il mapping da "factual last subject token" a "arithmetic = token" richiede discussione.|ROME parla di subject token per factual recall — è direttamente applicabile all'aritmetica?|
|**E-P-03**|**Training/test contamination**|Standard ML|I pesi della sonda non devono essere influenzati dai dati di test. Save_test_indices prima del training è corretto — ma i QOL-02 checks lo verificano formalmente?|Test indices salvati prima del training ✓ — ma verificato che il modello non ha visto esempi identici in MetaMath?|
|**E-P-04**|**OOD generalization**|Zhang et al. (2025) probing §5|Una sonda che funziona su un template non generalizza necessariamente ad altri. Claim sulla struttura geometrica devono essere qualificati come "su questo specifico template"|Tutti gli stimoli usano lo stesso template aritmetico — i risultati sono template-specific?|

---

### Livello 4 — Geometric/Representational Analysis Specific

|ID|Principio|Fonte|Regola|Segnale di violazione|
|---|---|---|---|---|
|**E-G-01**|**Isotropy interpretation**|Ethayarajh (2019); Mu & Viswanath (2018)|Anisotropia ≠ struttura semantica ricca. Anisotropia può derivare da distribuzione sbilanciata dei token nel corpus di training. La ΔIso deve essere interpretata come differenza relativa tra categorie, non come misura assoluta.|Claims tipo "le rappresentazioni matematiche sono più strutturate perché più anisotrope" senza qualifier|
|**E-G-02**|**CKA interpretation**|Kornblith et al. (2019) §4; Nguyen et al. (2021)|CKA misura similarity tra rappresentazioni, non qualità. CKA = 0.919 al layer 23 tra math e ctrl non significa "divergenza" — significa che math e ctrl hanno rappresentazioni simili al 91.9%. La divergenza di 0.07 rispetto al layer precedente deve essere messa in scala con varianza attesa.|CKA 0.919 descritto come "divergenza significativa" senza confronto con varianza baseline|
|**E-G-03**|**Emergence layer definition**|Lindsey et al. (2024) §2; Olsson et al. (2022)|"Layer di emergenza" è un termine della letteratura con definizioni diverse. La nostra definizione (primo layer con accuracy > 0.7) deve essere esplicitata e confrontata con definizioni alternative.|Emergence layer usato senza citare la definizione operativa|
|**E-G-04**|**Geometric change ≠ capability change**|RQ3 limitation (documentato)|Il drift Frobenius misura cambiamento dei pesi, non cambiamento delle capacità. Correlazione tra drift geometrico e Δ GSM8K richiede cautela — terze variabili confondenti (es. distribuzione MetaMath vs GSM8K).|Qualsiasi causal language tra drift e performance|

---

### Livello 5 — Fine-Tuning & Evaluation Standards

|ID|Principio|Fonte|Regola|Segnale di violazione|
|---|---|---|---|---|
|**E-F-01**|**Few-shot consistency**|Brown et al. (2020) GPT-3 §3; GSM8K paper (Cobbe et al. 2021)|GSM8K è tipicamente valutato 5-shot in letteratura. Noi usiamo 0-shot. Questo abbassa i numeri sistematicamente — va dichiarato esplicitamente con stima dell'impatto (tipicamente 5-15% di differenza).|Confronto risultati con letteratura GSM8K senza dichiarare 0-shot vs 5-shot|
|**E-F-02**|**Epoch sufficiency**|Training dynamics literature|1 epoch su MetaMath ~395k esempi a batch effettivo 32 = ~12k steps. Per un modello da 1.4B con 3.1M parametri trainable questo è insufficiente per saturare la convergenza. Il training loss finale (2.56) va confrontato con training loss di modelli simili in letteratura.|Training loss 2.56 dichiarato come buon risultato senza reference|
|**E-F-03**|**Quantization transparency**|Dettmers et al. (2023) QLoRA paper §4|NF4 introduce degradazione sistematica rispetto a FP16. T16 (nf4_degradation.py) misura questa degradazione — i risultati di T16 devono essere citati quando si riportano risultati RQ3.|Risultati RQ3 senza disclosure del NF4 baseline degradation|

---

## Reference Table — Key Paper Claims

_Per ogni paper: claim specifico rilevante alla nostra metodologia + implicazione critica._

|Paper|Claim rilevante|Implicazione per la tesi|
|---|---|---|
|Belinkov (2022) _Probing classifiers: promises, shortcomings, and advances_|"High probe accuracy does not entail that the probed property is used by the model"|Ogni claim di RQ2 va qualificato come "linearmente decodificabile", mai "computazionalmente usato"|
|Hewitt & Liang (2019) _Designing probes with control tasks_|Propongono selectivity = accuracy(task) - accuracy(control task) come metrica robusta|Abbiamo control task? (CTRL-NEU, CTRL-NUM) — ma selectivity è calcolata formalmente?|
|Kornblith et al. (2019) _Similarity of neural network representations_|CKA lineare è invariante a trasformazioni ortogonali e scaling isotropo — non a scaling non-isotropo|La nostra CKA è lineare: questa invarianza è una feature o un problema per misurare cambiamenti geometrici?|
|Quirke & Barez (ICLR 2024) _Understanding Addition in Transformers_|"Double staircase" attention pattern da "=" position; circuiti diversi per digit class diversi|Giustifica la scelta [10,50] 2-digit; i nostri layer di emergence sono compatibili con la loro analisi circuitale?|
|Meng et al. (2022) ROME|Factual integration avviene all'ultimo subject token nei layer MLP intermedi (layers 3-8 GPT-J scale)|Giustifica l'estrazione al "="; ma "=" è subject token o operator token per l'aritmetica? Il mapping non è diretto.|
|Ethayarajh (2019) _How contextual are contextualized word representations_|Embedding BERT/GPT sono anisotropi di default — occupano un cono nello spazio|ΔIso negativo (math più anisotropo) potrebbe riflettere densità diversa nel corpus, non struttura matematica|
|Lindsey et al. (NeurIPS 2024) _Belief State Geometry_|Le belief state si distribuiscono su più layer — non c'è un singolo "layer della conoscenza"|Giustifica probe_layer_strategy="all_layers"; tensione con il concetto di emergence_layer singolo|
|Zhang et al. (2025) _Probing Hidden States_|Probe OOD generalization è limitata — math→logic fallisce|Documentata come limitazione; ma è stata testata su almeno 2 template diversi?|
|Cobbe et al. (2021) _Training verifiers to solve math word problems_ (GSM8K)|Baseline GPT-3 (175B) 5-shot = 58.1%. Modelli <2B: << 10% 5-shot|Il nostro 0-shot è atteso vicino a 0% per un base model — baseline 0.0 è corretto e atteso|
|Dettmers et al. (2023) QLoRA|NF4 con r=16 su modelli ~1-7B: degradazione Frobenius relativa tipicamente < 3%|T16 deve confermare questo — se > 5% la degradazione confonde RQ3|
|Pimentel et al. (2020) _Information-theoretic probing_|Probing come information theory: MDL (minimum description length) è più robusto di accuracy|Non implementato — limitazione da documentare|
|Tenney et al. (2019) _BERT rediscovers the classical NLP pipeline_|Proprietà linguistiche diverse emergono a layer diversi in modo consistente|Pattern compatibile con nostra RQ2 — sign al layer 3, parity al layer 13-14|
|Olsson et al. (2022) _In-context learning and induction heads_|Mechanistic interpretability mostra che componenti specifici (induction heads) mediano comportamenti specifici|Il nostro approccio è correlativo, non mechanistic — limitazione da discutere|
|Warstadt et al. (2019) BLiMP|1000 coppie per sub-dataset come standard per contrastive probing|Noi: 500 coppie per categoria — sotto lo standard BLiMP; giustificare con vincoli computazionali|

---

## HANDOFF — Peer Review Metodologico

**Progetto:** Geometric Dynamics in Transformer Internal Representations **Tesi:** Laurea Triennale · Bonagura N46007216 · Federico II Napoli **Ruolo istanza:** Peer Reviewer — analisi metodologica e quantitativa

---

### Obiettivo

Revisionare criticamente la metodologia e i risultati della tesi come farebbe un reviewer anonimo per una conferenza NLP (ACL, EMNLP, ICLR). Il target di rigore è quello di una tesi triennale di eccellenza, non di un paper peer-reviewed — ma ogni deviazione dagli standard accademici deve essere identificata, classificata come _limitazione difendibile_ o _flaw metodologico_, e suggerita una mitigazione.

---

### Accesso

- Codebase completa navigabile
- `abstract_updated.docx` e `onboarding_guide_updated.docx` in project files
- Web search per verifica di claim specifici
- Tabella dei principi epistemologici (sopra) come guida

---

### Risultati da revisionare

**RQ1 — Isotropy + CKA:**

- ΔIso negativo layers 7–22, minimo ≈ −0.12 a layer 14
- CKA math al layer 23: 0.919 vs ctrl 0.989 (Δ=0.070)

**RQ2 — Linear Probing:**

- Sign: emergence layer 0, peak layer 3, accuracy 1.0
- Parity: emergence layer 13, peak layer 14, accuracy 1.0
- C-sweep invariante su C ∈ {0.01, 0.1, 1.0, 10.0}

**T02 — Confound N-01:**

- cosine(w_sign, w_mag) max = 0.078 su 24 layer
- mag_r2 = 0.86–0.98 da layer 2

**GSM8K:** baseline 0.0 (pre-FT), final pending

---

### Domande aperte prioritarie da investigare

**P0 — Potenzialmente invalidanti:**

1. Sign accuracy = 1.0 al layer 0: questo è plausibile o è un artefatto? Il layer 0 è il layer di embedding — prima di qualsiasi trasformazione. Un'accuracy perfetta qui suggerisce che l'informazione è nel token di input, non nelle rappresentazioni interne. Va discusso con riferimento a Tenney et al. (2019).
    
2. Il confound N-01 (primo operando diverso tra pair members in CAT-SIGN) è stato escluso tramite cosine similarity dei pesi — ma questa è una misura indiretta. Esiste un test diretto? La magnitudine del primo operando è correlata con il label sign?
    
3. 500 coppie vs 1000 standard BLiMP — giustificazione quantitativa?
    

**P1 — Limitazioni da documentare più precisamente:** 4. Multiple comparison correction: 24 layer × 2 properties = 48 test. È stata applicata? Se no, quanti risultati potrebbero essere falsi positivi?

5. 0-shot GSM8K vs 5-shot standard letteratura — stima della differenza sistematica?
    
6. L'emergenza di sign al layer 0 è compatibile con la "double staircase" di Quirke & Barez? I loro layer di computation coincidono con i nostri layer di decodificabilità?
    

**P2 — Qualificazioni linguistiche:** 7. Verificare che nessun claim nella tesi usi linguaggio causale per risultati correlativi.

8. Verificare che "emergence layer" sia definito operativamente nel testo (threshold 0.7 accuracy).

---

### Output atteso

Un documento strutturato con:

1. **Executive summary** — il claim principale della tesi è difendibile? Sì/No/Con qualificazioni
2. **Per ogni RQ:** forza delle evidenze, limitazioni critiche, claim da riformulare
3. **Tabella dei fix** — ordinata per priorità: P0 (invalida il risultato), P1 (limita il claim), P2 (qualificazione linguistica)
4. **Suggested additions** — quali analisi aggiuntive (anche semplici) rafforzerebbero significativamente la tesi