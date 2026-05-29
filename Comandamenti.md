## Scala di principi — Design Hierarchy v1

```
LIVELLO 0 — PROJECT INVARIANTS          (non principi, vincoli fissi)
LIVELLO 1 — OVERARCHING                 (sempre attivi, massima priorità)
LIVELLO 2 — STRUCTURAL                  (decisioni architetturali)
LIVELLO 3 — CONTRACT                    (interfacce tra moduli)
LIVELLO 4 — AI-NAVIGABILITY             (scrittura del codice)
LIVELLO 5 — ANTI-BLOAT                  (revisione continua)

Regola di conflitto: L(n) batte L(n+1) per default.
Override richiede argomento esplicito + approvazione human-in-the-loop.
```

---

### LIVELLO 0 — Project Invariants

_Non sono principi da applicare: sono i vincoli del progetto che framano tutto il resto._

|ID|Invariante|
|---|---|
|I-01|Progetto di ricerca accademico, singolo sviluppatore, lifetime finito (tesi)|
|I-02|AI-assisted development come workflow primario|
|I-03|Hardware: singola GPU RTX 5080, WSL2, conda env|
|I-04|Lingua: codebase inglese, tesi italiana|
|I-05|Zero requirement di scalabilità orizzontale o distribuzione|

---

### LIVELLO 1 — Overarching Principles

_Sempre attivi. Battono qualsiasi principio di livello inferiore._

|ID|Principio|Fonte|Regola|Vieta|Override|
|---|---|---|---|---|---|
|**O-01**|**Complexity is the enemy**|APOSD §1|Ogni decisione di design deve ridurre la complessità cognitiva percepita dal lettore|Qualsiasi struttura che richiede di tenere più di 2 file in testa simultaneamente per capire un'operazione|Solo se riduzione di complessità locale produce complessità globale maggiore — va dimostrato|
|**O-02**|**Clarity over cleverness**|Pragmatic Programmer §7|Il codice deve essere comprensibile al primo passaggio di lettura|Metaprogramming, dynamic dispatch, abstract factory chains, overuse di decoratori|Se il "clever" elimina una categoria intera di bug per costruzione (es. TypedDict strict)|
|**O-03**|**YAGNI**|Pragmatic Programmer §17|Non costruire ciò che non serve adesso|Astrazioni premature, interfacce per usi futuri ipotetici, framework interni|Se il costo di aggiungere dopo è dimostratamente O(n) invece di O(1)|

---

### LIVELLO 2 — Structural Principles

_Applicare nelle decisioni di architettura: quanti file, come raggrupparli, dove mettere cosa._

|ID|Principio|Fonte|Regola|Vieta|Override|
|---|---|---|---|---|---|
|**S-01**|**Deep modules**|APOSD §4|Preferire pochi file con interfacce semplici e implementazioni ricche a molti file con interfacce semplici e implementazioni banali|File con < 50 righe di logica reale (shallow modules)|Se il modulo è genuinamente indipendente e riusabile in altri progetti|
|**S-02**|**Vertical cohesion**|Art of Unix (Rule of Modularity)|Raggruppare per dominio verticale (tutto il probing in probing.py), non per tipo tecnico (tutti i dataclass in types.py, tutte le utility in utils.py)|Cartelle orizzontali tipo `helpers/`, `utils/`, `types/`|Se un componente serve a più domini (es. seeds.py → rimane in probing.py che è il suo dominio primario)|
|**S-03**|**Single Source of Truth**|Pragmatic Programmer §11|Ogni concetto ha esattamente un posto canonico nel codice|Costanti ridefinite in più file, logica di validazione duplicata|Mai — questo principio non ha override validi|
|**S-04**|**Information hiding**|APOSD §6|Un modulo espone solo ciò che il chiamante deve sapere; nasconde le decisioni di implementazione|Funzioni interne esposte nell'interfaccia pubblica, strutture dati interne trapelate nei return types|Se il testing richiede accesso a internals — usare il prefisso `_` e documentare|

---

### LIVELLO 3 — Contract Principles

_Applicare alla progettazione delle interfacce tra orchestratori e worker._

|ID|Principio|Fonte|Regola|Vieta|Overide|
|---|---|---|---|---|---|
|**C-01**|**Explicit contracts**|APOSD §10 + Pragmatic Programmer §23|Ogni handoff tra moduli usa TypedDict o dataclass con campi tipizzati esplicitamente|`dict[str, Any]` come tipo di ritorno o parametro tra moduli, tuple non nominate con più di 2 elementi|Se il contratto è usato in un solo posto e ha vita breve (< 1 sessione di sviluppo)|
|**C-02**|**Fail fast**|Pragmatic Programmer §24|Le precondizioni vengono verificate all'ingresso della funzione con `raise ValueError` — non a metà esecuzione|Codice difensivo sparso all'interno del corpo delle funzioni, silent failure, return `None` come segnale di errore|Mai per input che provengono dall'utente o da file — sempre validare|
|**C-03**|**Postel's Law — selectively**|Art of Unix (Robustness Principle)|Essere liberali in input (accettare float dove int è atteso, Path dove str è atteso), severi in output (restituire sempre il tipo dichiarato)|Applicare Postel ai contratti interni tra moduli — lì la severità bidirezionale è corretta|Solo alle interfacce verso l'esterno (CLI args, file input)|

---

### LIVELLO 4 — AI-Navigability Principles

_Applicare durante la scrittura del codice per massimizzare la comprensione da parte di agenti AI._

|ID|Principio|Fonte|Regola|Vieta|Override|
|---|---|---|---|---|---|
|**A-01**|**Self-contained context units**|Dhamani (AI-Optimised Codebases)|Ogni file deve essere comprensibile senza leggere altri file — le dipendenze esterne sono minimizzate e documentate in testa al file|File che richiedono di leggere 3+ altri file per capire cosa fanno|Se il dominio è genuinamente condiviso — allora va consolidato in un file solo (→ S-01)|
|**A-02**|**Comments explain WHY**|Dhamani + APOSD §13|I commenti spiegano l'intenzione e i trade-off, non la meccanica|Commenti che parafrasano il codice (`# increment i by 1`), commenti che descrivono cosa fa una funzione invece di perché esiste|Mai — un commento che spiega il "cosa" è un segnale che il codice va riscritto più chiaramente|
|**A-03**|**Predictable patterns**|SO Coding Guidelines (selettivo)|Struttura interna dei file monolitici è uniforme: sezioni nell'ordine fisso (Constants → TypedDict → Private helpers → Public API → `__main__`)|Funzioni helper sparse random nel file, ordine diverso per ogni file|Solo se il dominio impone un ordine logico diverso — va documentato con un commento in testa|
|**A-04**|**Boring code**|Dhamani + Pragmatic Programmer §8|Usare i costrutti più espliciti e prevedibili disponibili — il codice deve essere estendibile da un agente senza conoscere il contesto del progetto|Metaprogramming, `__getattr__` dinamici, decoratori personalizzati complessi, monkey patching|Se l'alternativa "boring" produce una quantità di duplicazione clinicamente insostenibile (> 200 righe identiche)|

---

### LIVELLO 5 — Anti-Bloat Principles

_Applicare continuamente, in particolare durante review e refactoring._

|ID|Principio|Fonte|Regola|Vieta|Override|
|---|---|---|---|---|---|
|**B-01**|**Ruthless pruning**|Beyond Frankenstein + Google Docs Best Practices|Ad ogni sessione di modifica: eliminare codice morto, TODO stantii, commenti obsoleti, funzioni non chiamate|Codice "tenuto per sicurezza", funzioni commentate fuori, parametri non usati|Mai — il codice morto è sempre peggio della sua assenza|
|**B-02**|**Rule of Three for abstraction**|Pragmatic Programmer §12|Astrarre un pattern solo quando compare la terza volta. Prima: duplica. Seconda: nota. Terza: astrai.|Astrazioni create per una singola istanza d'uso, interfacce "flessibili" per casi mai verificatisi|Se il costo di aggiungere la terza istanza senza astrazione è asimmetrico (es. bug di sicurezza)|
|**B-03**|**Lean documentation**|Google Doc Best Practices + Berkeley Doc Guide|Documentare: (1) perché esiste il modulo, (2) i contratti non ovvi, (3) le decisioni di design con alternative scartate. Eliminare tutto il resto.|README che elenca ogni funzione, docstring che parafrasano la signature, commenti in linea che spiegano sintassi Python|Se il modulo sarà usato da terzi (pubblicazione) — allora documentazione completa è giustificata|
|**B-04**|**No Frankenstein integration**|Beyond Frankenstein|Ogni aggiunta di funzionalità richiede una valutazione esplicita del suo impatto sulla complessità totale del modulo ricevente|Aggiungere funzioni a un modulo solo perché "ci sta bene", senza considerare se il modulo diventa conceptualmente incoerente|Mai — se non si riesce a descrivere il modulo in una frase dopo l'aggiunta, l'aggiunta va in un modulo diverso|

---

## Regole di conflitto esplicite

```
CONFLITTO TIPO 1 — O-03 (YAGNI) vs S-01 (Deep modules)
  Scenario: "creo un modulo ricco ora anche se userò solo una funzione oggi"
  Risoluzione: YAGNI vince. Crea solo ciò che serve. Il modulo cresce
  organicamente. L1 > L2.

CONFLITTO TIPO 2 — S-02 (Vertical cohesion) vs C-01 (Explicit contracts)
  Scenario: "il contratto TypedDict serve a 3 moduli verticali diversi"
  Risoluzione: il TypedDict va in config.py (source of truth) e importato.
  Non duplicato. S-03 (SSOT) media il conflitto.

CONFLITTO TIPO 3 — A-04 (Boring code) vs B-02 (Rule of Three)
  Scenario: "la terza istanza del pattern è complessa e l'astrazione
  richiederebbe un decoratore personalizzato"
  Risoluzione: A-04 vince. Duplica la terza volta piuttosto che
  introdurre metaprogramming. La semplicità batte l'eleganza. L4 > L5
  ma A-04 trae forza da O-02 (L1) → in realtà L1 > L5.

CONFLITTO TIPO 4 — O-01 (Reduce complexity) vs A-01 (Self-contained units)
  Scenario: "rendere il file self-contained richiede di copiare logica
  da un altro modulo"
  Risoluzione: non copiare — importare. Self-contained significa
  "comprensibile senza leggere altri file", non "zero imports".
  I-03 (SSOT) proibisce la copia. Nessun conflitto reale.
```

---

## Tabella di applicazione per fase di CoT

Questa tabella guida l'AI durante la chain of thought su una decisione architetturale specifica.

| Fase CoT                               | Domanda guida                                                           | Principi attivi (in ordine) |
| -------------------------------------- | ----------------------------------------------------------------------- | --------------------------- |
| **1. Cosa costruire**                  | Questo componente è necessario adesso?                                  | O-03 → O-01 → B-04          |
| **2. Dove metterlo**                   | Quale dominio verticale è il suo home naturale?                         | S-02 → S-01 → S-03          |
| **3. Come esporre l'interfaccia**      | Cosa deve sapere il chiamante? Cosa deve restare nascosto?              | S-04 → C-01 → C-02          |
| **4. Come scrivere l'implementazione** | Un agente AI potrebbe estendere questo codice senza context aggiuntivo? | A-01 → A-03 → A-04 → A-02   |
| **5. Quanto documentare**              | Cosa non è deducibile dal codice stesso?                                | B-03 → A-02 → B-01          |
| **6. Review finale**                   | Questo ha aumentato o ridotto la complessità totale?                    | O-01 → B-01 → B-02 → B-04   |
