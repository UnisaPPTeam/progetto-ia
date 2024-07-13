
# Rilevazione di Parafrasi in NLP

## Introduzione
Nel contesto in continua evoluzione dell’elaborazione del linguaggio naturale (NLP), la rilevazione di parafrasi rappresenta una sfida significativa e di grande rilevanza. Affrontare questa sfida è cruciale per migliorare la comprensione semantica e la rilevazione di similitudini tra testi. Nel corso di questo progetto, ci siamo dedicati allo sviluppo di modelli avanzati per la rilevazione di parafrasi, sfruttando tre approcci distinti: rete Transformer, rete LSTM bidirezionale (BiLSTM), e rete siamese. A tale scopo, abbiamo utilizzato il dataset ”Quora Question Pairs”, composto da 400.000 coppie di domande provenienti dalla piattaforma Quora [2]. Questo documento presenta un’analisi di ciascun approccio, esplorando la loro efficacia nella rilevazione di parafrasi.

## Descrizione del problema
La rilevazione di parafrasi rappresenta una sfida cruciale nell’ambito dell’elaborazione del linguaggio naturale (NLP), poiché richiede la capacità di discernere sottili variazioni semantiche e strutturali tra le espressioni linguistiche. Il problema centrale consiste nell’identificare se due frasi o testi distinti condividono un significato simile, nonostante possibili differenze nella formulazione. Questa complessità deriva dalla natura sfaccettata del linguaggio, che può presentare molteplici modi per esprimere concetti equivalenti.

L’importanza della rilevazione di parafrasi si riflette in molteplici ambiti, inclusi motori di ricerca, assistenti virtuali e comprensione automatica del testo. Affrontare questo problema richiede l’adozione di approcci avanzati che vadano oltre una mera confrontazione di parole, ma che siano in grado di cogliere le sfumature semantiche e sintattiche intrinseche alle relazioni di parafrasi. Nel corso di questo progetto, ci concentriamo su tre approcci principali: l’utilizzo di reti Transformer, reti LSTM bidirezionali (BiLSTM), e reti siamesi, ciascuno dei quali si propone di affrontare il problema della rilevazione di parafrasi in modo diverso.

## Scelte progettuali
### Preprocessing e scelta degli Embedding
È stata eseguita la lemmatizzazione dell’intero dataset mediante l’utilizzo della libreria Spacy, contribuendo a una migliore comprensione e generalizzazione del modello.

In aggiunta, è stata adottata una serie di accorgimenti per uniformare ulteriormente le frasi nel preprocessing del testo:
- Tutti i token sono stati convertiti in minuscolo, garantendo una consistenza nelle rappresentazioni delle parole.
- I segni di punteggiatura sono stati eliminati, semplificando la struttura delle frasi e focalizzando l’attenzione sul contenuto semantico.
- I token composti da singole lettere, come ’s’ o ’a’, sono stati rimossi, contribuendo a eliminare informazioni non significative.
- Sono stati eliminati tutti i token che contenevano numeri, migliorando la coerenza e la generalizzazione delle rappresentazioni linguistiche.

Queste operazioni di normalizzazione e pulizia del testo mirano a ottimizzare la qualità e la coerenza dei dati di input, preparando così il terreno per una fase di addestramento più efficace e una migliore capacità del modello di cogliere le relazioni semantiche tra le frasi.

Per gli embedding sono stati utilizzati quelli pre-addestrati dagli autori di GloVe [3] (Global Vectors for Word Representation), un algoritmo che permette la creazione di embedding a partire da corpus.

### Strategie di Tokenizzazione
Per la tokenizzazione è stato sfruttato il tokenizer di Keras [1], optando per una soluzione SOTA usata in diversi paper.

## Soluzione proposta
### Approccio Transformers
L’architettura basata su blocco Transformer è stata implementata per sfruttare le capacità di autoattenzione e modellazione delle relazioni a lungo termine. Il modello è rappresentato nella figura 1.

#### Architettura rete Transformer
La struttura di una rete transformer si compone come segue:
- **Embedding condiviso:** Entrambe le sequenze di input vengono embeddate utilizzando uno strato di embedding condiviso. Questo strato cattura le rappresentazioni semantiche delle parole grazie agli embedding pre-addestrati.
- **Encoder Block:** L’input viene poi mandato ad un Encoder Block. Questo blocco implementa il meccanismo di autoattenzione, consentendo al modello di catturare le dipendenze a lungo termine e di identificare relazioni semantiche complesse nelle sequenze di testo.
- **Concatenazione e Normalizzazione:** Le rappresentazioni delle sequenze di input ottenute dal blocco Transformer Encoder sono concatenate lungo l’asse temporale. Successivamente, viene applicato uno strato di Batch Normalization per normalizzare le feature e migliorare la stabilità del modello.
- **Flattening e MLP:** Dopo la normalizzazione, le rappresentazioni concatenate passano attraverso uno strato di flattening e quindi attraverso un Multi-Layer Perceptron (MLP). L’MLP è composto da uno strato di attivazione tangente iperbolica (tanh) e uno strato di Dropout per ridurre il rischio di overfitting.
- **Strato di Output:** L’output dello strato MLP è quindi fornito a uno strato di output con attivazione softmax. Questo strato restituisce la probabilità di appartenenza a ciascuna delle due classi ("parafrasi" e "non parafrasi").
- **Funzione di Perdita e Metriche di Valutazione:** L’ottimizzatore RMSprop e la funzione di perdita categorical-crossentropy sono utilizzati per l’addestramento del modello. Le prestazioni del modello sono state valutate utilizzando tutte le principali metriche, tra cui accuracy, precision e f1 score.

### Approccio BiLSTM
L’architettura basata su Bidirectional Long Short-Term Memory (BiLSTM) è stata concepita per sfruttare la capacità delle reti ricorrenti di catturare informazioni contestuali in entrambe le direzioni temporali. Il modello è rappresentato nella figura 2.

#### Architettura rete BiLSTM
La struttura di una rete BiLSTM si compone come segue:
- **Embedding condiviso:** Utilizziamo un embedding condiviso tra le due sequenze di input. Questo assicura che le parole siano rappresentate in maniera coerente in entrambi i percorsi della rete, facilitando la comparazione.
- **Bidirectional LSTM Encoder:** Entrambe le sequenze vengono elaborate da uno strato Bidirectional LSTM. Ciò consente al modello di catturare informazioni sia in avanti che all’indietro, migliorando la comprensione del contesto delle parole.
- **Gated Relevance Network (GRN):** Introduciamo un Gated Relevance Network per valutare l’importanza delle informazioni tra le due sequenze. Questo meccanismo di attenzione permette al modello di concentrarsi su parti rilevanti delle sequenze, agevolando la rilevazione di parafrasi.
- **Max Pooling e MLP:** Dopo l’applicazione di GRN, effettuiamo il max pooling 2D non sovrapposto per ridurre la dimensionalità dell’output. Successivamente, attraverso uno strato di Multi-Layer Perceptron (MLP), estraiamo rappresentazioni più astratte e complesse delle sequenze.
- **Strato di Output:** Infine, attraverso uno strato di output softmax con due nodi, il modello genera le probabilità di appartenenza alle classi di parafrasi o non parafrasi.
- **Ottimizzazione:** Utilizziamo l’ottimizzatore RMSprop e la funzione di perdita binaria di entropia incrociata.

### Approccio Siamese
L’architettura Siamese è stata scelta per la sua capacità di confrontare direttamente due sequenze, rendendola idonea per la rilevazione di parafrasi attraverso una misurazione accurata della similarità tra le frasi. Il modello è rappresentato nella figura 3.

#### Architettura rete Siamese
La struttura di una rete Siamese si compone come segue:
- **Embedding condiviso:** Anche in questo caso utilizziamo un embedding condiviso per entrambe le sequenze, garantendo una rappresentazione coerente delle parole in entrambi i percorsi del modello.
- **LSTM Condivisa:** Le sequenze elaborate dall’embedding condiviso vengono inviate attraverso uno strato LSTM condiviso. Questo strato è progettato per catturare relazioni semantiche complesse all’interno di ciascuna sequenza.
- **Confronto Diretto:** Utilizzando uno strato di sottrazione (Subtract), viene calcolata la differenza elemento per elemento tra i vettori risultanti dalle LSTM condivise. Successivamente, uno strato di moltiplicazione (Multiply) calcola il quadrato di questa differenza, enfatizzando ulteriormente le discrepanze tra le sequenze.
- **Calcolo della Distanza Coseno:** La distanza coseno tra i vettori risultanti dalle LSTM condivise viene calcolata utilizzando una funzione Lambda. Questa misura fornisce una rappresentazione della similarità tra le sequenze, incorporando informazioni semantiche più astratte rispetto al confronto diretto.
- **Concatenazione delle Caratteristiche:** Le caratteristiche estratte dal confronto diretto e dalla distanza coseno vengono concatenate lungo l’asse delle caratteristiche. Questo processo aggrega le informazioni da entrambi gli approcci, creando una rappresentazione complessiva delle relazioni tra le sequenze.
- **Multi-Layer Perceptron (MLP):** Le caratteristiche concatenate sono fornite a un MLP composto da uno strato di attivazione ReLU, uno strato di Dropout per mitigare l’overfitting, e uno strato di output con attivazione sigmoide. L’output di questo strato indica la probabilità che le due sequenze siano parafrasi o meno.
- **Funzione di Perdita e Metriche di Valutazione:** L’ottimizzatore Adam è stato scelto per guidare l’addestramento del modello. La funzione di perdita utilizzata è la binary-crossentropy, adatta per problemi di classificazione binaria. Metriche come l’accuracy, l’Area Under the Receiver Operating Characteristic Curve (AUROC), precision, recall e F1-score sono monitorate per valutare le prestazioni del modello.

## Risultati e considerazioni
Abbiamo esaminato tre diverse architetture neurali progettate per affrontare il problema di rilevamento di parafrasi in contesti di elaborazione del linguaggio naturale (NLP). Dopo una valutazione comparativa, emergono le seguenti considerazioni sulla performance relativa:

- **Approcio Siamese:**
  - **Punti chiave:**
    - Architettura che condivide embedding e LSTM comuni per entrambe le sequenze.
    - Incorpora uno strato di confronto diretto e distanza coseno per misurare la similarità.
    - Utilizzo efficace della distanza coseno per catturare relazioni semantiche astratte.
  - **Considerazioni:**
    - Eccelle nel valutare direttamente la similarità tra sequenze.
    - Adatto a problemi di parafrasi che richiedono una valutazione accurata delle relazioni semantiche.

- **Approccio BiLSTM con Gated Relevance Network (GRN):**
  - **Punti chiave:**
    - Utilizza uno strato Bidirectional LSTM per catturare relazioni contestuali.
    - Introduce uno strato Gated Relevance Network (GRN) per modellare meccanismi di attenzione.
    - Applicazione di Max Pooling 2D, Flatten e Multi-Layer Perceptron per apprendere relazioni.
  - **Considerazioni:**
    - Buona capacità di catturare dipendenze contestuali complesse grazie alla bidirezionalità di LSTM.
    - GRN offre un meccanismo di attenzione, ma può essere meno efficace rispetto alla distanza cosinica di Siamese.

- **Blocco Transformer:**
  - **Punti chiave:**
    - Implementa uno strato Transformer Encoder Block per sfruttare l’autoattenzione.
    - Concatenazione delle rappresentazioni seguita da Batch Normalization e Multi-Layer Perceptron per l’apprendimento.
    - Utilizza la funzione di perdita categorical-crossentropy e l’ottimizzatore RMSprop.
  - **Considerazioni:**
    - Eccellente nell’apprendere relazioni a lungo termine grazie all’autoattenzione di Transformer.
    - Batch Normalization contribuisce alla stabilità del modello, ma potrebbe essere meno adatto per problemi di parafrasi specifici.

## Conclusioni
La migliore performance è stata osservata con l’architettura Siamese, grazie alla sua abilità nel catturare relazioni semantiche astratte. La BiLSTM con GRN fornisce una buona alternativa, in particolare la bidirezionalità di LSTM risulta essere un fattore determinante. Il Blocco Transformer si colloca come terza opzione, mostrando forza nell’apprendere relazioni a lungo termine, ma potrebbe essere meno specializzato per problemi di parafrasi specifici.

In conclusione, la scelta tra queste architetture dovrebbe essere guidata dalla natura specifica del problema di parafrasi e dalle esigenze del contesto applicativo, con un occhio attento alla precisione della similarità semantica.



