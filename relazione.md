# Crowd Density Estimation in Ambienti Museali con Singola Telecamera

## Corso: Advanced Digital Image Processing — Università di Siena

---

## 1. Introduzione

Il monitoraggio del flusso di visitatori all'interno di spazi espositivi rappresenta una necessità crescente per la gestione della sicurezza, la tutela delle opere e l'ottimizzazione dell'esperienza del pubblico. Il presente progetto affronta il problema della stima della densità e della distribuzione delle persone all'interno di una sala museale, utilizzando una singola telecamera fissa, senza ricorrere a sistemi stereoscopici o multi-camera.

### 1.1 Obiettivi

Gli obiettivi del sistema sono tre:

1. **Conteggio delle persone**: stimare il numero di visitatori presenti nella scena in ogni istante, con un intervallo di confidenza associato.
2. **Distribuzione spaziale**: comprendere come i visitatori si distribuiscono nello spazio della sala, producendo una mappa di densità.
3. **Rilevamento assembramenti**: identificare automaticamente raggruppamenti di persone che superano una soglia configurabile e generare allarmi.

### 1.2 Contesto applicativo

Il sistema è stato sviluppato e testato sulla sala delle statue del Museo dell'Opera del Duomo di Siena. L'ambiente presenta caratteristiche particolari che rendono il problema significativamente più complesso rispetto a scenari di videosorveglianza tradizionali:

- **Statue a figura umana**: la sala contiene numerose statue in marmo bianco che riproducono figure umane a grandezza naturale. Queste vengono sistematicamente rilevate come persone dai detector generici, generando falsi positivi.
- **Illuminazione variabile**: l'illuminazione artificiale crea zone di forte contrasto, con aree ben illuminate e zone scure che penalizzano il rilevamento.
- **Camera grandangolare fissa**: la telecamera è posizionata in alto con lente grandangolare, introducendo distorsione prospettica e rendendo le persone relativamente piccole nel frame.
- **Occlusioni ambientali**: colonne, piedistalli e le stesse statue creano frequenti occlusioni parziali o totali dei visitatori.

### 1.3 Vincoli di progetto

- Elaborazione basata su un singolo flusso video monoculare
- Utilizzo di modelli pre-addestrati (nessun training specifico)
- Esecuzione completamente locale (nessuna dipendenza da API cloud)
- Dati video confidenziali: i video del museo non possono essere condivisi o diffusi

---

## 2. Architettura del Sistema

Il sistema è organizzato come una pipeline modulare in cui ogni componente ha una responsabilità ben definita. L'architettura è stata progettata per supportare due modalità operative: una completa con calibrazione omografica e una semplificata (detection-only) per operare senza calibrazione.

### 2.1 Moduli della pipeline

La pipeline è composta dai seguenti moduli:

| Modulo | File | Responsabilità |
|--------|------|---------------|
| HeadDetector | `detection.py` | Rilevamento persone tramite YOLOv8, estrazione regione testa |
| BackgroundFilter | `background.py` | Filtraggio detection statiche (statue) via background subtraction |
| HomographyModule | `homography.py` | Trasformazione pixel → coordinate metriche sul piano del suolo |
| StateMachine | `tracking.py` | Tracking temporale con macchina a stati (visibile → occluso → scomparso) |
| CountingModule | `counting.py` | Conteggio con intervallo di confidenza |
| DensityMapGenerator | `density.py` | Generazione mappa di densità 2D con kernel gaussiano |
| CrowdClusterer | `clustering.py` | Rilevamento assembramenti con DBSCAN e generazione allarmi |
| Visualizer | `visualization.py` | Overlay grafico su frame video |

### 2.2 Flusso di elaborazione

Il flusso differisce a seconda della disponibilità della calibrazione omografica.

**Modalità completa (con omografia):**

```
Frame → HeadDetector → BackgroundFilter → HomographyModule → StateMachine
      → CountingModule → DensityMapGenerator → CrowdClusterer → Output
```

In questa modalità il tracking temporale mantiene lo stato delle persone tra frame successivi, gestendo le occlusioni temporanee e il decadimento della confidenza.

**Modalità detection-only (senza omografia — MVP):**

```
Frame → HeadDetector → BackgroundFilter → Conteggio diretto
      → Clustering su detection correnti → Output
```

In questa modalità ogni frame è elaborato indipendentemente. Non c'è memoria tra frame: il conteggio riflette esattamente ciò che il detector vede nell'istante corrente. Questa scelta è motivata nella Sezione 4.

### 2.3 Configurazione e parametri

Il sistema è configurabile tramite argomenti da linea di comando:

```bash
python main.py --video video.mp4 [opzioni]
```

Parametri principali:

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--calibration` | None | File JSON con punti di calibrazione omografica |
| `--bg-filter` | False | Abilita background subtraction (warmup dai primi N frame) |
| `--bg-image` | None | Immagine di sfondo pre-calcolata (salta warmup) |
| `--bg-frames` | 60 | Frame per costruire il modello di sfondo |
| `--cluster-radius` | 2.0 | Raggio clustering in metri |
| `--density-threshold` | 1.0 | Soglia densità per assembramento (persone/m²) |
| `--critical-threshold` | 2.0 | Soglia critica per allarme |
| `--output` | None | Percorso video di output con overlay |

---

## 3. Rilevamento delle Persone

### 3.1 Scelta del modello

Il rilevamento si basa su YOLOv8n (nano), la variante più leggera della famiglia YOLOv8, pre-addestrata sul dataset COCO. La scelta è motivata da:

- **Nessun training richiesto**: il modello è utilizzato direttamente senza fine-tuning
- **Velocità**: il modello nano (~6 MB) permette l'elaborazione a ~11 fps su CPU
- **Disponibilità**: il modello viene scaricato automaticamente dalla libreria ultralytics

YOLOv8n rileva la classe "person" (classe 0 di COCO) con bounding box e confidenza associata.

### 3.2 Estrazione della regione testa

Poiché YOLOv8 non ha una classe "head" specifica, il detector adotta una strategia di approssimazione geometrica: dalla bounding box della persona intera, viene estratto il **22% superiore** come regione testa. Questa proporzione è basata sull'antropometria media (la testa occupa circa 1/5 dell'altezza corporea).

La regione testa estratta viene poi sottoposta a:

1. **Filtro area minima** (16×16 pixel): scarta le teste troppo piccole per essere affidabili
2. **Calcolo confidenza piecewise**: visible_ratio < 0.30 → scartata; 0.30–0.90 → proporzionale; ≥ 0.90 → confidenza 1.0
3. **Non-Maximum Suppression** (IoU threshold 0.5): elimina detection sovrapposte

### 3.3 Limiti del detector

Il detector presenta limiti intrinseci legati all'uso di un modello generico:

- **Undercount sistematico**: YOLO è addestrato su immagini con persone ben visibili e a dimensione ragionevole. In scene dall'alto con persone piccole e parzialmente occluse, molte non vengono rilevate.
- **Falsi positivi su statue**: le statue a figura umana vengono classificate come "person" con alta confidenza, essendo indistinguibili a livello di features visuali.
- **Detection intermittente**: la stessa persona può essere rilevata in un frame e non in quello successivo, a causa di variazioni di posa, illuminazione o occlusione parziale.

---

## 4. Background Subtraction per Ambienti Museali

### 4.1 Il problema delle statue

La sala del museo contiene circa 15 statue in marmo bianco che riproducono figure umane in pose naturali. YOLOv8 le rileva sistematicamente come persone, con confidenze elevate (spesso > 0.80). Questo genera un conteggio base di 6-8 "persone" anche quando la sala è completamente vuota di visitatori.

### 4.2 Soluzione: modello di sfondo basato sulla mediana

Il principio è semplice: tutto ciò che non si muove fa parte dello sfondo. Le statue, i piedistalli e le opere d'arte sono statici; i visitatori si muovono.

Il BackgroundFilter opera in due fasi:

**Fase di warmup** (primi N frame):
- Ogni frame viene convertito in scala di grigi e accumulato in un buffer
- Al raggiungimento di N frame, si calcola la **mediana pixel-per-pixel**
- La mediana è robusta a oggetti transitori (una persona che passa durante il warmup non contamina il modello)
- Il modello risultante viene salvato come `background_model.png` per riutilizzo futuro

**Fase di filtraggio** (frame successivi):
- Per ogni detection di YOLO, si estrae la regione della bounding box (espansa al corpo intero)
- Si calcola la differenza assoluta tra la regione nel frame corrente e la stessa regione nel modello di sfondo
- Si conta la percentuale di pixel con differenza superiore a una soglia (default: 30 su scala 0-255)
- Se la percentuale di pixel "cambiati" è inferiore al 15% → la detection è **statica** (statua) e viene scartata
- Se è superiore → la detection è **dinamica** (persona vera) e viene mantenuta

### 4.3 Riutilizzo del modello

Poiché la telecamera è fissa e la scena non cambia strutturalmente, il modello di sfondo calcolato una volta può essere riutilizzato per tutti i video successivi. Questo elimina la fase di warmup e permette il filtraggio dal primo frame:

```bash
# Prima esecuzione: calcola e salva il modello
python main.py --video video1.mp4 --bg-filter

# Esecuzioni successive: riusa il modello salvato
python main.py --video video2.mp4 --bg-image background_model.png
```

---

## 5. Modalità Detection-Only

### 5.1 Motivazione

Il design originale del sistema prevede un tracking temporale completo, con una macchina a stati che gestisce le transizioni visibile → parzialmente occluso → occluso → scomparso. Questo richiede però un'associazione affidabile tra le detection nei frame successivi, basata sulla distanza euclidea tra le posizioni stimate nel piano del suolo.

Senza calibrazione omografica, le posizioni sono in coordinate pixel. Il matching del tracker, progettato per distanze in metri, non funziona correttamente in pixel: la stessa persona che si muove viene vista come una nuova entità ogni pochi frame, mentre le vecchie entità restano in stato "occluso" e si accumulano indefinitamente. Il risultato è un conteggio che cresce monotonicamente nel tempo fino a raggiungere decine di "persone" quando nella scena ce n'è una sola.

### 5.2 Soluzione: elaborazione frame-indipendente

In assenza di omografia, la pipeline opera in modalità detection-only:

- **Conteggio**: basato direttamente sul numero di head detection nel frame corrente. Se YOLO rileva 3 persone, il conteggio è 3. Se nel frame successivo ne rileva 2, il conteggio è 2.
- **Clustering**: DBSCAN opera sulle coordinate pixel delle detection del frame corrente, con raggio scalato in base alla risoluzione del frame. Non c'è persistenza tra frame.
- **Nessun accumulo**: ogni frame parte da zero. Non ci sono persone "ricordate" dai frame precedenti.

Questa modalità sacrifica la capacità di gestire le occlusioni temporanee (una persona che passa dietro una colonna viene "persa" fino a quando riappare) ma elimina completamente il problema dell'accumulo di falsi positivi.

### 5.3 Clustering in coordinate pixel

In assenza di coordinate metriche, DBSCAN opera in pixel con parametri scalati. Per un frame 1920×1080 che copre una sala di circa 12 metri di larghezza, la scala approssimativa è ~150 pixel/metro. Il raggio di clustering di 2 metri diventa ~300 pixel.

Le soglie di densità sono adattate di conseguenza per garantire che i cluster con almeno 3 persone vicine vengano rilevati.

---

## 6. Benchmark sul Mall Dataset

### 6.1 Dataset

Il Mall Dataset è un benchmark pubblico per il crowd counting, composto da:

- 2000 frame estratti da una webcam in un centro commerciale
- Risoluzione 640×480, frame rate < 2 Hz
- Oltre 60.000 pedestri annotati (posizione della testa per ogni persona in ogni frame)
- Media di ~31 persone per frame

Il dataset è stato scelto perché fornisce un ground truth quantitativo che permette di misurare la precisione del sistema in modo oggettivo.

### 6.2 Metodologia

Per ogni frame del dataset:
1. L'immagine viene caricata e passata alla pipeline in modalità detection-only
2. Il conteggio predetto è `len(head_detections)` — il numero di teste rilevate da YOLO
3. Il conteggio viene confrontato con il ground truth annotato

Le metriche calcolate sono:
- **MAE** (Mean Absolute Error): errore medio assoluto tra conteggio predetto e reale
- **RMSE** (Root Mean Square Error): radice dell'errore quadratico medio
- **MAPE** (Mean Absolute Percentage Error): errore percentuale medio

### 6.3 Risultati

| Metrica | Valore |
|---------|--------|
| Frame elaborati | 2000 |
| FPS medio | 11.11 |
| MAE | 20.61 |
| RMSE | 21.25 |
| MAPE | 66.32% |
| Media GT | 31.16 persone |
| Media predetto | 10.55 persone |
| Errore minimo | -37 |
| Errore massimo | -7 |

**[Inserire qui: grafico andamento temporale GT vs predetto]**

**[Inserire qui: scatter plot GT vs predetto]**

**[Inserire qui: istogramma distribuzione errori]**

### 6.4 Analisi dei risultati

Il sistema rileva in media circa un terzo delle persone presenti (10.55 su 31.16). L'errore è **sistematicamente in difetto** (undercount): il valore massimo dell'errore è -7, il che significa che il sistema non ha mai sovrastimato il conteggio reale.

Le cause dell'undercount sono:

1. **Vista dall'alto**: il Mall Dataset è ripreso da una webcam sopraelevata. YOLO è addestrato prevalentemente su immagini con vista frontale o leggermente inclinata.
2. **Persone piccole**: nella scena affollata, molte persone occupano pochi pixel e la regione testa estratta è sotto la soglia minima di 16×16 pixel.
3. **Occlusione interpersonale**: in una scena con 30+ persone, molte sono parzialmente coperte da altre e non vengono rilevate.
4. **Nessun fine-tuning**: il modello è generico (COCO) e non è stato adattato a scene di crowd counting.

### 6.5 Confronto con lo stato dell'arte

I metodi dedicati al crowd counting raggiungono MAE significativamente inferiori sul Mall Dataset. Ad esempio, i modelli basati su density regression (come CSRNet) stimano una mappa di densità continua tramite reti convoluzionali addestrate specificamente su dataset di crowd counting, senza passare per la detection individuale delle persone. Questi approcci raggiungono MAE nell'ordine di 2-3 persone su questo dataset.

La differenza è strutturale: il nostro approccio è basato su detection (localizza ogni persona individualmente), mentre lo stato dell'arte usa regressione (stima direttamente il conteggio senza localizzare). Il vantaggio del nostro approccio è che fornisce la posizione di ogni persona rilevata, necessaria per il rilevamento assembramenti. Lo svantaggio è la sensibilità alla qualità del detector.

---

## 7. Test sui Video del Museo

### 7.1 Dataset

Il dataset del museo consiste in registrazioni video dalla sala delle statue del Museo dell'Opera del Duomo di Siena:

- Durata totale: 6 ore e 30 minuti
- Suddivisione: blocchi da 30 minuti
- Risoluzione: 1920×1080 a 25 fps
- Camera: grandangolare fissa, posizionata in alto

I video sono soggetti a restrizioni di riservatezza e non possono essere diffusi.

### 7.2 Risultati qualitativi

**Senza background filter:**
Il detector rileva costantemente 6-8 entità corrispondenti alle statue della sala. L'aggiunta di visitatori reali porta il conteggio a salire ulteriormente. Non è possibile distinguere automaticamente statue da persone.

**Con background filter:**
Le statue vengono correttamente filtrate nella maggior parte dei frame. Il conteggio riflette il numero effettivo di visitatori presenti. Quando un visitatore entra nella scena, il conteggio sale; quando esce, torna a zero.

**[Inserire qui: screenshot frame con overlay — sala vuota, conteggio 0]**

**[Inserire qui: screenshot frame con overlay — 1 persona rilevata]**

**[Inserire qui: screenshot frame con overlay — gruppo di visitatori, eventuale allarme assembramento]**

### 7.3 Limiti osservati

- **Detection intermittente**: la stessa persona può essere rilevata in un frame e non in quello successivo, causando un conteggio che fluttua. Questo è dovuto alla combinazione di illuminazione scarsa, vista grandangolare e modello non specializzato.
- **Zone morte**: alcune aree della sala, particolarmente scure o ai margini del grandangolo, hanno un tasso di detection più basso.
- **Assenza di tracking**: in modalità detection-only, se una persona passa dietro una colonna, viene "persa" fino alla riapparizione. Non c'è continuità del conteggio durante le occlusioni.

---

## 8. Scelte Progettuali e Compromessi

### 8.1 Perché non il tracking temporale (senza omografia)

Il tracking temporale con macchina a stati è implementato nel codice e funziona correttamente quando l'omografia è calibrata (le posizioni sono in metri). Senza omografia, il matching basato su distanza euclidea in pixel non è affidabile: la stessa persona in movimento genera continuamente nuove entità tracciate, e le vecchie entità persistono in stato occluso, producendo un conteggio che cresce monotonicamente. La scelta di disabilitare il tracking in assenza di omografia è un compromesso pragmatico che privilegia la correttezza del conteggio.

### 8.2 Perché non la pose estimation

Il design del sistema prevede l'integrazione di pose estimation (skeleton/body pose) per:
- Stimare la posizione a terra dai piedi quando il corpo è completamente visibile
- Usare la silhouette per classificare il tipo di occlusione
- Migliorare la precisione della localizzazione

Nell'MVP questa funzionalità non è implementata per due motivi:
1. Senza omografia, la posizione dei piedi in pixel non fornisce informazioni significativamente migliori del centro della bounding box
2. Le statue del museo hanno pose umane realistiche e ingannerebbero anche un pose estimator, richiedendo comunque il background filter

La pose estimation resta come sviluppo futuro per quando sarà disponibile la calibrazione omografica.

### 8.3 Perché YOLOv8n e non un modello più grande

YOLOv8n è il modello più piccolo della famiglia (6 MB). Modelli più grandi (YOLOv8s, m, l, x) offrono maggiore precisione a costo di velocità inferiore. La scelta del nano è motivata dalla volontà di mantenere un throughput ragionevole su CPU (~11 fps). Per migliorare la precisione, il passaggio a YOLOv8s (22 MB) rappresenterebbe il primo upgrade naturale.

---

## 9. Tecnologie Utilizzate

| Componente | Tecnologia | Versione |
|-----------|-----------|---------|
| Linguaggio | Python | 3.10+ |
| Person Detection | YOLOv8n (ultralytics) | 8.x |
| Computer Vision | OpenCV | 4.x |
| Clustering | scikit-learn (DBSCAN) | 1.x |
| Calcolo numerico | NumPy | 1.x |
| Visualizzazione | Matplotlib | 3.x |
| Ground truth | SciPy (loadmat) | 1.x |

Tutto il codice è eseguibile in locale senza dipendenze da servizi cloud.

---

## 10. Sviluppi Futuri

1. **Calibrazione omografica**: stimare l'omografia dai riferimenti geometrici visibili nel frame (pattern del pavimento, distanze note) per abilitare il tracking temporale e le metriche in metri.

2. **Pose estimation**: integrare YOLOv8-pose per ottenere i keypoint corporei, migliorando la localizzazione e la classificazione delle occlusioni.

3. **Modello specializzato**: fine-tuning di YOLO su un dataset di persone viste dall'alto, o adozione di un modello dedicato al crowd counting (density regression).

4. **Tracking con Re-ID**: aggiungere un modulo di re-identification per mantenere l'identità delle persone attraverso le occlusioni, anche senza omografia.

5. **Analisi temporale**: generare statistiche aggregate nel tempo (flusso orario, tempo medio di permanenza, zone più frequentate) per supportare la gestione del museo.

---

## 11. Conclusioni

Il sistema dimostra la fattibilità del monitoraggio dei visitatori in un ambiente museale utilizzando una singola telecamera fissa e modelli pre-addestrati. La sfida principale — la presenza di statue a forma umana che ingannano il detector — è stata affrontata con successo attraverso un filtro di background subtraction basato sulla mediana temporale.

Il benchmark sul Mall Dataset quantifica i limiti del detector generico in scene affollate (MAE ≈ 20.6), mentre i test qualitativi sui video reali del museo confermano che il sistema è adeguato al caso d'uso specifico, dove il numero di visitatori simultanei è tipicamente ridotto (1-10 persone).

L'architettura modulare del sistema permette di migliorare ogni componente indipendentemente: un detector migliore, l'aggiunta dell'omografia o della pose estimation possono essere integrati senza riscrivere l'intero sistema. Il design completo con tracking temporale, macchina a stati e gestione delle occlusioni è già implementato nel codice e attivabile non appena sarà disponibile una calibrazione omografica affidabile.
