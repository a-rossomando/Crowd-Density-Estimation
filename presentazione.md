# Presentazione — Crowd Density Estimation in Ambienti Museali

Guida slide per slide. Ogni sezione = una slide (o al massimo due).
I testi sono volutamente brevi: nella presentazione orale espandi a voce.

---

## SLIDE 1 — Titolo

**Crowd Density Estimation in Ambienti Museali con Singola Telecamera**

Advanced Digital Image Processing — Università di Siena

[Il tuo nome]

---

## SLIDE 2 — Il problema

**Obiettivo:** monitorare i visitatori di un museo con una singola telecamera fissa.

Tre compiti:
- Contare quante persone ci sono nella scena
- Capire come sono distribuite nello spazio
- Rilevare assembramenti e generare allarmi

**La sfida:** la sala è piena di statue a forma umana che ingannano i detector.

[Immagine: frame del museo con statue e visitatori]

---

## SLIDE 3 — L'ambiente: Museo dell'Opera del Duomo di Siena

Caratteristiche della scena:
- 10 statue in marmo bianco a grandezza naturale
- Illuminazione artificiale con forti contrasti
- Camera grandangolare fissa in alto
- Occlusioni da colonne, piedistalli e statue

[Immagine: frame panoramico della sala vuota]

---

## SLIDE 4 — Pipeline del sistema

Diagramma a blocchi:

```
Frame video
    ↓
YOLOv8n (person detection)
    ↓
Background Filter (rimuove statue)
    ↓
Conteggio persone + Clustering (DBSCAN)
    ↓
Overlay video + Allarmi assembramento
```

Tutto eseguito in locale, nessuna API cloud, modelli pre-addestrati.

---

## SLIDE 5 — Detection: da persona a testa

Come funziona:
- YOLOv8n rileva bounding box "person"
- Si estrae il 22% superiore della bbox come regione testa
- Filtro area minima (16×16 px) e Non-Maximum Suppression

Perché la testa? È l'ultima parte a essere occlusa in una scena affollata.

[Immagine: frame con bounding box verdi sulle teste rilevate]

---

## SLIDE 6 — Il problema delle statue

YOLO rileva sporadicamente le statue come persone (in alcuni frame sì, in altri no).

Con il **tracking temporale attivo**, queste detection intermittenti si accumulavano:
- Ogni statua rilevata diventava una TrackedPerson permanente
- Il conteggio cresceva fino a **59 "persone"** con 1 sola persona reale nella sala
- Allarmi assembramento continui su raggruppamenti inesistenti

Evidenza dai log: "cluster con 6-8 persone, area 31415 m²" — sala vuota.

---

## SLIDE 7 — Doppia soluzione

**1. Background Subtraction** — filtra le detection statiche:
- Mediana dei primi 60 frame → modello di sfondo
- Per ogni detection: confronto col modello
- Regione non cambiata → statua → scartata
- Il modello si calcola una volta e si riusa per tutti i video

**2. Modalità detection-only** — elimina l'accumulo:
- Ogni frame elaborato indipendentemente, senza memoria
- Il conteggio riflette solo ciò che YOLO vede nell'istante corrente
- Nessuna persona "fantasma" che persiste tra frame

[Immagine: background_model.png affiancato a un frame con persona rilevata]

---

## SLIDE 8 — Due modalità operative

| | Con omografia | Senza omografia (MVP) |
|---|---|---|
| Coordinate | Metri reali | Pixel |
| Tracking | Macchina a stati temporale | Frame indipendenti |
| Occlusioni | Gestite (persona "ricordata") | Non gestite |
| Clustering | Distanze in metri | Distanze in pixel scalate |

**MVP usato per i test:** senza calibrazione, ogni frame è elaborato indipendentemente. Nessun accumulo di falsi positivi.

---

## SLIDE 9 — Perché non il tracking senza omografia?

Il tracking senza coordinate metriche **accumula persone fantasma**:

- La stessa persona viene vista come nuova entità ogni pochi frame
- Le vecchie entità restano in stato "occluso"
- Il conteggio cresce all'infinito

Risultato con tracking: una sala con 1 persona → conteggio fino a 59.

Soluzione: modalità detection-only, ogni frame parte da zero.

---

## SLIDE 10 — Rilevamento assembramenti

DBSCAN raggruppa le persone vicine:
- Raggio: ~2 metri (300 pixel nella scena del museo)
- Minimo: 3 persone per formare un cluster
- Se il cluster supera la soglia critica → **allarme**

[Immagine: frame con gruppo di visitatori e allarme attivo]

---

## SLIDE 11 — Benchmark: Modelli generici YOLOv8 (COCO)

Dataset pubblico Mall: 2000 frame, webcam centro commerciale, ~31 persone/frame, ground truth annotato.

**Approccio:** detection "person" → crop 22% superiore bbox come testa.

| Modello | MAE | RMSE | MAPE (%) | Media pred. | FPS |
|---------|-------|-------|----------|-------------|------|
| YOLOv8x | 21.15 | 21.96 | 67.5 | 10.0 | 0.4 |
| YOLOv8l | 20.54 | 21.26 | 65.7 | 10.6 | 0.6 |
| YOLOv8m | 19.83 | 20.48 | 63.7 | 11.3 | 1.3 |
| YOLOv8s | 20.27 | 20.90 | 65.2 | 10.9 | 2.9 |
| YOLOv8n | 20.61 | 21.25 | 66.3 | 10.5 | 5.7 |

GT media: 31.2. Tutti predicono ~10-11 persone (circa 1/3 del reale). Il modello 14× più pesante (x) non migliora rispetto al più leggero (n).

[Immagine: benchmark_results/confronto_modelli_yolov8/tabella_confronto_modelli.png]

---

## SLIDE 12 — Benchmark: Modelli specializzati head detection

Ipotesi: un modello addestrato per rilevare teste direttamente potrebbe fare meglio.

Testati due modelli YOLOv8 fine-tuned per head detection:
- **SCUT-HEAD** — addestrato su scene di aule universitarie (vista dall'alto)
- **iRail-HEAD** — addestrato su piattaforme ferroviarie e ingressi eventi

| Modello | MAE | RMSE | MAPE (%) | Media pred. | FPS |
|---------|-------|-------|----------|-------------|------|
| YOLOv8m_SCUT-HEAD | 26.14 | 26.87 | 84.1 | 5.0 | 1.3 |
| YOLOv8n_iRail-HEAD | 20.16 | 20.82 | 65.4 | 11.0 | 4.7 |

SCUT-HEAD è il peggiore (MAE 26.1, predice solo 5 persone): il suo dataset di training è troppo diverso dal Mall (domain shift). iRail-HEAD è competitivo con i modelli COCO (MAE 20.2).

[Immagine: benchmark_results/confronto_head/tabella_confronto_modelli.png]

---

## SLIDE 13 — Confronto completo: generici vs head detection

| Modello | Tipo | MAE | Media pred. | FPS |
|---------|------|-----|-------------|-----|
| YOLOv8m (COCO) | person→crop | 19.83 | 11.3 | 1.3 |
| YOLOv8n_iRail-HEAD | head diretto | 20.16 | 11.0 | 4.7 |
| YOLOv8n (COCO) | person→crop | 20.61 | 10.5 | 5.7 |
| YOLOv8m_SCUT-HEAD | head diretto | 26.14 | 5.0 | 1.3 |

**Risultato chiave:** cambiare modello non risolve l'undercount. Sia i modelli generici che quelli specializzati head detection predicono ~10-11 persone su 31.

Il problema non è il modello, ma il dominio:
- Persone piccole e parzialmente occluse
- Nessun modello pre-trained copre bene questa scena
- Servirebbero fine-tuning sul dataset specifico o density regression

**YOLOv8n COCO resta la scelta ottimale:** MAE comparabile, massima velocità (5.7 FPS).

[Immagine: benchmark_results/confronto_head/confronto_temporale_modelli.png]

---

## SLIDE 14 — Test sul museo: risultati

Video: 30 minuti, 1920×1080, 25 fps, sala delle statue.

- Senza bg filter: conteggio base 6-8 (statue)
- Con bg filter: conteggio corretto, riflette i visitatori reali
- Allarmi generati quando ≥3 persone vicine

[Immagine: screenshot con 1 persona rilevata, conteggio = 1]

[Immagine: screenshot con gruppo, allarme attivo]

---

## SLIDE 15 — Limiti del sistema

- **Detection intermittente**: la persona appare e scompare tra frame consecutivi
- **Zone morte**: aree scure o ai margini del grandangolo
- **Nessun tracking**: se una persona passa dietro una colonna, il conteggio cala temporaneamente
- **Sensibilità alla luce**: cambi di illuminazione possono attivare il background filter in modo errato

---

## SLIDE 16 — Sviluppi futuri

- **Omografia** → abilita tracking in metri, density map reale, metriche precise
- **Pose estimation** → posizione dai piedi, gestione occlusioni
- **Modello specializzato** → fine-tuning o density regression
- **Re-identification** → identità persistente attraverso le occlusioni
- **Analisi temporale** → flusso orario, tempo di permanenza, heatmap zone

---

## SLIDE 17 — Conclusioni

- Sistema funzionante per il monitoraggio museale con camera fissa
- Soluzione originale al problema delle statue (background subtraction)
- Architettura modulare: ogni componente migliorabile indipendentemente
- Benchmark quantitativo su Mall Dataset + test qualitativi su video reali
- Il design completo (tracking, pose, omografia) è nel codice, pronto per attivazione

---

## SLIDE 18 — Demo / Domande

[Video demo o live demo del sistema in funzione]

Grazie per l'attenzione.
