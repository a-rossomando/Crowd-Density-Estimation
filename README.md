# Crowd Density Estimation

Sistema di stima della densità e distribuzione di persone in una scena interna (sala museale con opere d'arte, statue e colonne) tramite una singola telecamera monoculare.

## Descrizione

Il progetto implementa un approccio a **due livelli** per il conteggio e il tracciamento delle persone:

1. **Rilevamento per-frame** — localizzazione delle teste e stima della posa (scheletro) quando il corpo è completamente visibile.
2. **Macchina a stati temporale** — tracking delle persone attraverso le occlusioni utilizzando la storia dei frame passati.

### Funzionalità principali

- Localizzazione delle teste tramite rete neurale (YOLOv8)
- Tracking temporale con gestione degli stati (visibile, parzialmente occluso, occluso, scomparso)
- Proiezione omografica delle posizioni sul piano del suolo
- Generazione di mappe di densità spaziale
- Rilevamento assembramenti con allarmi configurabili
- Gestione dell'occlusione ambientale (colonne, statue) e interpersonale

## Struttura del progetto

```
Crowd-Density-Estimation/
├── main.py                    # Script di esecuzione demo (CLI)
├── requirements.txt           # Dipendenze Python
├── yolov8n.pt                 # Modello YOLOv8 pre-addestrato
├── crowd_density/             # Package principale
│   ├── __init__.py
│   ├── config.py              # Configurazione e validazione parametri
│   ├── detection.py           # Localizzazione teste
│   ├── tracking.py            # Macchina a stati temporale
│   ├── homography.py          # Trasformazione omografica
│   ├── density.py             # Mappa di densità spaziale
│   ├── counting.py            # Conteggio persone (due livelli)
│   ├── clustering.py          # Rilevamento assembramenti
│   ├── models.py              # Modelli dati (dataclass)
│   ├── pipeline.py            # Pipeline di elaborazione completa
│   └── visualization.py       # Visualizzazione risultati
└── tests/                     # Test suite
    ├── test_counting.py
    ├── test_density.py
    ├── test_detection.py
    └── test_tracking.py
```

## Requisiti

- Python 3.10+
- Telecamera monoculare con risoluzione minima 1920×1080 @ 15 fps

### Dipendenze

- [Ultralytics (YOLOv8)](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

## Installazione

```bash
# Clonare il repository
git clone <url-repository>
cd Crowd-Density-Estimation

# Creare un virtual environment (consigliato)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS

# Installare le dipendenze
pip install -r requirements.txt
```

## Utilizzo

### Calibrazione

Prima dell'esecuzione è necessario un file JSON di calibrazione con almeno 4 punti di corrispondenza tra coordinate pixel e coordinate nel piano del suolo:

```json
{
  "correspondences": [
    {"pixel": [100, 500], "ground": [0.0, 0.0]},
    {"pixel": [800, 500], "ground": [5.0, 0.0]},
    {"pixel": [800, 200], "ground": [5.0, 8.0]},
    {"pixel": [100, 200], "ground": [0.0, 8.0]}
  ],
  "scene_bounds": [0.0, 0.0, 5.0, 8.0]
}
```

### Esecuzione

```bash
python main.py --video percorso/al/video.mp4 --calibration percorso/calibrazione.json
```

### Opzioni CLI

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--video` | *(obbligatorio)* | Percorso al file video (MP4, AVI, MOV) |
| `--calibration` | *(obbligatorio)* | Percorso al file JSON di calibrazione |
| `--temporal-window` | `5.0` | Finestra temporale in secondi (range: 1–60) |
| `--cluster-radius` | `2.0` | Raggio di clustering in metri (range: 0.5–5.0) |
| `--density-threshold` | `1.0` | Soglia densità per assembramento (persone/m²) |
| `--critical-threshold` | `2.0` | Soglia critica per allarme (deve essere > density-threshold) |
| `--output` | `None` | Percorso per salvare il video di output |
| `--no-display` | `False` | Non mostrare la finestra di visualizzazione |

### Esempio

```bash
python main.py \
  --video sala_museo.mp4 \
  --calibration calibrazione_sala.json \
  --temporal-window 8.0 \
  --cluster-radius 1.5 \
  --density-threshold 0.8 \
  --critical-threshold 1.5 \
  --output risultato.mp4
```

## Test

```bash
pytest tests/
```

## Output

Il sistema produce:

- **Conteggio in tempo reale** — numero stimato di persone con intervallo di confidenza
- **Mappa di densità** — distribuzione spaziale delle persone nel piano del suolo
- **Allarmi assembramento** — segnalazioni quando la densità supera la soglia critica
- **Video annotato** (opzionale) — frame con bounding box, conteggio e allarmi sovrapposti

### Risultati finali (stdout)

```
=== Risultati elaborazione ===
Frame totali: 450
Conteggio medio: 7.3 persone
Conteggio picco: 12.0 persone
Allarmi generati: 2
```

## Contesto accademico

Progetto sviluppato per il corso **Advanced Digital Image Processing** (ADIP), Università di Siena.

## Licenza

Progetto accademico — uso interno.
