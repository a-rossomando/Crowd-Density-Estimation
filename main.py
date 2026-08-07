"""Script di esecuzione demo per la pipeline di crowd density estimation.

Accetta argomenti CLI per configurare ed eseguire la pipeline su un video,
stampando i risultati finali: conteggio medio, picco e numero allarmi.

Requirements: 12.4, 3.1, 8.4
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from crowd_density.config import CameraCalibration, CrowdConfig, PipelineConfig, TrackingConfig
from crowd_density.homography import HomographyModule, SceneBounds
from crowd_density.models import GroundPoint, PixelPoint
from crowd_density.pipeline import CrowdDensityPipeline, FrameResult

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parsing degli argomenti da linea di comando.

    Returns:
        Namespace con tutti gli argomenti parsati.
    """
    parser = argparse.ArgumentParser(
        description="Demo pipeline crowd density estimation. "
        "Elabora un video e stampa statistiche di conteggio e allarmi.",
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Percorso al file video (MP4, AVI, MOV).",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        help="Percorso al file JSON di calibrazione (almeno 4 punti).",
    )
    parser.add_argument(
        "--temporal-window",
        type=float,
        default=5.0,
        help="Finestra temporale in secondi (default: 5.0).",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=2.0,
        help="Raggio di clustering in metri (default: 2.0).",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=1.0,
        help="Soglia densità per assembramento in persone/m² (default: 1.0).",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=2.0,
        help="Soglia critica di densità per allarme (default: 2.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Percorso per salvare il video di output.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Non mostrare la finestra cv2.imshow.",
    )

    return parser.parse_args()


def load_calibration(calibration_path: str) -> tuple[list[tuple[PixelPoint, GroundPoint]], SceneBounds | None]:
    """Carica i dati di calibrazione da file JSON.

    Formato atteso:
    {
        "correspondences": [
            {"pixel": [x1, y1], "ground": [gx1, gy1]},
            ...
        ],
        "scene_bounds": [x_min, y_min, x_max, y_max]  (opzionale)
    }

    Args:
        calibration_path: Percorso al file JSON di calibrazione.

    Returns:
        Tupla (corrispondenze, scene_bounds).

    Raises:
        FileNotFoundError: Se il file non esiste.
        ValueError: Se il formato è invalido o ci sono meno di 4 punti.
    """
    path = Path(calibration_path)
    if not path.exists():
        raise FileNotFoundError(f"File di calibrazione non trovato: {calibration_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "correspondences" not in data:
        raise ValueError(
            "Il file di calibrazione deve contenere il campo 'correspondences'."
        )

    raw_correspondences = data["correspondences"]
    if len(raw_correspondences) < 4:
        raise ValueError(
            f"Servono almeno 4 punti di corrispondenza, trovati: {len(raw_correspondences)}"
        )

    correspondences: list[tuple[PixelPoint, GroundPoint]] = []
    for i, entry in enumerate(raw_correspondences):
        if "pixel" not in entry or "ground" not in entry:
            raise ValueError(
                f"Corrispondenza {i}: deve avere campi 'pixel' e 'ground'."
            )
        px, py = entry["pixel"]
        gx, gy = entry["ground"]
        correspondences.append(
            (PixelPoint(x=float(px), y=float(py)), GroundPoint(x=float(gx), y=float(gy)))
        )

    scene_bounds: SceneBounds | None = None
    if "scene_bounds" in data:
        sb = data["scene_bounds"]
        if len(sb) != 4:
            raise ValueError("scene_bounds deve contenere 4 valori: [x_min, y_min, x_max, y_max]")
        scene_bounds = (float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3]))

    return correspondences, scene_bounds


def draw_frame_overlay(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Disegna overlay informativo sul frame.

    Mostra conteggio corrente, numero di persone tracciate e allarmi attivi.

    Args:
        frame: Frame BGR originale.
        result: Risultato dell'elaborazione del frame.

    Returns:
        Frame con overlay disegnato.
    """
    overlay = frame.copy()

    # Informazioni di conteggio
    count_text = f"Conteggio: {result.count_result.total_count:.1f}"
    cv2.putText(
        overlay, count_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    # Numero di tracciati
    tracked_text = f"Tracciati: {len(result.tracked_persons)}"
    cv2.putText(
        overlay, tracked_text, (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
    )

    # Allarmi attivi
    if result.alarms:
        alarm_text = f"ALLARMI: {len(result.alarms)}"
        cv2.putText(
            overlay, alarm_text, (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )

    # Bounding box per ogni testa rilevata
    for detection in result.head_detections:
        bbox = detection.bbox
        x1, y1 = int(bbox.x), int(bbox.y)
        x2, y2 = int(bbox.x + bbox.width), int(bbox.y + bbox.height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return overlay


def main() -> None:
    """Punto di ingresso principale dello script demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Validazione file video
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("File video non trovato: %s", args.video)
        sys.exit(1)

    # Caricare calibrazione da JSON
    logger.info("Caricamento calibrazione da: %s", args.calibration)
    try:
        correspondences, scene_bounds = load_calibration(args.calibration)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.error("Errore nel caricamento della calibrazione: %s", e)
        sys.exit(1)

    logger.info("Calibrazione caricata: %d punti di corrispondenza", len(correspondences))

    # Configurazione pipeline
    tracking_config = TrackingConfig(
        temporal_window_seconds=args.temporal_window,
    )
    crowd_config = CrowdConfig(
        cluster_radius_meters=args.cluster_radius,
        density_threshold=args.density_threshold,
        critical_threshold=args.critical_threshold,
    )

    config = PipelineConfig(
        tracking=tracking_config,
        crowd=crowd_config,
        video_source=args.video,
        output_path=args.output,
        display_output=not args.no_display,
    )

    # Validare configurazione
    try:
        config.validate()
    except ValueError as e:
        logger.error("Configurazione non valida: %s", e)
        sys.exit(1)

    # Creare pipeline
    pipeline = CrowdDensityPipeline(config=config)

    # Calibrare omografia manuale (Req 12.4)
    logger.info("Calibrazione omografia manuale con %d punti...", len(correspondences))
    try:
        homography_module = HomographyModule.calibrate_manual(
            correspondences=correspondences,
            scene_bounds=scene_bounds,
        )
        # Impostare la matrice calcolata nel modulo della pipeline
        pipeline.homography._homography_matrix = homography_module.homography_matrix
        if scene_bounds is not None:
            pipeline.homography.scene_bounds = scene_bounds
        logger.info("Calibrazione omografia completata con successo.")
    except ValueError as e:
        logger.error("Errore nella calibrazione omografica: %s", e)
        sys.exit(1)

    # Inizializzare video writer se richiesto
    video_writer = None
    if args.output:
        cap_temp = cv2.VideoCapture(args.video)
        fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_temp.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        logger.info("Video di output: %s (%dx%d @ %.1f fps)", args.output, width, height, fps)

    # Statistiche
    total_frames = 0
    total_count = 0.0
    peak_count = 0.0
    total_alarms = 0

    # Apertura separata del video per la visualizzazione (il pipeline usa la sua)
    display_cap = None
    if not args.no_display or video_writer is not None:
        display_cap = cv2.VideoCapture(args.video)

    # Elaborazione video
    logger.info("Avvio elaborazione video: %s", args.video)

    try:
        for result in pipeline.run(args.video):
            total_frames += 1
            frame_count = result.count_result.total_count
            total_count += frame_count
            if frame_count > peak_count:
                peak_count = frame_count
            total_alarms += len(result.alarms)

            # Visualizzazione
            if display_cap is not None:
                ret, frame_bgr = display_cap.read()
                if ret:
                    frame_overlay = draw_frame_overlay(frame_bgr, result)

                    if video_writer is not None:
                        video_writer.write(frame_overlay)

                    if not args.no_display:
                        cv2.imshow("Crowd Density Estimation", frame_overlay)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:  # 'q' o ESC
                            logger.info("Interruzione utente (tasto premuto).")
                            break

    except KeyboardInterrupt:
        logger.info("Interruzione utente (Ctrl+C). Stampa risultati parziali.")
    finally:
        # Cleanup
        if display_cap is not None:
            display_cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Stampare risultati finali
    avg_count = total_count / total_frames if total_frames > 0 else 0.0

    print()
    print("=== Risultati elaborazione ===")
    print(f"Frame totali: {total_frames}")
    print(f"Conteggio medio: {avg_count:.1f} persone")
    print(f"Conteggio picco: {peak_count:.1f} persone")
    print(f"Allarmi generati: {total_alarms}")


if __name__ == "__main__":
    main()
