"""Pipeline completa per la stima della densità di folla.

Orchestra tutti i moduli del sistema in una catena sequenziale:
FrameGrabber → HeadDetector → HomographyModule → StateMachine →
CountingModule → DensityMapGenerator → CrowdClusterer.

Requirements: 12.1, 1.1, 2.1, 6.1, 7.1, 8.1
"""

import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

import cv2
import numpy as np

from .clustering import CrowdClusterer
from .config import CameraCalibration, PipelineConfig
from .counting import CountingModule
from .density import DensityMap, DensityMapGenerator
from .detection import HeadDetector
from .homography import HomographyModule, SceneBounds
from .models import (
    CountResult,
    CrowdAlarm,
    CrowdCluster,
    FusedDetection,
    HeadDetection,
    OcclusionType,
    TrackedPerson,
)
from .tracking import StateMachine

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Risultato dell'elaborazione di un singolo frame.

    Attributes:
        frame_id: Identificativo progressivo del frame.
        timestamp: Timestamp del frame in secondi.
        head_detections: Lista delle teste rilevate nel frame.
        tracked_persons: Lista delle persone tracciate attive.
        count_result: Risultato del conteggio persone.
        density_map: Mappa di densità generata, o None se non disponibile.
        clusters: Lista dei cluster rilevati.
        alarms: Lista degli allarmi generati.
    """

    frame_id: int
    timestamp: float
    head_detections: list[HeadDetection] = field(default_factory=list)
    tracked_persons: list[TrackedPerson] = field(default_factory=list)
    count_result: CountResult = field(
        default_factory=lambda: CountResult(
            frame_count=0.0,
            occluded_count=0.0,
            total_count=0.0,
            confidence_interval=(0.0, 0.0),
            persons_visible=0,
            persons_partial=0,
            persons_occluded=0,
        )
    )
    density_map: Optional[DensityMap] = None
    clusters: list[CrowdCluster] = field(default_factory=list)
    alarms: list[CrowdAlarm] = field(default_factory=list)


# Confini di scena di default quando l'omografia non è calibrata
_DEFAULT_SCENE_BOUNDS: SceneBounds = (0.0, 0.0, 20.0, 20.0)


class CrowdDensityPipeline:
    """Orchestra la catena completa di elaborazione per la crowd density estimation.

    Moduli orchestrati:
        HeadDetector → HomographyModule → StateMachine →
        CountingModule → DensityMapGenerator → CrowdClusterer

    La pipeline opera con un singolo flusso video monoculare (Req 12.1).
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Inizializza la pipeline con tutti i moduli interni.

        Args:
            config: Configurazione della pipeline. Se None, usa valori di default.
        """
        self._config = config or PipelineConfig()
        self._is_running = False

        # Inizializza i moduli
        self._detector = HeadDetector()
        self._homography = HomographyModule(
            calibration=self._config.camera,
            reprojection_threshold=self._config.reprojection_error_threshold,
        )
        self._state_machine = StateMachine(config=self._config.tracking)
        self._counting = CountingModule()
        self._density_generator = DensityMapGenerator(
            resolution=self._config.density_map_resolution,
        )
        self._clusterer = CrowdClusterer()

    @property
    def config(self) -> PipelineConfig:
        """Configurazione corrente della pipeline."""
        return self._config

    @property
    def is_running(self) -> bool:
        """True se la pipeline sta elaborando un video."""
        return self._is_running

    @property
    def homography(self) -> HomographyModule:
        """Accesso al modulo omografico (per calibrazione esterna)."""
        return self._homography

    def _is_homography_calibrated(self) -> bool:
        """Verifica se il modulo omografico ha una matrice valida."""
        return self._homography.homography_matrix is not None

    def _get_scene_bounds(self) -> SceneBounds:
        """Restituisce i confini della scena o un default ragionevole."""
        bounds = self._homography.scene_bounds
        if bounds is not None:
            return bounds
        return _DEFAULT_SCENE_BOUNDS

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> FrameResult:
        """Esegue la catena completa di elaborazione per un singolo frame.

        Sequenza:
            1. HeadDetector.detect(frame) → head_detections
            2. Per ogni head, HomographyModule.image_to_ground(center) → ground_position
            3. Costruisce FusedDetection (pose=None nel MVP)
            4. StateMachine.update(frame_id, timestamp, detections) → tracked_persons
            5. CountingModule.compute_count(tracked_persons) → count_result
            6. DensityMapGenerator.generate(tracked_persons, scene_bounds) → density_map
            7. CrowdClusterer.detect_crowds(tracked_persons, config.crowd) → clusters, alarms

        Args:
            frame: Frame RGB con shape (H, W, 3) e dtype uint8.
            frame_id: Identificativo progressivo del frame.
            timestamp: Timestamp del frame in secondi.

        Returns:
            FrameResult con tutti i risultati dell'elaborazione.
        """
        # 1. Rilevamento teste
        head_detections = self._detector.detect(frame)

        # 2-3. Proiezione omografica e creazione FusedDetection
        fused_detections = self._create_fused_detections(head_detections)

        # 4. Aggiornamento macchina a stati (tracking)
        tracked_persons = self._state_machine.update(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=fused_detections,
        )

        # 5. Conteggio persone
        count_result = self._counting.compute_count(tracked_persons)

        # 6. Generazione mappa di densità
        scene_bounds = self._get_scene_bounds()
        density_map: Optional[DensityMap] = None
        try:
            density_map = self._density_generator.generate(
                persons=tracked_persons,
                scene_bounds=scene_bounds,
            )
        except Exception as e:
            logger.warning("Errore nella generazione della mappa di densità: %s", e)

        # 7. Rilevamento assembramenti (clustering)
        clusters, alarms = self._clusterer.detect_crowds(
            persons=tracked_persons,
            config=self._config.crowd,
        )

        # Log allarmi
        for alarm in alarms:
            logger.warning("ALLARME %s: %s", alarm.alarm_level.upper(), alarm.message)

        return FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            head_detections=head_detections,
            tracked_persons=tracked_persons,
            count_result=count_result,
            density_map=density_map,
            clusters=clusters,
            alarms=alarms,
        )

    def _create_fused_detections(
        self, head_detections: list[HeadDetection]
    ) -> list[FusedDetection]:
        """Crea FusedDetection da ogni HeadDetection con proiezione omografica.

        Nel MVP non c'è PoseEstimator, quindi:
        - pose = None
        - position_source = "head"
        - occlusion_type = OcclusionType.NESSUNA

        Se l'omografia non è calibrata, ground_position sarà None
        e verrà loggato un warning.

        Args:
            head_detections: Lista delle teste rilevate.

        Returns:
            Lista di FusedDetection pronte per il tracking.
        """
        calibrated = self._is_homography_calibrated()
        if not calibrated and head_detections:
            logger.warning(
                "Omografia non calibrata: le posizioni ground non saranno disponibili. "
                "Calibrare il modulo omografico per abilitare la proiezione."
            )

        fused: list[FusedDetection] = []
        for head in head_detections:
            ground_position = None
            if calibrated:
                try:
                    ground_position = self._homography.image_to_ground(head.center)
                except ValueError:
                    # Non dovrebbe accadere se calibrated == True, ma gestisci
                    logger.warning(
                        "Errore nella proiezione omografica per punto (%f, %f)",
                        head.center.x,
                        head.center.y,
                    )

            fused.append(
                FusedDetection(
                    head=head,
                    pose=None,
                    confidence=head.confidence,
                    occlusion_type=OcclusionType.NESSUNA,
                    ground_position=ground_position,
                    position_source="head",
                )
            )

        return fused

    def run(self, video_path: str) -> Generator[FrameResult, None, None]:
        """Elabora un video completo frame per frame.

        Apre il video con cv2.VideoCapture e cicla su tutti i frame,
        invocando process_frame per ciascuno. Gestisce un singolo flusso
        video monoculare (Req 12.1): se la pipeline è già in esecuzione,
        rifiuta il nuovo flusso.

        Args:
            video_path: Percorso al file video da elaborare.

        Yields:
            FrameResult per ogni frame elaborato.

        Raises:
            RuntimeError: Se la pipeline è già in esecuzione (Req 12.1).
            FileNotFoundError: Se il file video non esiste o non è apribile.
        """
        # Req 12.1: rifiutare flussi aggiuntivi
        if self._is_running:
            raise RuntimeError(
                "La pipeline è già in esecuzione con un flusso video. "
                "Operare con un singolo flusso video monoculare (Req 12.1)."
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Impossibile aprire il video: {video_path}"
            )

        self._is_running = True
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_id / fps

                # cv2 legge in BGR, converti a RGB per il detector
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = self.process_frame(
                    frame=frame_rgb,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
                yield result

                frame_id += 1

            logger.info(
                "Elaborazione video completata: %d frame elaborati.", frame_id
            )
        finally:
            cap.release()
            self._is_running = False
