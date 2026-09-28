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

from .background import BackgroundFilter
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
    GroundPoint,
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
        self._pixel_fallback_warned = False

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
        self._background_filter: Optional[BackgroundFilter] = None

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

    @property
    def background_filter(self) -> Optional[BackgroundFilter]:
        """Accesso al filtro background, o None se non abilitato."""
        return self._background_filter

    def enable_background_filter(
        self,
        warmup_frames: int = 60,
        diff_threshold: float = 30.0,
        static_ratio: float = 0.15,
    ) -> None:
        """Abilita il filtro background subtraction.

        Le detection in regioni statiche (es. statue) saranno scartate.
        I primi warmup_frames frame vengono usati per costruire il modello
        di sfondo; durante quel periodo non vengono emesse detection.

        Args:
            warmup_frames: Frame per costruire il modello di sfondo.
            diff_threshold: Soglia di differenza per pixel (0-255).
            static_ratio: Proporzione minima di pixel cambiati per
                considerare una detection come persona vera.
        """
        self._background_filter = BackgroundFilter(
            warmup_frames=warmup_frames,
            diff_threshold=diff_threshold,
            static_ratio=static_ratio,
        )
        logger.info(
            "Background filter abilitato: warmup=%d frame, "
            "diff_threshold=%.1f, static_ratio=%.2f",
            warmup_frames, diff_threshold, static_ratio,
        )

    def enable_background_filter_from_image(
        self,
        image_path: str,
        diff_threshold: float = 30.0,
        static_ratio: float = 0.15,
    ) -> None:
        """Abilita il filtro usando un modello di sfondo pre-calcolato.

        Nessun warmup necessario: il filtro è attivo dal primo frame.

        Args:
            image_path: Percorso all'immagine di sfondo.
            diff_threshold: Soglia di differenza per pixel (0-255).
            static_ratio: Proporzione minima di pixel cambiati.
        """
        self._background_filter = BackgroundFilter.from_image(
            image_path=image_path,
            diff_threshold=diff_threshold,
            static_ratio=static_ratio,
        )

    def _is_homography_calibrated(self) -> bool:
        """Verifica se il modulo omografico ha una matrice valida."""
        return self._homography.homography_matrix is not None

    def _get_scene_bounds(self, frame_shape: tuple[int, ...] | None = None) -> SceneBounds:
        """Restituisce i confini della scena.

        Se l'omografia è calibrata, usa i bounds configurati o un default
        in metri. Altrimenti, usa le dimensioni del frame in pixel come
        bounds (fallback pixel-space).

        Args:
            frame_shape: Shape del frame (H, W, C). Usato per il fallback pixel.
        """
        bounds = self._homography.scene_bounds
        if bounds is not None:
            return bounds
        if not self._is_homography_calibrated() and frame_shape is not None:
            h, w = frame_shape[:2]
            return (0.0, 0.0, float(w), float(h))
        return _DEFAULT_SCENE_BOUNDS

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> FrameResult:
        """Esegue la catena completa di elaborazione per un singolo frame.

        Con omografia calibrata (modalità completa):
            1. HeadDetector.detect(frame) → head_detections
            2. HomographyModule.image_to_ground → ground_position (metri)
            3. StateMachine.update → tracked_persons (tracking temporale)
            4. CountingModule.compute_count → count_result
            5. DensityMapGenerator.generate → density_map
            6. CrowdClusterer.detect_crowds → clusters, alarms

        Senza omografia (modalità detection-only):
            1. HeadDetector.detect(frame) → head_detections
            2. Background filter (se abilitato) → filtra statici
            3. Conteggio diretto da detection (no tracking)
            4. Clustering sulle detection del frame corrente (no accumulo)

        Args:
            frame: Frame RGB con shape (H, W, 3) e dtype uint8.
            frame_id: Identificativo progressivo del frame.
            timestamp: Timestamp del frame in secondi.

        Returns:
            FrameResult con tutti i risultati dell'elaborazione.
        """
        # 1. Rilevamento teste
        head_detections = self._detector.detect(frame)

        # 1b. Filtro background: scarta detection statiche (es. statue)
        if self._background_filter is not None:
            if not self._background_filter.is_ready:
                self._background_filter.feed_frame(frame)
                head_detections = []
            else:
                head_detections = self._background_filter.filter_detections(
                    frame, head_detections
                )

        if self._is_homography_calibrated():
            return self._process_frame_calibrated(
                frame, head_detections, frame_id, timestamp
            )
        else:
            return self._process_frame_pixel(
                frame, head_detections, frame_id, timestamp
            )

    def _process_frame_calibrated(
        self,
        frame: np.ndarray,
        head_detections: list[HeadDetection],
        frame_id: int,
        timestamp: float,
    ) -> FrameResult:
        """Elaborazione completa con omografia: tracking + counting + clustering in metri."""
        # 2-3. Proiezione omografica e creazione FusedDetection
        fused_detections = self._create_fused_detections(head_detections)

        # 4. Tracking temporale
        tracked_persons = self._state_machine.update(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=fused_detections,
        )

        # 5. Conteggio
        count_result = self._counting.compute_count(tracked_persons)

        # 6. Density map
        scene_bounds = self._get_scene_bounds(frame_shape=frame.shape)
        density_map: Optional[DensityMap] = None
        try:
            density_map = self._density_generator.generate(
                persons=tracked_persons,
                scene_bounds=scene_bounds,
            )
        except Exception as e:
            logger.warning("Errore nella generazione della mappa di densità: %s", e)

        # 7. Clustering
        clusters, alarms = self._clusterer.detect_crowds(
            persons=tracked_persons,
            config=self._config.crowd,
        )

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

    def _process_frame_pixel(
        self,
        frame: np.ndarray,
        head_detections: list[HeadDetection],
        frame_id: int,
        timestamp: float,
    ) -> FrameResult:
        """Elaborazione detection-only senza omografia.

        Non usa il tracking temporale per evitare accumulo di persone fantasma.
        Il conteggio si basa direttamente sulle detection del frame corrente.
        Il clustering lavora sulle posizioni pixel delle detection correnti.
        """
        if not self._pixel_fallback_warned and head_detections:
            logger.info(
                "Modalità detection-only (no omografia): conteggio e clustering "
                "basati sulle detection del frame corrente, senza tracking temporale."
            )
            self._pixel_fallback_warned = True

        num_detected = len(head_detections)

        # Conteggio diretto dalle detection
        count_result = CountResult(
            frame_count=float(num_detected),
            occluded_count=0.0,
            total_count=float(num_detected),
            confidence_interval=(float(num_detected), float(num_detected)),
            persons_visible=num_detected,
            persons_partial=0,
            persons_occluded=0,
        )

        # Creare TrackedPerson temporanee per clustering e density map
        # (solo per il frame corrente, non persistono tra frame)
        temp_persons: list[TrackedPerson] = []
        from .models import PersonState
        for i, det in enumerate(head_detections):
            gp = GroundPoint(x=det.center.x, y=det.center.y)
            temp_persons.append(TrackedPerson(
                person_id=i,
                state=PersonState.VISIBILE,
                confidence=det.confidence,
                ground_position=gp,
            ))

        # Density map in pixel
        scene_bounds = self._get_scene_bounds(frame_shape=frame.shape)
        density_map: Optional[DensityMap] = None
        try:
            pixel_density_gen = DensityMapGenerator(
                resolution=0.5,
                sigma=25.0,
            )
            density_map = pixel_density_gen.generate(
                persons=temp_persons,
                scene_bounds=scene_bounds,
            )
        except Exception as e:
            logger.warning("Errore nella generazione della mappa di densità: %s", e)

        # Clustering in pixel
        clusters: list[CrowdCluster] = []
        alarms: list[CrowdAlarm] = []
        if len(temp_persons) >= self._config.crowd.min_cluster_size:
            from dataclasses import replace
            import math
            # La scala pixel/metro dipende dalla scena. Per un frame 1920px
            # che copre ~12 metri di larghezza: ~160 pixel/metro.
            pixel_scale = 150.0
            eps_pixel = self._config.crowd.cluster_radius_meters * pixel_scale
            min_possible_density = (
                self._config.crowd.min_cluster_size
                / (math.pi * eps_pixel * eps_pixel)
            )
            pixel_crowd_config = replace(
                self._config.crowd,
                cluster_radius_meters=eps_pixel,
                density_threshold=min_possible_density * 0.9,
                critical_threshold=min_possible_density * 1.8,
            )
            clusters, alarms = self._clusterer.detect_crowds(
                persons=temp_persons,
                config=pixel_crowd_config,
            )

        for alarm in alarms:
            cluster = alarm.cluster
            logger.warning(
                "ALLARME %s: assembramento di %d persone rilevato",
                alarm.alarm_level.upper(),
                cluster.person_count,
            )

        return FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            head_detections=head_detections,
            tracked_persons=temp_persons,
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

        Se l'omografia non è calibrata, usa le coordinate pixel del centro
        della testa come fallback per ground_position. In questo modo il
        tracking, il clustering e la density map funzionano anche senza
        calibrazione (le coordinate saranno in pixel anziché in metri).

        Args:
            head_detections: Lista delle teste rilevate.

        Returns:
            Lista di FusedDetection pronte per il tracking.
        """
        calibrated = self._is_homography_calibrated()
        if not calibrated and head_detections and not self._pixel_fallback_warned:
            logger.info(
                "Omografia non calibrata: uso coordinate pixel come fallback. "
                "Clustering e density map opereranno in pixel."
            )
            self._pixel_fallback_warned = True

        fused: list[FusedDetection] = []
        for head in head_detections:
            ground_position = None
            if calibrated:
                try:
                    ground_position = self._homography.image_to_ground(head.center)
                except ValueError:
                    logger.warning(
                        "Errore nella proiezione omografica per punto (%f, %f)",
                        head.center.x,
                        head.center.y,
                    )

            # Fallback: usa coordinate pixel del centro testa
            if ground_position is None:
                ground_position = GroundPoint(
                    x=head.center.x, y=head.center.y
                )

            fused.append(
                FusedDetection(
                    head=head,
                    pose=None,
                    confidence=head.confidence,
                    occlusion_type=OcclusionType.NESSUNA,
                    ground_position=ground_position,
                    position_source="head_pixel" if not calibrated else "head",
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
