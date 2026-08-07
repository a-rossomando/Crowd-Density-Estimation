"""Data models e enumerazioni per il sistema di crowd density estimation.

Questo modulo definisce tutte le strutture dati fondamentali utilizzate dal sistema:
- Enumerazioni per stati persona e tipi di occlusione
- Dataclass per coordinate (pixel e ground plane)
- Dataclass per rilevamenti (testa, pose, fusione)
- Dataclass per tracking e conteggio
- Dataclass per clustering e allarmi
- Configurazioni con valori di default
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# --- Enumerazioni ---


class PersonState(Enum):
    """Stato di una persona tracciata nella macchina a stati.

    Transizioni possibili:
        visibile <-> parzialmente_occluso <-> occluso -> scomparso
    """

    VISIBILE = "visibile"
    PARZIALMENTE_OCCLUSO = "parzialmente_occluso"
    OCCLUSO = "occluso"
    SCOMPARSO = "scomparso"


class OcclusionType(Enum):
    """Tipo di occlusione rilevata per una persona.

    - NESSUNA: corpo intero visibile
    - PARZIALE_INTERPERSONALE: occlusa da un'altra persona
    - OCCLUSA_AMBIENTALE: occlusa da un oggetto della scena (colonna, statua, ecc.)
    """

    NESSUNA = "nessuna"
    PARZIALE_INTERPERSONALE = "parziale_interpersonale"
    OCCLUSA_AMBIENTALE = "occlusa_ambientale"


# --- Coordinate ---


@dataclass(frozen=True)
class PixelPoint:
    """Punto in coordinate pixel nel piano immagine."""

    x: float
    y: float


@dataclass(frozen=True)
class GroundPoint:
    """Punto in coordinate metriche nel piano del suolo."""

    x: float  # metri
    y: float  # metri


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box rettangolare in coordinate pixel."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        """Calcola l'area della bounding box in pixel."""
        return self.width * self.height


# --- Rilevamenti ---


@dataclass
class HeadDetection:
    """Rilevamento di una testa nel frame corrente.

    Attributes:
        bbox: Bounding box della testa in coordinate pixel.
        confidence: Punteggio di confidenza nel range [0.0, 1.0].
        center: Centro stimato della testa in coordinate pixel.
        visible_ratio: Percentuale dell'area della testa visibile (non occlusa).
    """

    bbox: BoundingBox
    confidence: float  # [0.0, 1.0]
    center: PixelPoint
    visible_ratio: float  # percentuale area visibile


@dataclass
class PoseEstimation:
    """Stima della posa corporea di una persona.

    Attributes:
        keypoints: Array numpy di shape (17, 3) con x, y, confidence per ogni keypoint.
        full_body: True se almeno 80% dei keypoint hanno confidenza >= 0.5.
        feet_position: Punto medio delle caviglie, o None se non disponibile.
        silhouette_mask: Maschera binaria della silhouette, o None se non estratta.
    """

    keypoints: np.ndarray  # shape (17, 3) — x, y, confidence
    full_body: bool
    feet_position: Optional[PixelPoint]
    silhouette_mask: Optional[np.ndarray]


@dataclass
class FusedDetection:
    """Rilevamento fuso che combina testa e posa.

    Attributes:
        head: Rilevamento della testa associato.
        pose: Stima della posa, se disponibile.
        confidence: Confidenza finale del rilevamento fuso.
        occlusion_type: Tipo di occlusione classificata.
        ground_position: Posizione nel piano del suolo, se stimabile.
        position_source: Sorgente della posizione: "feet", "head", o "predicted".
    """

    head: HeadDetection
    pose: Optional[PoseEstimation]
    confidence: float
    occlusion_type: OcclusionType
    ground_position: Optional[GroundPoint]
    position_source: str  # "feet" | "head" | "predicted"


# --- Tracking ---


@dataclass
class TrackedPerson:
    """Persona tracciata dalla macchina a stati temporale.

    Attributes:
        person_id: Identificativo univoco della persona.
        state: Stato corrente nella macchina a stati.
        confidence: Confidenza corrente associata alla persona.
        ground_position: Ultima posizione nota nel piano del suolo.
        trajectory: Storico delle posizioni nel piano del suolo.
        frames_since_last_seen: Numero di frame dall'ultimo rilevamento diretto.
        occlusion_start_time: Timestamp di inizio occlusione, se applicabile.
        predicted_reappearance: Posizione prevista di riapparizione.
        last_detection: Ultimo rilevamento fuso associato.
    """

    person_id: int
    state: PersonState
    confidence: float
    ground_position: Optional[GroundPoint]
    trajectory: list[GroundPoint] = field(default_factory=list)
    frames_since_last_seen: int = 0
    occlusion_start_time: Optional[float] = None
    predicted_reappearance: Optional[GroundPoint] = None
    last_detection: Optional[FusedDetection] = None


# --- Risultati ---


@dataclass
class CountResult:
    """Risultato del conteggio persone per un frame.

    Attributes:
        frame_count: Somma delle confidenze dei rilevamenti diretti nel frame.
        occluded_count: Somma delle confidenze delle persone in stato occluso.
        total_count: frame_count + occluded_count.
        confidence_interval: Tupla (min, max) dell'intervallo di confidenza.
        persons_visible: Numero di persone rilevate con confidenza pari a 1.0.
        persons_partial: Numero di persone rilevate con confidenza inferiore a 1.0.
        persons_occluded: Numero di persone attualmente in stato occluso.
    """

    frame_count: float  # somma confidenze rilevamenti diretti
    occluded_count: float  # somma confidenze persone occluse
    total_count: float  # frame_count + occluded_count
    confidence_interval: tuple[float, float]  # (min, max)
    persons_visible: int  # numero persone con conf = 1.0
    persons_partial: int  # numero persone con conf < 1.0
    persons_occluded: int  # numero persone in stato occluso


@dataclass
class CrowdCluster:
    """Cluster di persone identificato nella mappa di densità.

    Attributes:
        cluster_id: Identificativo univoco del cluster.
        center: Centro geometrico del cluster nel piano del suolo.
        area_m2: Area del cluster in metri quadrati.
        person_count: Numero di persone nel cluster.
        density: Densità locale in persone/m².
        persons: Lista degli ID delle persone appartenenti al cluster.
        active_frames: Numero di frame in cui il cluster è stato attivo.
    """

    cluster_id: int
    center: GroundPoint
    area_m2: float
    person_count: int
    density: float  # persone/m²
    persons: list[int]  # ID delle persone nel cluster
    active_frames: int = 0


@dataclass
class CrowdAlarm:
    """Allarme generato quando un cluster supera la soglia critica di densità.

    Attributes:
        cluster: Il cluster che ha generato l'allarme.
        alarm_level: Livello dell'allarme: "warning" o "critical".
        message: Messaggio descrittivo dell'allarme.
    """

    cluster: CrowdCluster
    alarm_level: str  # "warning" | "critical"
    message: str


# --- Configurazione ---


@dataclass
class TrackingConfig:
    """Configurazione per il modulo di tracking temporale.

    Attributes:
        temporal_window_seconds: Finestra temporale in secondi [1.0, 60.0].
        tolerance_radius_meters: Raggio di tolleranza per riassociazione in metri.
        min_trajectory_frames: Numero minimo di frame per una traiettoria valida.
        frames_to_occluded: Frame consecutivi senza rilevamento per transitare a occluso.
        frames_to_partially_occluded: Frame con confidenza ridotta per transitare a parzialmente occluso.
        avg_person_height: Altezza media persona in metri [1.40, 2.10].
    """

    temporal_window_seconds: float = 5.0  # [1.0, 60.0]
    tolerance_radius_meters: float = 1.5
    min_trajectory_frames: int = 5
    frames_to_occluded: int = 3
    frames_to_partially_occluded: int = 2
    avg_person_height: float = 1.75  # [1.40, 2.10]


@dataclass
class CrowdConfig:
    """Configurazione per il rilevamento assembramenti.

    Attributes:
        min_cluster_size: Numero minimo di persone per formare un cluster.
        cluster_radius_meters: Raggio di clustering in metri [0.5, 5.0].
        density_threshold: Soglia di densità per rilevamento in persone/m².
        critical_threshold: Soglia critica (deve essere > density_threshold).
        dispersal_frames: Frame consecutivi sotto soglia per dichiarare dispersione.
    """

    min_cluster_size: int = 3
    cluster_radius_meters: float = 2.0  # [0.5, 5.0]
    density_threshold: float = 1.0  # persone/m²
    critical_threshold: float = 2.0  # > density_threshold
    dispersal_frames: int = 3
