"""Configurazione della pipeline di crowd density estimation.

Contiene tutte le dataclass di configurazione con valori default e validazione.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackingConfig:
    """Configurazione per il modulo di tracking temporale.

    Attributes:
        temporal_window_seconds: Finestra temporale in secondi per mantenere
            persone occluse nel conteggio. Range: [1.0, 60.0], default: 5.0.
            Risoluzione: 0.1 secondi.
        tolerance_radius_meters: Raggio di tolleranza in metri per il matching
            tra detection e persona tracciata.
        min_trajectory_frames: Numero minimo di frame nella traiettoria
            per predire la posizione di riapparizione.
        frames_to_occluded: Numero di frame consecutivi senza detection
            per transitare a stato occluso.
        frames_to_partially_occluded: Numero di frame consecutivi con
            confidenza ridotta per transitare a parzialmente_occluso.
        avg_person_height: Altezza media stimata di una persona in metri.
            Range: [1.40, 2.10], default: 1.75.
    """

    temporal_window_seconds: float = 5.0
    tolerance_radius_meters: float = 1.5
    min_trajectory_frames: int = 5
    frames_to_occluded: int = 3
    frames_to_partially_occluded: int = 2
    avg_person_height: float = 1.75

    def validate(self) -> None:
        """Valida i parametri di configurazione.

        Raises:
            ValueError: Se un parametro è fuori dal range consentito.
        """
        if not (1.0 <= self.temporal_window_seconds <= 60.0):
            raise ValueError(
                f"temporal_window_seconds deve essere tra 1.0 e 60.0 secondi, "
                f"ricevuto: {self.temporal_window_seconds}"
            )
        if not (1.40 <= self.avg_person_height <= 2.10):
            raise ValueError(
                f"avg_person_height deve essere tra 1.40 e 2.10 metri, "
                f"ricevuto: {self.avg_person_height}"
            )
        if self.tolerance_radius_meters <= 0:
            raise ValueError(
                f"tolerance_radius_meters deve essere positivo, "
                f"ricevuto: {self.tolerance_radius_meters}"
            )
        if self.min_trajectory_frames < 1:
            raise ValueError(
                f"min_trajectory_frames deve essere almeno 1, "
                f"ricevuto: {self.min_trajectory_frames}"
            )
        if self.frames_to_occluded < 1:
            raise ValueError(
                f"frames_to_occluded deve essere almeno 1, "
                f"ricevuto: {self.frames_to_occluded}"
            )
        if self.frames_to_partially_occluded < 1:
            raise ValueError(
                f"frames_to_partially_occluded deve essere almeno 1, "
                f"ricevuto: {self.frames_to_partially_occluded}"
            )


@dataclass
class CrowdConfig:
    """Configurazione per il rilevamento assembramenti.

    Attributes:
        min_cluster_size: Numero minimo di persone per formare un cluster.
        cluster_radius_meters: Raggio in metri per DBSCAN eps.
            Range: [0.5, 5.0], default: 2.0.
        density_threshold: Soglia di densità (persone/m²) per rilevare un assembramento.
        critical_threshold: Soglia critica di densità per generare un allarme.
            Deve essere strettamente maggiore di density_threshold.
        dispersal_frames: Numero di aggiornamenti consecutivi sotto soglia
            per dichiarare un assembramento disperso.
    """

    min_cluster_size: int = 3
    cluster_radius_meters: float = 2.0
    density_threshold: float = 1.0
    critical_threshold: float = 2.0
    dispersal_frames: int = 3

    def validate(self) -> None:
        """Valida i parametri di configurazione.

        Raises:
            ValueError: Se un parametro è fuori dal range consentito.
        """
        if not (0.5 <= self.cluster_radius_meters <= 5.0):
            raise ValueError(
                f"cluster_radius_meters deve essere tra 0.5 e 5.0 metri, "
                f"ricevuto: {self.cluster_radius_meters}"
            )
        if self.min_cluster_size < 1:
            raise ValueError(
                f"min_cluster_size deve essere almeno 1, "
                f"ricevuto: {self.min_cluster_size}"
            )
        if self.density_threshold <= 0:
            raise ValueError(
                f"density_threshold deve essere positivo, "
                f"ricevuto: {self.density_threshold}"
            )
        if self.critical_threshold <= self.density_threshold:
            raise ValueError(
                f"critical_threshold ({self.critical_threshold}) deve essere "
                f"strettamente maggiore di density_threshold ({self.density_threshold})"
            )
        if self.dispersal_frames < 1:
            raise ValueError(
                f"dispersal_frames deve essere almeno 1, "
                f"ricevuto: {self.dispersal_frames}"
            )


@dataclass
class CameraCalibration:
    """Parametri di calibrazione della telecamera.

    Attributes:
        intrinsic_matrix: Matrice intrinseca 3x3 della camera.
        distortion_coeffs: Coefficienti di distorsione della lente.
        extrinsic_rotation: Matrice di rotazione estrinseca 3x3.
        extrinsic_translation: Vettore di traslazione estrinseca 3x1.
        homography_matrix: Matrice omografica 3x3 (calcolata dalla calibrazione).
        reprojection_error: Errore di riproiezione medio.
    """

    intrinsic_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    distortion_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )
    extrinsic_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    extrinsic_translation: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 1), dtype=np.float64)
    )
    homography_matrix: Optional[np.ndarray] = None
    reprojection_error: Optional[float] = None


@dataclass
class PipelineConfig:
    """Configurazione aggregata per l'intera pipeline di crowd density estimation.

    Aggrega tutti i parametri configurabili del sistema: tracking, crowd detection,
    calibrazione camera e parametri generali della pipeline.

    Attributes:
        tracking: Configurazione del modulo di tracking temporale.
        crowd: Configurazione del rilevamento assembramenti.
        camera: Parametri di calibrazione della telecamera.
        video_source: Percorso del video o sorgente della telecamera.
        output_path: Percorso opzionale per salvare il video risultante.
        display_output: Se True, mostra l'output con cv2.imshow.
        density_map_resolution: Risoluzione della griglia per la mappa densità (celle/metro).
        min_video_resolution: Risoluzione minima richiesta per il video (larghezza, altezza).
        min_fps: Frame rate minimo richiesto.
        max_video_duration_seconds: Durata massima del video in secondi.
        reprojection_error_threshold: Soglia massima di errore di riproiezione.
    """

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    crowd: CrowdConfig = field(default_factory=CrowdConfig)
    camera: CameraCalibration = field(default_factory=CameraCalibration)
    video_source: str = ""
    output_path: Optional[str] = None
    display_output: bool = True
    density_map_resolution: float = 10.0
    min_video_resolution: tuple[int, int] = (1920, 1080)
    min_fps: float = 15.0
    max_video_duration_seconds: float = 1800.0
    reprojection_error_threshold: float = 5.0

    def validate(self) -> None:
        """Valida l'intera configurazione della pipeline.

        Raises:
            ValueError: Se un parametro è fuori dal range consentito.
        """
        self.tracking.validate()
        self.crowd.validate()

        if self.density_map_resolution <= 0:
            raise ValueError(
                f"density_map_resolution deve essere positivo, "
                f"ricevuto: {self.density_map_resolution}"
            )
        if self.min_fps <= 0:
            raise ValueError(
                f"min_fps deve essere positivo, ricevuto: {self.min_fps}"
            )
        if self.reprojection_error_threshold <= 0:
            raise ValueError(
                f"reprojection_error_threshold deve essere positivo, "
                f"ricevuto: {self.reprojection_error_threshold}"
            )
