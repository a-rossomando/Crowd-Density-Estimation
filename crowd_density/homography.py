"""Modulo omografia e calibrazione camera.

Gestisce la trasformazione prospettica immagine → piano del suolo,
la calibrazione manuale con almeno 4 punti, e la stima della posizione
a terra da posizione testa come fallback.

Requirements: 7.1, 7.4, 7.5, 12.3, 12.4
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import CameraCalibration
from .models import GroundPoint, PixelPoint

logger = logging.getLogger(__name__)


@dataclass
class ReprojectionResult:
    """Risultato della validazione di riproiezione.

    Attributes:
        mean_error: Errore di riproiezione medio in pixel.
        max_error: Errore di riproiezione massimo.
        errors: Lista degli errori per ogni punto.
        is_valid: True se l'errore medio è sotto la soglia.
    """

    mean_error: float
    max_error: float
    errors: list[float]
    is_valid: bool


# Tipo per le corrispondenze: (punto pixel, punto ground)
CorrespondencePoint = tuple[PixelPoint, GroundPoint]

# Tipo per i confini della scena: (x_min, y_min, x_max, y_max)
SceneBounds = tuple[float, float, float, float]


class HomographyModule:
    """Modulo per la trasformazione omografica immagine → piano del suolo.

    Gestisce:
    - Proiezione punti dal piano immagine al piano del suolo (Req 7.1)
    - Stima posizione a terra da testa con altezza media (Req 7.4)
    - Controllo confini scena e segnalazione anomalie (Req 7.5)
    - Calcolo e validazione errore di riproiezione (Req 12.3)
    - Calibrazione manuale con almeno 4 punti (Req 12.4)
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        scene_bounds: Optional[SceneBounds] = None,
        reprojection_threshold: float = 5.0,
    ):
        """Inizializza il modulo con i parametri di calibrazione.

        Args:
            calibration: Parametri di calibrazione della telecamera.
            scene_bounds: Confini della scena (x_min, y_min, x_max, y_max)
                in coordinate del piano del suolo. Se None, nessun controllo.
            reprojection_threshold: Soglia massima di errore di riproiezione
                accettabile in pixel.
        """
        self._calibration = calibration
        self._scene_bounds = scene_bounds
        self._reprojection_threshold = reprojection_threshold
        self._homography_matrix: Optional[np.ndarray] = None

        # Se la calibrazione contiene già una matrice omografica, usala
        if calibration.homography_matrix is not None:
            self._homography_matrix = calibration.homography_matrix.copy()

    @property
    def homography_matrix(self) -> Optional[np.ndarray]:
        """Restituisce la matrice omografica corrente, o None se non calibrato."""
        return self._homography_matrix

    @property
    def scene_bounds(self) -> Optional[SceneBounds]:
        """Restituisce i confini della scena configurati."""
        return self._scene_bounds

    @scene_bounds.setter
    def scene_bounds(self, bounds: Optional[SceneBounds]) -> None:
        """Imposta i confini della scena."""
        self._scene_bounds = bounds

    def image_to_ground(self, point: PixelPoint) -> Optional[GroundPoint]:
        """Proietta un punto dal piano immagine al piano del suolo.

        Implementa l'Algoritmo 5 della specifica:
        1. Costruisce vettore omogeneo [x, y, 1]^T
        2. Applica H: p' = H · p
        3. Normalizza: ground_x = p'[0]/p'[2], ground_y = p'[1]/p'[2]
        4. Controlla confini scena
        5. Restituisce GroundPoint o None

        Args:
            point: Punto in coordinate pixel nel piano immagine.

        Returns:
            GroundPoint con coordinate metriche nel piano del suolo,
            oppure None se la proiezione cade fuori dai confini della scena.

        Raises:
            ValueError: Se la matrice omografica non è stata calcolata.
        """
        if self._homography_matrix is None:
            raise ValueError(
                "Matrice omografica non disponibile. "
                "Eseguire la calibrazione prima della proiezione."
            )

        # 1. Costruire vettore omogeneo
        p = np.array([point.x, point.y, 1.0], dtype=np.float64)

        # 2. Applicare H
        p_prime = self._homography_matrix @ p

        # 3. Normalizzare (divisione per componente omogenea)
        if abs(p_prime[2]) < 1e-10:
            logger.warning(
                "Proiezione degenere: componente omogenea ~0 per punto (%f, %f)",
                point.x,
                point.y,
            )
            return None

        ground_x = p_prime[0] / p_prime[2]
        ground_y = p_prime[1] / p_prime[2]

        # 4. Controllare confini scena
        if self._scene_bounds is not None:
            x_min, y_min, x_max, y_max = self._scene_bounds
            if not (x_min <= ground_x <= x_max and y_min <= ground_y <= y_max):
                logger.warning(
                    "Anomalia di proiezione: punto (%f, %f) proiettato in "
                    "(%f, %f) fuori dai confini della scena [%f, %f, %f, %f]",
                    point.x,
                    point.y,
                    ground_x,
                    ground_y,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                )
                return None

        # 5. Restituire GroundPoint
        return GroundPoint(x=float(ground_x), y=float(ground_y))

    def validate_reprojection(
        self,
        calibration_points: list[CorrespondencePoint],
    ) -> ReprojectionResult:
        """Calcola l'errore di riproiezione medio sui punti di calibrazione.

        Per ogni coppia (pixel_point, ground_point):
            err_i = ||H · p_i - p'_i||

        Args:
            calibration_points: Lista di coppie (PixelPoint, GroundPoint)
                usate per la calibrazione.

        Returns:
            ReprojectionResult con errore medio, massimo, lista errori
            e flag di validità.

        Raises:
            ValueError: Se la matrice omografica non è disponibile o
                la lista è vuota.
        """
        if self._homography_matrix is None:
            raise ValueError(
                "Matrice omografica non disponibile per la validazione."
            )

        if not calibration_points:
            raise ValueError(
                "Fornire almeno un punto di corrispondenza per la validazione."
            )

        errors: list[float] = []

        for pixel_pt, ground_pt in calibration_points:
            # Proiettare il punto pixel usando H
            p = np.array([pixel_pt.x, pixel_pt.y, 1.0], dtype=np.float64)
            p_prime = self._homography_matrix @ p

            if abs(p_prime[2]) < 1e-10:
                # Proiezione degenere, errore massimo
                errors.append(float("inf"))
                continue

            projected_x = p_prime[0] / p_prime[2]
            projected_y = p_prime[1] / p_prime[2]

            # Calcolare distanza euclidea dal punto ground atteso
            err = np.sqrt(
                (projected_x - ground_pt.x) ** 2
                + (projected_y - ground_pt.y) ** 2
            )
            errors.append(float(err))

        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        is_valid = mean_error < self._reprojection_threshold

        return ReprojectionResult(
            mean_error=mean_error,
            max_error=max_error,
            errors=errors,
            is_valid=is_valid,
        )

    def estimate_ground_from_head(
        self, head_pos: PixelPoint, avg_height: float = 1.75
    ) -> Optional[GroundPoint]:
        """Stima la posizione a terra dalla posizione della testa.

        Fallback usato quando solo la testa è visibile (Req 7.4).
        Usa un modello geometrico semplificato: la testa è a avg_height
        metri sopra il suolo. Sposta il punto verso il basso nel piano
        immagine di un offset basato sull'altezza stimata, poi proietta
        con l'omografia.

        Args:
            head_pos: Posizione della testa in coordinate pixel.
            avg_height: Altezza media della persona in metri.
                Deve essere nel range [1.40, 2.10].

        Returns:
            GroundPoint stimato, o None se fuori confini.

        Raises:
            ValueError: Se avg_height è fuori dal range consentito o
                la matrice omografica non è disponibile.
        """
        if not (1.40 <= avg_height <= 2.10):
            raise ValueError(
                f"avg_height deve essere tra 1.40 e 2.10 metri, "
                f"ricevuto: {avg_height}"
            )

        if self._homography_matrix is None:
            raise ValueError(
                "Matrice omografica non disponibile. "
                "Eseguire la calibrazione prima della stima."
            )

        # Modello geometrico semplificato:
        # La testa è circa a avg_height dal suolo.
        # Usiamo l'inversa dell'omografia per stimare la scala locale,
        # poi proiettiamo la testa direttamente come approssimazione.
        #
        # Approccio: proiettiamo il punto testa con l'omografia.
        # Questo dà una stima ragionevole perché l'omografia mappa
        # il piano immagine al piano del suolo. Per punti che non sono
        # sul piano del suolo (come la testa), l'errore è proporzionale
        # all'altezza e all'angolo della camera.
        #
        # Per compensare, calcoliamo un offset verticale nel piano immagine
        # basato sulla scala locale dell'omografia nel punto.

        # Calcolare la scala locale dell'omografia nel punto testa
        # Usando la derivata dell'omografia per stimare pixel/metro
        h = self._homography_matrix
        p = np.array([head_pos.x, head_pos.y, 1.0], dtype=np.float64)
        p_prime = h @ p

        if abs(p_prime[2]) < 1e-10:
            logger.warning(
                "Proiezione degenere nella stima da testa per punto (%f, %f)",
                head_pos.x,
                head_pos.y,
            )
            return None

        # Stima della scala verticale locale (pixel per metro nel ground plane)
        # Calcoliamo come un piccolo spostamento verticale in pixel si traduce
        # in spostamento nel ground plane
        delta_px = 10.0  # piccolo offset in pixel
        p_shifted = np.array(
            [head_pos.x, head_pos.y + delta_px, 1.0], dtype=np.float64
        )
        p_shifted_prime = h @ p_shifted

        if abs(p_shifted_prime[2]) < 1e-10:
            # Fallback: proietta direttamente il punto testa
            return self.image_to_ground(head_pos)

        ground_shifted_x = p_shifted_prime[0] / p_shifted_prime[2]
        ground_shifted_y = p_shifted_prime[1] / p_shifted_prime[2]

        ground_x = p_prime[0] / p_prime[2]
        ground_y = p_prime[1] / p_prime[2]

        # Distanza in metri per delta_px pixel di spostamento verticale
        meters_per_delta = np.sqrt(
            (ground_shifted_x - ground_x) ** 2
            + (ground_shifted_y - ground_y) ** 2
        )

        if meters_per_delta < 1e-10:
            # Fallback: proietta direttamente
            return self.image_to_ground(head_pos)

        # Calcolare l'offset in pixel corrispondente ad avg_height
        # La testa è avg_height sopra il suolo → nel piano immagine,
        # il piede è più in basso (y maggiore) della testa
        pixels_per_meter = delta_px / meters_per_delta
        foot_offset_pixels = avg_height * pixels_per_meter

        # Creare il punto piede stimato (più in basso nel frame)
        estimated_foot = PixelPoint(
            x=head_pos.x, y=head_pos.y + foot_offset_pixels
        )

        # Proiettare il punto piede stimato sul piano del suolo
        return self.image_to_ground(estimated_foot)

    @classmethod
    def calibrate_manual(
        cls,
        correspondences: list[CorrespondencePoint],
        scene_bounds: Optional[SceneBounds] = None,
        reprojection_threshold: float = 5.0,
    ) -> "HomographyModule":
        """Calibrazione manuale con almeno 4 punti di corrispondenza.

        Implementa la calibrazione manuale (Req 12.4):
        1. Verifica >= 4 punti di corrispondenza
        2. Calcola H con cv2.findHomography (RANSAC)
        3. Calcola e valida errore di riproiezione medio
        4. Rifiuta se errore > soglia

        Args:
            correspondences: Lista di coppie (PixelPoint, GroundPoint),
                almeno 4 coppie richieste.
            scene_bounds: Confini della scena (x_min, y_min, x_max, y_max).
            reprojection_threshold: Soglia massima errore di riproiezione.

        Returns:
            HomographyModule configurato con la matrice calcolata.

        Raises:
            ValueError: Se < 4 punti forniti o errore di riproiezione
                supera la soglia.
        """
        # 1. Verificare almeno 4 punti
        if len(correspondences) < 4:
            raise ValueError(
                f"Servono almeno 4 punti di corrispondenza per la calibrazione, "
                f"ricevuti: {len(correspondences)}"
            )

        # Preparare array sorgente (pixel) e destinazione (ground)
        src_pts = np.array(
            [[pt[0].x, pt[0].y] for pt in correspondences], dtype=np.float64
        )
        dst_pts = np.array(
            [[pt[1].x, pt[1].y] for pt in correspondences], dtype=np.float64
        )

        # 2. Calcolare H con RANSAC
        homography_matrix, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if homography_matrix is None:
            raise ValueError(
                "Impossibile calcolare la matrice omografica. "
                "Verificare che i punti non siano collineari."
            )

        # 3. Calcolare errore di riproiezione medio
        # Proiettare tutti i punti sorgente usando H
        src_pts_homogeneous = np.hstack(
            [src_pts, np.ones((len(src_pts), 1), dtype=np.float64)]
        )
        projected = (homography_matrix @ src_pts_homogeneous.T).T

        # Normalizzare le coordinate omogenee
        projected_normalized = np.zeros((len(projected), 2), dtype=np.float64)
        for i in range(len(projected)):
            if abs(projected[i, 2]) < 1e-10:
                projected_normalized[i] = [float("inf"), float("inf")]
            else:
                projected_normalized[i, 0] = projected[i, 0] / projected[i, 2]
                projected_normalized[i, 1] = projected[i, 1] / projected[i, 2]

        # Calcolare errori
        errors = np.sqrt(
            np.sum((projected_normalized - dst_pts) ** 2, axis=1)
        )
        mean_error = float(np.mean(errors))

        # 4. Validare errore di riproiezione
        if mean_error > reprojection_threshold:
            raise ValueError(
                f"Errore di riproiezione medio ({mean_error:.4f}) supera la "
                f"soglia configurata ({reprojection_threshold:.4f}). "
                f"Calibrazione rifiutata."
            )

        logger.info(
            "Calibrazione manuale completata con successo. "
            "Errore di riproiezione medio: %.4f (soglia: %.4f)",
            mean_error,
            reprojection_threshold,
        )

        # Creare CameraCalibration con la matrice calcolata
        calibration = CameraCalibration(
            homography_matrix=homography_matrix,
            reprojection_error=mean_error,
        )

        return cls(
            calibration=calibration,
            scene_bounds=scene_bounds,
            reprojection_threshold=reprojection_threshold,
        )
