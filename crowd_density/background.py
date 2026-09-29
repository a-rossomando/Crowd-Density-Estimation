"""Filtro background subtraction per escludere detection statiche.

Costruisce un modello di sfondo dalla mediana dei primi N frame,
poi per ogni detection verifica se la regione corrispondente nel frame
corrente è significativamente diversa dallo sfondo. Le detection la cui
area non mostra variazione (es. statue, mobili) vengono scartate.

Questo modulo è progettato per scene con camera fissa e oggetti statici
a forma umana (statue in un museo) che ingannano il person detector.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from .models import HeadDetection

logger = logging.getLogger(__name__)


class BackgroundFilter:
    """Filtra detection statiche usando un modello di sfondo.

    Il modello di sfondo è la mediana pixel-per-pixel dei primi N frame,
    convertiti in scala di grigi. Per ogni detection, si estrae la regione
    corrispondente (bounding box della persona intera, non solo la testa)
    e si confronta con lo sfondo: se la differenza media è sotto una soglia,
    la detection è considerata statica e viene scartata.

    Attributes:
        warmup_frames: Numero di frame per costruire il modello di sfondo.
        diff_threshold: Soglia sulla differenza media (0-255) per considerare
            una regione come dinamica (persona vera).
        static_ratio: Se la percentuale di pixel con differenza > diff_threshold
            è sotto questo valore, la detection è statica.
    """

    def __init__(
        self,
        warmup_frames: int = 60,
        diff_threshold: float = 30.0,
        static_ratio: float = 0.15,
    ) -> None:
        """Inizializza il filtro.

        Args:
            warmup_frames: Quanti frame usare per costruire lo sfondo.
                Default 60 (~2 secondi a 30fps).
            diff_threshold: Soglia di differenza per pixel (0-255).
                Pixel con |frame - bg| > threshold sono considerati "cambiati".
            static_ratio: Proporzione minima di pixel cambiati nella bbox
                per considerare la detection come dinamica (persona vera).
                Se sotto questa soglia → statica (statua).
        """
        self._warmup_frames = warmup_frames
        self._diff_threshold = diff_threshold
        self._static_ratio = static_ratio

        self._frame_buffer: list[np.ndarray] = []
        self._background: Optional[np.ndarray] = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """True se il modello di sfondo è stato costruito."""
        return self._is_ready

    @property
    def warmup_frames(self) -> int:
        """Numero di frame necessari per il warmup."""
        return self._warmup_frames

    @property
    def background(self) -> Optional[np.ndarray]:
        """Immagine di sfondo (scala di grigi), o None se non pronta."""
        return self._background

    @classmethod
    def from_image(
        cls,
        image_path: str,
        diff_threshold: float = 30.0,
        static_ratio: float = 0.15,
    ) -> "BackgroundFilter":
        """Crea un BackgroundFilter con un modello di sfondo pre-calcolato.

        Salta completamente il warmup: il filtro è pronto immediatamente.

        Args:
            image_path: Percorso all'immagine di sfondo (PNG, JPG).
            diff_threshold: Soglia di differenza per pixel (0-255).
            static_ratio: Proporzione minima di pixel cambiati.

        Returns:
            BackgroundFilter pronto all'uso.

        Raises:
            FileNotFoundError: Se l'immagine non esiste.
            ValueError: Se l'immagine non è leggibile.
        """
        bg = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if bg is None:
            raise ValueError(f"Impossibile leggere l'immagine di sfondo: {image_path}")

        instance = cls(
            warmup_frames=0,
            diff_threshold=diff_threshold,
            static_ratio=static_ratio,
        )
        instance._background = bg
        instance._is_ready = True
        logger.info("Modello di sfondo caricato da: %s (%dx%d)",
                     image_path, bg.shape[1], bg.shape[0])
        return instance

    def feed_frame(self, frame: np.ndarray) -> bool:
        """Alimenta un frame al buffer di warmup.

        Durante il warmup, i frame vengono accumulati per costruire
        il modello di sfondo. Una volta raggiunti warmup_frames, il
        modello viene calcolato automaticamente.

        Args:
            frame: Frame RGB con shape (H, W, 3) e dtype uint8.

        Returns:
            True se il modello è pronto (warmup completato), False altrimenti.
        """
        if self._is_ready:
            return True

        # Converti a scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self._frame_buffer.append(gray)

        if len(self._frame_buffer) >= self._warmup_frames:
            self._build_background()
            return True

        return False

    def _build_background(self) -> None:
        """Costruisce il modello di sfondo come mediana dei frame accumulati."""
        logger.info(
            "Costruzione modello di sfondo da %d frame...",
            len(self._frame_buffer),
        )

        # Stack dei frame e calcolo della mediana
        stack = np.stack(self._frame_buffer, axis=0)  # (N, H, W)
        self._background = np.median(stack, axis=0).astype(np.uint8)

        # Libera la memoria del buffer
        self._frame_buffer.clear()
        self._is_ready = True

        # Salva l'immagine di sfondo per ispezione
        cv2.imwrite("background_model.png", self._background)
        logger.info("Modello di sfondo costruito e salvato in background_model.png")

    def filter_detections(
        self,
        frame: np.ndarray,
        detections: list[HeadDetection],
    ) -> list[HeadDetection]:
        """Filtra le detection, mantenendo solo quelle in regioni dinamiche.

        Per ogni detection, confronta la regione della bounding box nel
        frame corrente con la stessa regione nello sfondo. Se la differenza
        è troppo bassa (regione statica), la detection viene scartata.

        Args:
            frame: Frame RGB corrente con shape (H, W, 3) e dtype uint8.
            detections: Lista delle HeadDetection da filtrare.

        Returns:
            Lista delle detection in regioni dinamiche (persone vere).
            Se il modello non è pronto, restituisce la lista vuota
            (durante il warmup non si rileva nulla).
        """
        if not self._is_ready or self._background is None:
            # Durante il warmup non rilasciamo detection per evitare
            # che le statue vengano tracciate prima del filtro
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        dynamic_detections: list[HeadDetection] = []
        for det in detections:
            if self._is_dynamic(gray, det):
                dynamic_detections.append(det)

        filtered_count = len(detections) - len(dynamic_detections)
        if filtered_count > 0:
            logger.debug(
                "Background filter: %d/%d detection scartate (statiche)",
                filtered_count,
                len(detections),
            )

        return dynamic_detections

    def _is_dynamic(self, gray_frame: np.ndarray, detection: HeadDetection) -> bool:
        """Verifica se una detection corrisponde a un oggetto dinamico.

        Confronta la regione della bbox nel frame corrente con lo sfondo.
        Usa la bbox della testa ma la espande verso il basso per coprire
        una porzione più ampia del corpo (la testa da sola potrebbe
        essere troppo piccola per un confronto affidabile).

        Args:
            gray_frame: Frame corrente in scala di grigi.
            detection: HeadDetection da verificare.

        Returns:
            True se la regione è dinamica (persona vera), False se statica.
        """
        h, w = gray_frame.shape[:2]
        bbox = detection.bbox

        # Espandi la bbox della testa verso il basso per coprire il corpo
        # La testa è il 22% superiore → il corpo si estende per ~3.5x
        body_height = int(bbox.height * 4.5)
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(w, bbox.x + bbox.width)
        y2 = min(h, bbox.y + body_height)

        if x2 <= x1 or y2 <= y1:
            return True  # bbox degenere, non filtrare

        # Estrai le regioni
        roi_frame = gray_frame[y1:y2, x1:x2]
        roi_bg = self._background[y1:y2, x1:x2]

        # Calcola la differenza assoluta
        diff = cv2.absdiff(roi_frame, roi_bg)

        # Conta i pixel che superano la soglia di differenza
        changed_pixels = np.count_nonzero(diff > self._diff_threshold)
        total_pixels = diff.size

        if total_pixels == 0:
            return True

        change_ratio = changed_pixels / total_pixels

        return change_ratio >= self._static_ratio
