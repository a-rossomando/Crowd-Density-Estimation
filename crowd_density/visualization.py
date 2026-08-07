"""Modulo di visualizzazione con overlay OpenCV.

Genera overlay grafici sui frame video per mostrare:
- Bounding box delle teste con colore per stato (visibile/parziale/occluso)
- Conteggio totale e intervallo di confidenza
- Mini-mappa densità sovrapposta in un angolo
- Allarme assembramento con rettangolo rosso lampeggiante

Requirements: 7.2, 8.2, 8.4
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .density import DensityMap
from .models import (
    BoundingBox,
    CrowdAlarm,
    PersonState,
    TrackedPerson,
)
from .pipeline import FrameResult


# Colori BGR per stato persona
_COLOR_VISIBILE = (0, 255, 0)  # Verde
_COLOR_PARZIALE = (0, 255, 255)  # Giallo
_COLOR_OCCLUSO = (0, 0, 255)  # Rosso
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)
_COLOR_ALARM_RED = (0, 0, 255)

# Font
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_FONT_THICKNESS = 2
_FONT_SCALE_SMALL = 0.5
_FONT_THICKNESS_SMALL = 1


@dataclass
class VisualizerConfig:
    """Configurazione per il Visualizer.

    Attributes:
        minimap_size: Dimensione della mini-mappa densità (width, height).
        minimap_alpha: Trasparenza della mini-mappa sovrapposta [0.0, 1.0].
        minimap_margin: Margine in pixel dal bordo del frame.
        alarm_border_thickness: Spessore del bordo dell'allarme in pixel.
        show_person_id: Se mostrare l'ID persona sopra il bounding box.
    """

    minimap_size: Tuple[int, int] = (200, 200)
    minimap_alpha: float = 0.6
    minimap_margin: int = 10
    alarm_border_thickness: int = 4
    show_person_id: bool = True


class Visualizer:
    """Genera overlay OpenCV per la visualizzazione dei risultati della pipeline.

    Disegna bounding box colorati per stato persona, conteggio con intervallo
    di confidenza, mini-mappa densità e allarmi assembramento.

    Args:
        config: Configurazione del visualizzatore. Se None, usa valori di default.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        self._config = config or VisualizerConfig()

    @property
    def config(self) -> VisualizerConfig:
        """Configurazione corrente del visualizzatore."""
        return self._config

    def draw_overlay(self, frame: np.ndarray, frame_result: FrameResult) -> np.ndarray:
        """Disegna l'overlay completo su un frame video.

        Overlay include:
        - Bounding box teste colorati per stato (Req 7.2)
        - Conteggio totale e intervallo di confidenza in alto a sinistra
        - Mini-mappa densità in basso a destra
        - Allarmi assembramento con rettangolo rosso lampeggiante (Req 8.2, 8.4)

        Args:
            frame: Frame BGR con shape (H, W, 3) e dtype uint8.
            frame_result: Risultato della pipeline per il frame corrente.

        Returns:
            Frame con overlay disegnato (copia del frame originale).
        """
        output = frame.copy()

        # 1. Disegna bounding box per persone tracciate
        output = self._draw_tracked_persons(output, frame_result.tracked_persons)

        # 2. Disegna conteggio in alto a sinistra
        output = self._draw_count_info(output, frame_result)

        # 3. Disegna mini-mappa densità in basso a destra
        if frame_result.density_map is not None:
            h, w = output.shape[:2]
            mw, mh = self._config.minimap_size
            margin = self._config.minimap_margin
            position = (w - mw - margin, h - mh - margin)
            output = self.draw_density_minimap(
                output,
                frame_result.density_map,
                position=position,
                size=self._config.minimap_size,
            )

        # 4. Disegna allarmi
        for alarm in frame_result.alarms:
            output = self.draw_alarm(output, alarm, frame_result.frame_id)

        return output

    def _draw_tracked_persons(
        self, frame: np.ndarray, tracked_persons: list[TrackedPerson]
    ) -> np.ndarray:
        """Disegna bounding box delle teste colorate per stato persona.

        Req 7.2: Distinguere visivamente persone rilevate vs occluse.

        Args:
            frame: Frame BGR su cui disegnare.
            tracked_persons: Lista delle persone tracciate.

        Returns:
            Frame con bounding box disegnati.
        """
        for person in tracked_persons:
            color = self._get_color_for_state(person.state)

            # Se la persona ha un ultimo rilevamento con bounding box
            if person.last_detection is not None:
                bbox = person.last_detection.head.bbox
                self._draw_bbox(frame, bbox, color, person)
            elif person.state == PersonState.OCCLUSO:
                # Persona occlusa senza rilevamento: disegna un marker alla
                # posizione prevista (se disponibile come pixel approssimativo)
                # Usiamo un cerchio tratteggiato per indicare predizione
                if person.last_detection is not None:
                    bbox = person.last_detection.head.bbox
                    self._draw_bbox(frame, bbox, color, person, dashed=True)

        return frame

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
        person: TrackedPerson,
        dashed: bool = False,
    ) -> None:
        """Disegna un singolo bounding box con etichetta opzionale.

        Args:
            frame: Frame BGR su cui disegnare.
            bbox: Bounding box da disegnare.
            color: Colore BGR del rettangolo.
            person: Persona associata (per ID e stato).
            dashed: Se True, disegna un rettangolo tratteggiato (occluso predetto).
        """
        pt1 = (bbox.x, bbox.y)
        pt2 = (bbox.x + bbox.width, bbox.y + bbox.height)

        if dashed:
            # Rettangolo tratteggiato per posizioni predette
            self._draw_dashed_rect(frame, pt1, pt2, color, thickness=2, dash_length=8)
        else:
            cv2.rectangle(frame, pt1, pt2, color, thickness=2)

        # Etichetta con ID persona
        if self._config.show_person_id:
            label = f"ID:{person.person_id}"
            label_pos = (bbox.x, bbox.y - 5)
            cv2.putText(
                frame,
                label,
                label_pos,
                _FONT,
                _FONT_SCALE_SMALL,
                color,
                _FONT_THICKNESS_SMALL,
                cv2.LINE_AA,
            )

    def _draw_dashed_rect(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 8,
    ) -> None:
        """Disegna un rettangolo tratteggiato.

        Args:
            frame: Frame BGR su cui disegnare.
            pt1: Angolo superiore sinistro (x, y).
            pt2: Angolo inferiore destro (x, y).
            color: Colore BGR.
            thickness: Spessore linea.
            dash_length: Lunghezza di ogni trattino in pixel.
        """
        x1, y1 = pt1
        x2, y2 = pt2

        # Lati del rettangolo
        edges = [
            ((x1, y1), (x2, y1)),  # top
            ((x2, y1), (x2, y2)),  # right
            ((x2, y2), (x1, y2)),  # bottom
            ((x1, y2), (x1, y1)),  # left
        ]

        for start, end in edges:
            self._draw_dashed_line(frame, start, end, color, thickness, dash_length)

    def _draw_dashed_line(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 8,
    ) -> None:
        """Disegna una linea tratteggiata tra due punti.

        Args:
            frame: Frame BGR.
            pt1: Punto iniziale.
            pt2: Punto finale.
            color: Colore BGR.
            thickness: Spessore.
            dash_length: Lunghezza trattino.
        """
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist == 0:
            return

        num_dashes = int(dist / dash_length)
        for i in range(0, num_dashes, 2):
            t_start = i / num_dashes
            t_end = min((i + 1) / num_dashes, 1.0)
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * t_start),
                int(pt1[1] + (pt2[1] - pt1[1]) * t_start),
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * t_end),
                int(pt1[1] + (pt2[1] - pt1[1]) * t_end),
            )
            cv2.line(frame, start, end, color, thickness)

    def _draw_count_info(
        self, frame: np.ndarray, frame_result: FrameResult
    ) -> np.ndarray:
        """Disegna le informazioni di conteggio in alto a sinistra.

        Mostra:
        - "Persone: X.X [min, max]"
        - "Occluse: X.X"

        Args:
            frame: Frame BGR su cui disegnare.
            frame_result: Risultato del frame con count_result.

        Returns:
            Frame con testo del conteggio.
        """
        count = frame_result.count_result
        ci_min, ci_max = count.confidence_interval

        # Riga 1: conteggio totale con intervallo di confidenza
        text_total = (
            f"Persone: {count.total_count:.1f} [{ci_min:.1f}, {ci_max:.1f}]"
        )
        # Riga 2: conteggio occluse
        text_occluded = f"Occluse: {count.occluded_count:.1f}"

        # Background semi-trasparente per leggibilità
        y_offset = 30
        lines = [text_total, text_occluded]
        max_width = 0
        line_height = 30

        for line in lines:
            (text_w, _), _ = cv2.getTextSize(line, _FONT, _FONT_SCALE, _FONT_THICKNESS)
            max_width = max(max_width, text_w)

        # Disegna sfondo
        bg_pt1 = (5, 5)
        bg_pt2 = (max_width + 15, 5 + len(lines) * line_height + 10)
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_pt1, bg_pt2, _COLOR_BLACK, cv2.FILLED)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Disegna testo
        for i, line in enumerate(lines):
            pos = (10, y_offset + i * line_height)
            cv2.putText(
                frame,
                line,
                pos,
                _FONT,
                _FONT_SCALE,
                _COLOR_WHITE,
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return frame

    def draw_density_minimap(
        self,
        frame: np.ndarray,
        density_map: DensityMap,
        position: Tuple[int, int] = (0, 0),
        size: Tuple[int, int] = (200, 200),
    ) -> np.ndarray:
        """Disegna la mini-mappa di densità sovrapposta al frame.

        Converte la griglia di densità in una colormap (JET) e la
        sovrappone con trasparenza nella posizione specificata.

        Args:
            frame: Frame BGR su cui sovrapporre la mappa.
            density_map: Mappa di densità generata dal DensityMapGenerator.
            position: Posizione (x, y) dell'angolo superiore sinistro della minimap.
            size: Dimensione (width, height) della minimap in pixel.

        Returns:
            Frame con minimap sovrapposta.
        """
        grid = density_map.grid

        # Normalizza la griglia a [0, 255] per la colormap
        grid_max = grid.max()
        if grid_max > 0:
            normalized = (grid / grid_max * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(grid, dtype=np.uint8)

        # Applica colormap JET
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Ridimensiona alla dimensione desiderata
        minimap = cv2.resize(colored, size, interpolation=cv2.INTER_LINEAR)

        # Aggiungi bordo bianco
        cv2.rectangle(minimap, (0, 0), (size[0] - 1, size[1] - 1), _COLOR_WHITE, 1)

        # Sovrapponi con trasparenza
        x, y = position
        h, w = frame.shape[:2]
        mw, mh = size

        # Clipping ai bordi del frame
        x_end = min(x + mw, w)
        y_end = min(y + mh, h)
        x_start = max(x, 0)
        y_start = max(y, 0)

        if x_start >= x_end or y_start >= y_end:
            return frame

        # Regione del minimap da usare (in caso di clipping)
        mx_start = x_start - x
        my_start = y_start - y
        mx_end = mx_start + (x_end - x_start)
        my_end = my_start + (y_end - y_start)

        roi = frame[y_start:y_end, x_start:x_end]
        minimap_roi = minimap[my_start:my_end, mx_start:mx_end]

        alpha = self._config.minimap_alpha
        blended = cv2.addWeighted(minimap_roi, alpha, roi, 1.0 - alpha, 0)
        frame[y_start:y_end, x_start:x_end] = blended

        # Etichetta "Densità"
        label_pos = (x_start + 5, y_start + 15)
        cv2.putText(
            frame,
            "Densita'",
            label_pos,
            _FONT,
            _FONT_SCALE_SMALL,
            _COLOR_WHITE,
            _FONT_THICKNESS_SMALL,
            cv2.LINE_AA,
        )

        return frame

    def draw_alarm(
        self,
        frame: np.ndarray,
        alarm: CrowdAlarm,
        frame_id: int = 0,
    ) -> np.ndarray:
        """Disegna l'allarme assembramento con bordo rosso lampeggiante.

        Req 8.2: Segnalare posizione, area, numero persone di un assembramento.
        Req 8.4: Allarme critico distinto con posizione e livello densità.

        Il bordo lampeggia usando frame_id % 10 < 5 per alternare
        visibilità. Per allarmi critici il bordo è più spesso.

        Args:
            frame: Frame BGR su cui disegnare.
            alarm: Allarme da visualizzare.
            frame_id: ID del frame corrente (usato per il lampeggiamento).

        Returns:
            Frame con allarme disegnato.
        """
        # Effetto lampeggiamento: visibile per 5 frame, invisibile per 5
        blink_on = (frame_id % 10) < 5

        if not blink_on:
            return frame

        h, w = frame.shape[:2]

        # Spessore bordo: più spesso per allarmi critici
        thickness = self._config.alarm_border_thickness
        if alarm.alarm_level == "critical":
            thickness *= 2

        # Disegna bordo rosso attorno all'intero frame
        cv2.rectangle(
            frame,
            (0, 0),
            (w - 1, h - 1),
            _COLOR_ALARM_RED,
            thickness,
        )

        # Testo allarme in alto al centro
        cluster = alarm.cluster
        if alarm.alarm_level == "critical":
            alarm_text = (
                f"ALLARME CRITICO: {cluster.person_count} persone, "
                f"densita' {cluster.density:.1f} p/m2"
            )
        else:
            alarm_text = (
                f"ATTENZIONE: {cluster.person_count} persone, "
                f"area {cluster.area_m2:.1f} m2"
            )

        # Calcola posizione centrata
        (text_w, text_h), _ = cv2.getTextSize(
            alarm_text, _FONT, _FONT_SCALE, _FONT_THICKNESS
        )
        text_x = (w - text_w) // 2
        text_y = thickness + text_h + 10

        # Background per il testo dell'allarme
        bg_pt1 = (text_x - 5, text_y - text_h - 5)
        bg_pt2 = (text_x + text_w + 5, text_y + 5)
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_pt1, bg_pt2, _COLOR_BLACK, cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Testo allarme
        cv2.putText(
            frame,
            alarm_text,
            (text_x, text_y),
            _FONT,
            _FONT_SCALE,
            _COLOR_ALARM_RED,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

        return frame

    @staticmethod
    def _get_color_for_state(state: PersonState) -> Tuple[int, int, int]:
        """Restituisce il colore BGR associato allo stato della persona.

        Args:
            state: Stato della persona.

        Returns:
            Tupla BGR del colore.
        """
        if state == PersonState.VISIBILE:
            return _COLOR_VISIBILE
        elif state == PersonState.PARZIALMENTE_OCCLUSO:
            return _COLOR_PARZIALE
        elif state == PersonState.OCCLUSO:
            return _COLOR_OCCLUSO
        else:
            # SCOMPARSO o altro: grigio
            return (128, 128, 128)

    @staticmethod
    def show_frame(
        frame: np.ndarray,
        window_name: str = "Crowd Density",
        wait_ms: int = 1,
    ) -> bool:
        """Mostra un frame con cv2.imshow.

        Args:
            frame: Frame BGR da mostrare.
            window_name: Nome della finestra OpenCV.
            wait_ms: Millisecondi di attesa per cv2.waitKey.

        Returns:
            True se l'utente ha premuto 'q' o ESC (per uscire), False altrimenti.
        """
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        return key == ord("q") or key == 27  # 'q' o ESC

    @staticmethod
    def save_video(
        frames: list[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> None:
        """Salva una sequenza di frame come video.

        Args:
            frames: Lista di frame BGR con stesse dimensioni.
            output_path: Percorso del file video di output.
            fps: Frame per secondo del video di output.
            codec: Codice FourCC del codec (default: mp4v).

        Raises:
            ValueError: Se la lista di frame è vuota.
            RuntimeError: Se il VideoWriter non riesce ad aprirsi.
        """
        if not frames:
            raise ValueError("La lista di frame è vuota.")

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(
                f"Impossibile aprire il VideoWriter per: {output_path}"
            )

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
