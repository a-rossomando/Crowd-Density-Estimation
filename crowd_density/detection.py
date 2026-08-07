"""Modulo di rilevamento teste con YOLOv8.

Implementa il rilevamento delle teste nel frame corrente utilizzando YOLOv8n
pre-trained su COCO. Poiché YOLOv8n non ha una classe "head" specifica,
si utilizza la detection di "person" (classe 0) e si estrae la porzione
superiore della bounding box come approssimazione della regione testa.
"""

import logging
from typing import Optional

import numpy as np

from .models import BoundingBox, HeadDetection, PixelPoint

logger = logging.getLogger(__name__)


class HeadDetector:
    """Rilevatore di teste basato su YOLOv8n.

    Utilizza YOLOv8n pre-trained su COCO per rilevare persone (classe 0),
    poi estrae la porzione superiore del bounding box come approssimazione
    della regione testa.

    Attributes:
        MIN_HEAD_SIZE: Dimensione minima della testa in pixel (larghezza, altezza).
        HEAD_RATIO: Proporzione superiore della bounding box persona usata come testa.
        NMS_IOU_THRESHOLD: Soglia IoU per Non-Maximum Suppression sulle head box.
        PERSON_CLASS_ID: ID della classe "person" in COCO.
    """

    MIN_HEAD_SIZE: tuple[int, int] = (16, 16)
    HEAD_RATIO: float = 0.22  # top 22% della bbox persona ≈ regione testa
    NMS_IOU_THRESHOLD: float = 0.5
    PERSON_CLASS_ID: int = 0

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.25):
        """Inizializza il detector caricando il modello YOLOv8n.

        Args:
            model_path: Percorso al modello YOLOv8n. Default: "yolov8n.pt"
                (scaricato automaticamente se non presente).
            confidence_threshold: Soglia minima di confidenza YOLO per filtrare
                le detection grezze. Default: 0.25.
        """
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[HeadDetection]:
        """Rileva le teste nel frame corrente.

        Esegue inferenza YOLOv8 sul frame, filtra per classe "person",
        estrae la regione testa dalla porzione superiore della bbox,
        applica filtro area minima, calcola la confidenza piecewise
        e infine applica NMS con IoU threshold 0.5.

        Args:
            frame: Frame RGB con shape (H, W, 3) e dtype uint8.

        Returns:
            Lista di HeadDetection con bounding box, confidenza, centro
            e visible_ratio per ogni testa rilevata.
        """
        if frame is None or frame.size == 0:
            return []

        # 1. Inferenza YOLOv8
        raw_detections = self._run_inference(frame)

        # 2. Estrai head boxes e calcola confidenza
        head_detections = []
        for det in raw_detections:
            head_det = self._process_detection(det)
            if head_det is not None:
                head_detections.append(head_det)

        # 3. Applica NMS sulle head boxes
        if len(head_detections) > 1:
            head_detections = self._apply_nms(head_detections)

        return head_detections

    def compute_confidence(self, visible_ratio: float) -> Optional[float]:
        """Calcola la confidenza piecewise dal visible_ratio.

        Regole:
            - visible_ratio < 0.30 → None (scartato)
            - 0.30 <= visible_ratio < 0.90 → visible_ratio (lineare)
            - visible_ratio >= 0.90 → 1.0

        Args:
            visible_ratio: Rapporto tra area visibile e area totale [0.0, 1.0].

        Returns:
            Confidenza calcolata nel range [0.3, 1.0], oppure None se scartato.
        """
        if visible_ratio < 0.30:
            return None
        elif visible_ratio >= 0.90:
            return 1.0
        else:
            return visible_ratio

    def _run_inference(self, frame: np.ndarray) -> list[dict]:
        """Esegue inferenza YOLOv8 e restituisce detection grezze per classe person.

        Args:
            frame: Frame RGB (H, W, 3).

        Returns:
            Lista di dizionari con chiavi: 'bbox' (x1, y1, x2, y2),
            'confidence' (float), 'class_id' (int).
        """
        results = self._model(
            frame,
            conf=self._confidence_threshold,
            verbose=False,
        )

        raw_detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                if class_id != self.PERSON_CLASS_ID:
                    continue
                confidence = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                raw_detections.append({
                    "bbox": (
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                    ),
                    "confidence": confidence,
                    "class_id": class_id,
                })

        return raw_detections

    def _process_detection(self, detection: dict) -> Optional[HeadDetection]:
        """Processa una singola detection e produce un HeadDetection.

        Estrae la regione testa dalla porzione superiore della bbox persona,
        applica il filtro area minima e calcola la confidenza piecewise.

        Args:
            detection: Dizionario con 'bbox' (x1, y1, x2, y2) e 'confidence'.

        Returns:
            HeadDetection se valido, None se scartato.
        """
        x1, y1, x2, y2 = detection["bbox"]
        person_height = y2 - y1
        person_width = x2 - x1

        # Estrai porzione superiore come approssimazione testa
        head_height = int(person_height * self.HEAD_RATIO)
        head_width = int(person_width)
        head_x = int(x1)
        head_y = int(y1)

        # a. Filtro area minima 16x16 pixel
        if head_width < self.MIN_HEAD_SIZE[0] or head_height < self.MIN_HEAD_SIZE[1]:
            return None

        # Per MVP: visible_ratio derivato dalla confidenza YOLO
        # Mappa confidenza YOLO [0.25, 1.0] → visible_ratio [0.0, 1.0]
        yolo_conf = detection["confidence"]
        visible_ratio = yolo_conf

        # b. Calcolo confidenza piecewise
        confidence = self.compute_confidence(visible_ratio)
        if confidence is None:
            return None

        # Costruisci HeadDetection
        bbox = BoundingBox(
            x=head_x,
            y=head_y,
            width=head_width,
            height=head_height,
        )
        center = PixelPoint(
            x=head_x + head_width / 2.0,
            y=head_y + head_height / 2.0,
        )

        return HeadDetection(
            bbox=bbox,
            confidence=confidence,
            center=center,
            visible_ratio=visible_ratio,
        )

    def _apply_nms(self, detections: list[HeadDetection]) -> list[HeadDetection]:
        """Applica Non-Maximum Suppression sulle head detections.

        Utilizza IoU threshold di 0.5. Tra due box sovrapposte, mantiene
        quella con confidenza maggiore.

        Args:
            detections: Lista di HeadDetection da filtrare.

        Returns:
            Lista filtrata dopo NMS.
        """
        if not detections:
            return []

        # Ordina per confidenza decrescente
        detections_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        suppressed = set()

        for i, det_i in enumerate(detections_sorted):
            if i in suppressed:
                continue
            keep.append(det_i)
            for j in range(i + 1, len(detections_sorted)):
                if j in suppressed:
                    continue
                iou = self._compute_iou(det_i.bbox, detections_sorted[j].bbox)
                if iou >= self.NMS_IOU_THRESHOLD:
                    suppressed.add(j)

        return keep

    @staticmethod
    def _compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calcola l'Intersection over Union tra due bounding box.

        Args:
            box1: Prima bounding box.
            box2: Seconda bounding box.

        Returns:
            IoU nel range [0.0, 1.0].
        """
        # Coordinate angoli
        x1_1, y1_1 = box1.x, box1.y
        x2_1, y2_1 = box1.x + box1.width, box1.y + box1.height

        x1_2, y1_2 = box2.x, box2.y
        x2_2, y2_2 = box2.x + box2.width, box2.y + box2.height

        # Intersezione
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        # Unione
        area1 = box1.area
        area2 = box2.area
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
