"""Test per il modulo detection - HeadDetector."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Aggiungi il path del progetto
sys.path.insert(0, str(Path(__file__).parent.parent))

from crowd_density.detection import HeadDetector
from crowd_density.models import BoundingBox, HeadDetection, PixelPoint


class TestComputeConfidence:
    """Test per il calcolo della confidenza piecewise."""

    def setup_method(self):
        """Setup con mock del modello YOLO per evitare download."""
        with patch("crowd_density.detection.HeadDetector.__init__", lambda self, *a, **kw: None):
            self.detector = HeadDetector()

    def test_visible_ratio_below_30_percent_returns_none(self):
        """visible_ratio < 0.30 → scartato (None)."""
        assert self.detector.compute_confidence(0.0) is None
        assert self.detector.compute_confidence(0.15) is None
        assert self.detector.compute_confidence(0.29) is None

    def test_visible_ratio_at_30_percent_returns_linear(self):
        """visible_ratio = 0.30 → confidenza = 0.30 (lineare)."""
        result = self.detector.compute_confidence(0.30)
        assert result == pytest.approx(0.30)

    def test_visible_ratio_in_linear_range(self):
        """30% <= visible_ratio < 90% → confidenza = visible_ratio."""
        assert self.detector.compute_confidence(0.50) == pytest.approx(0.50)
        assert self.detector.compute_confidence(0.70) == pytest.approx(0.70)
        assert self.detector.compute_confidence(0.89) == pytest.approx(0.89)

    def test_visible_ratio_at_90_percent_returns_one(self):
        """visible_ratio >= 0.90 → confidenza = 1.0."""
        assert self.detector.compute_confidence(0.90) == 1.0

    def test_visible_ratio_above_90_percent_returns_one(self):
        """visible_ratio > 0.90 → confidenza = 1.0."""
        assert self.detector.compute_confidence(0.95) == 1.0
        assert self.detector.compute_confidence(1.0) == 1.0


class TestComputeIoU:
    """Test per il calcolo dell'IoU tra bounding box."""

    def test_identical_boxes_iou_is_one(self):
        """Due bbox identiche → IoU = 1.0."""
        box = BoundingBox(x=10, y=10, width=50, height=50)
        assert HeadDetector._compute_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_zero(self):
        """Due bbox non sovrapposte → IoU = 0.0."""
        box1 = BoundingBox(x=0, y=0, width=10, height=10)
        box2 = BoundingBox(x=100, y=100, width=10, height=10)
        assert HeadDetector._compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Due bbox parzialmente sovrapposte → 0 < IoU < 1."""
        box1 = BoundingBox(x=0, y=0, width=20, height=20)
        box2 = BoundingBox(x=10, y=10, width=20, height=20)
        # Intersezione: 10x10 = 100, Unione: 400 + 400 - 100 = 700
        expected_iou = 100.0 / 700.0
        assert HeadDetector._compute_iou(box1, box2) == pytest.approx(expected_iou)

    def test_contained_box(self):
        """Una bbox contenuta nell'altra."""
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=25, y=25, width=50, height=50)
        # Intersezione: 50*50 = 2500, Unione: 10000 + 2500 - 2500 = 10000
        expected_iou = 2500.0 / 10000.0
        assert HeadDetector._compute_iou(box1, box2) == pytest.approx(expected_iou)


class TestNMS:
    """Test per la Non-Maximum Suppression."""

    def setup_method(self):
        with patch("crowd_density.detection.HeadDetector.__init__", lambda self, *a, **kw: None):
            self.detector = HeadDetector()

    def test_nms_suppresses_overlapping_low_confidence(self):
        """NMS mantiene la detection con confidenza più alta tra overlapping."""
        det_high = HeadDetection(
            bbox=BoundingBox(x=10, y=10, width=30, height=30),
            confidence=0.9,
            center=PixelPoint(x=25.0, y=25.0),
            visible_ratio=0.9,
        )
        det_low = HeadDetection(
            bbox=BoundingBox(x=12, y=12, width=30, height=30),
            confidence=0.5,
            center=PixelPoint(x=27.0, y=27.0),
            visible_ratio=0.5,
        )
        result = self.detector._apply_nms([det_high, det_low])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_nms_keeps_non_overlapping(self):
        """NMS mantiene tutte le detection non sovrapposte."""
        det1 = HeadDetection(
            bbox=BoundingBox(x=0, y=0, width=20, height=20),
            confidence=0.8,
            center=PixelPoint(x=10.0, y=10.0),
            visible_ratio=0.8,
        )
        det2 = HeadDetection(
            bbox=BoundingBox(x=200, y=200, width=20, height=20),
            confidence=0.7,
            center=PixelPoint(x=210.0, y=210.0),
            visible_ratio=0.7,
        )
        result = self.detector._apply_nms([det1, det2])
        assert len(result) == 2

    def test_nms_empty_list(self):
        """NMS su lista vuota restituisce lista vuota."""
        result = self.detector._apply_nms([])
        assert result == []


class TestProcessDetection:
    """Test per l'elaborazione di una singola detection."""

    def setup_method(self):
        with patch("crowd_density.detection.HeadDetector.__init__", lambda self, *a, **kw: None):
            self.detector = HeadDetector()

    def test_small_head_filtered_out(self):
        """Detection con testa troppo piccola (< 16x16) viene scartata."""
        # Person bbox troppo piccola: larghezza 10px, altezza 50px
        # head_width = 10, head_height = int(50 * 0.22) = 11 < 16
        detection = {
            "bbox": (100.0, 100.0, 110.0, 150.0),
            "confidence": 0.8,
            "class_id": 0,
        }
        result = self.detector._process_detection(detection)
        assert result is None

    def test_low_confidence_filtered_out(self):
        """Detection con visible_ratio < 0.30 viene scartata."""
        # Person bbox grande ma confidenza YOLO bassa (0.25 < 0.30)
        detection = {
            "bbox": (0.0, 0.0, 200.0, 400.0),
            "confidence": 0.25,
            "class_id": 0,
        }
        result = self.detector._process_detection(detection)
        assert result is None

    def test_valid_detection_produces_head(self):
        """Detection valida produce un HeadDetection corretto."""
        # Person bbox: 200x400 → head_width=200, head_height=int(400*0.22)=88
        detection = {
            "bbox": (50.0, 50.0, 250.0, 450.0),
            "confidence": 0.85,
            "class_id": 0,
        }
        result = self.detector._process_detection(detection)
        assert result is not None
        assert result.bbox.x == 50
        assert result.bbox.y == 50
        assert result.bbox.width == 200
        assert result.bbox.height == 88  # int(400 * 0.22)
        assert result.confidence == pytest.approx(0.85)
        assert result.visible_ratio == pytest.approx(0.85)

    def test_high_confidence_capped_at_one(self):
        """Confidenza YOLO >= 0.90 produce confidence = 1.0."""
        detection = {
            "bbox": (0.0, 0.0, 200.0, 400.0),
            "confidence": 0.95,
            "class_id": 0,
        }
        result = self.detector._process_detection(detection)
        assert result is not None
        assert result.confidence == 1.0


class TestDetect:
    """Test per il metodo detect principale."""

    def setup_method(self):
        with patch("crowd_density.detection.HeadDetector.__init__", lambda self, *a, **kw: None):
            self.detector = HeadDetector()
            self.detector._model = MagicMock()
            self.detector._confidence_threshold = 0.25

    def test_detect_empty_frame_returns_empty(self):
        """Frame vuoto (None) restituisce lista vuota."""
        result = self.detector.detect(None)
        assert result == []

    def test_detect_empty_array_returns_empty(self):
        """Frame con size 0 restituisce lista vuota."""
        result = self.detector.detect(np.array([]))
        assert result == []

    def test_detect_with_mocked_inference(self):
        """Detect con inferenza mockata produce risultati corretti."""
        # Mock dell'inferenza per restituire una detection persona valida
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.cls = [MagicMock(item=lambda: 0)]  # classe person
        mock_boxes.conf = [MagicMock(item=lambda: 0.85)]
        mock_boxes.xyxy = [
            MagicMock(
                cpu=lambda: MagicMock(
                    numpy=lambda: np.array([50.0, 50.0, 250.0, 450.0])
                )
            )
        ]
        mock_boxes.__len__ = lambda _: 1
        mock_result.boxes = mock_boxes

        self.detector._model.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.85)
        assert result[0].bbox.width == 200
