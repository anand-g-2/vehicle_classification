"""
Unit tests for VehicleDetector. The YOLO model is mocked out entirely —
no real weights are loaded, so these run fast and deterministically.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.vehicle_detector import DEFAULT_TARGET_CLASSES, VehicleDetector

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)  # 640x480 -> area 307200
FRAME_AREA = FRAME.shape[0] * FRAME.shape[1]


def _fake_box(class_id: int, confidence: float, xyxy: tuple[float, float, float, float]):
    """Stand-in for an ultralytics Boxes row: only the attributes/shape
    VehicleDetector.detect() actually touches (box.cls[0], box.conf[0],
    box.xyxy[0].tolist())."""
    box = MagicMock()
    box.cls = [class_id]
    box.conf = [confidence]
    box.xyxy = [np.array(xyxy, dtype=float)]
    return box


def _make_detector(boxes: list) -> VehicleDetector:
    """Builds a VehicleDetector whose underlying YOLO model is a mock
    returning the given boxes for any frame."""
    with patch("src.detection.vehicle_detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        fake_result = MagicMock()
        fake_result.boxes = boxes
        mock_model.return_value = [fake_result]
        mock_yolo_cls.return_value = mock_model

        detector = VehicleDetector(model_path="unused.pt")
        detector.model = mock_model  # keep the reference for call assertions
        return detector


def test_default_target_classes_used_when_none_passed():
    """Tests that the default target classes are used when none are passed."""
    detector = _make_detector(boxes=[])
    assert detector.target_classes == DEFAULT_TARGET_CLASSES
    # must be a copy, not the shared module-level dict
    detector.target_classes[999] = "not_real"
    assert 999 not in DEFAULT_TARGET_CLASSES


def test_detect_returns_empty_list_for_none_frame():
    """Tests that detect() returns an empty list when passed None."""
    detector = _make_detector(boxes=[])
    assert detector.detect(None) == []


def test_detect_returns_empty_list_for_empty_frame():
    """Tests that detect() returns an empty list when passed an empty frame."""
    detector = _make_detector(boxes=[])
    assert detector.detect(np.array([])) == []


def test_detect_filters_out_non_target_classes():
    """Tests that detect() filters out boxes with non-target classes."""
    # class_id 0 ("person") is not in the default target classes
    boxes = [_fake_box(class_id=0, confidence=0.9, xyxy=(0, 0, 200, 200))]
    detector = _make_detector(boxes)
    assert detector.detect(FRAME) == []


def test_detect_keeps_target_classes_above_area_threshold():
    """Tests that detect() keeps boxes with target classes above the area threshold."""
    # car, box is 200x200 = 40000px, well above 0.1% of 307200
    boxes = [_fake_box(class_id=2, confidence=0.87, xyxy=(10, 20, 210, 220))]
    detector = _make_detector(boxes)

    results = detector.detect(FRAME)

    assert len(results) == 1
    bbox = results[0]
    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (10, 20, 210, 220)
    assert bbox.confidence == pytest.approx(0.87)
    assert bbox.class_id == 2
    assert bbox.class_name == "car"


def test_detect_drops_boxes_below_min_area_ratio():
    """Tests that detect() drops boxes with target classes below the area threshold."""
    # 5x5 = 25px box, far below 0.1% of a 640x480 frame (~307px)
    tiny_box = _fake_box(class_id=2, confidence=0.5, xyxy=(0, 0, 5, 5))
    detector = _make_detector([tiny_box])
    assert detector.detect(FRAME) == []


def test_detect_area_threshold_is_configurable():
    """Tests that detect() respects the min_area_ratio parameter."""
    # Same tiny box, but with min_area_ratio relaxed to 0 it should pass.
    tiny_box = _fake_box(class_id=2, confidence=0.5, xyxy=(0, 0, 5, 5))
    with patch("src.detection.vehicle_detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        fake_result = MagicMock()
        fake_result.boxes = [tiny_box]
        mock_model.return_value = [fake_result]
        mock_yolo_cls.return_value = mock_model

        detector = VehicleDetector(model_path="unused.pt", min_area_ratio=0.0)
        results = detector.detect(FRAME)

    assert len(results) == 1


def test_detect_handles_multiple_classes_in_one_frame():
    """Tests that detect() can handle multiple target classes in one frame."""
    boxes = [
        _fake_box(class_id=2, confidence=0.9, xyxy=(0, 0, 200, 200)),    # car
        _fake_box(class_id=5, confidence=0.8, xyxy=(210, 0, 410, 200)),  # bus
        _fake_box(class_id=7, confidence=0.7, xyxy=(0, 210, 200, 410)),  # truck
        _fake_box(class_id=0, confidence=0.99, xyxy=(210, 210, 410, 410)),  # person, filtered out
    ]
    detector = _make_detector(boxes)

    results = detector.detect(FRAME)

    class_names = {b.class_name for b in results}
    assert class_names == {"car", "bus", "truck"}


def test_detect_respects_custom_target_classes():
    """Tests that detect() respects the target_classes parameter."""
    boxes = [
        _fake_box(class_id=2, confidence=0.9, xyxy=(0, 0, 200, 200)),  # car
        _fake_box(class_id=7, confidence=0.9, xyxy=(0, 0, 200, 200)),  # truck
    ]
    with patch("src.detection.vehicle_detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        fake_result = MagicMock()
        fake_result.boxes = boxes
        mock_model.return_value = [fake_result]
        mock_yolo_cls.return_value = mock_model

        detector = VehicleDetector(model_path="unused.pt", target_classes={7: "truck"})
        results = detector.detect(FRAME)

    assert len(results) == 1
    assert results[0].class_name == "truck"
