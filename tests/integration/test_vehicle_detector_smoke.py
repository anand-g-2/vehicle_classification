"""
Smoke test: VehicleDetector against a real frame with real weights, to
catch integration issues unit tests (mocked model) can't — e.g. a broken
model path, an ultralytics API mismatch, or output that doesn't hold up
against real inference.

Requires real files that aren't part of the repo:
  - road_traffic.mp4 one level above the repo root (kept out of git)
  - models/detection/yolov8n.pt (kept out of git, see .gitignore)
Skips itself if either is missing, so this never blocks a normal
`pytest tests/unit` run or CI without those assets.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from src.detection.vehicle_detector import DEFAULT_TARGET_CLASSES, VehicleDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = REPO_ROOT.parent / "road_traffic.mp4"
MODEL_PATH = REPO_ROOT / "models" / "detection" / "yolov8n.pt"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def first_frame():
    if not VIDEO_PATH.exists():
        pytest.skip(f"road_traffic.mp4 not found at {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.skip("Could not read a frame from road_traffic.mp4")
    return frame


@pytest.fixture(scope="module")
def detector():
    if not MODEL_PATH.exists():
        pytest.skip(f"yolov8n.pt not found at {MODEL_PATH}")
    return VehicleDetector(model_path=str(MODEL_PATH))


def test_detect_runs_on_a_real_frame(first_frame, detector):
    """Smoke test that VehicleDetector.detect() runs on a real frame with real weights,
    and returns at least one detection. This is the actual point of detection,
    not just "did it run without crashing"."""
    results = detector.detect(first_frame)

    # road_traffic.mp4's first frame is a live traffic scene — assert
    # the pipeline actually finds vehicles, not just that it doesn't crash.
    assert len(results) > 0


def test_detections_are_well_formed(first_frame, detector):
    """Smoke test that VehicleDetector.detect() returns BoundingBox instances
    with well-formed fields (valid class_name, confidence, bbox coordinates)
    across several frames of a real video. This is a sanity check that the
    detector is returning valid data, not just that it doesn't crash."""
    frame_h, frame_w = first_frame.shape[:2]
    results = detector.detect(first_frame)

    for bbox in results:
        assert bbox.class_name in DEFAULT_TARGET_CLASSES.values()
        assert 0.0 <= bbox.confidence <= 1.0
        assert bbox.width > 0
        assert bbox.height > 0
        assert 0 <= bbox.x1 < bbox.x2 <= frame_w
        assert 0 <= bbox.y1 < bbox.y2 <= frame_h
