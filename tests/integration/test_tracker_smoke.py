"""
Smoke test: VehicleDetector + VehicleTracker (real ByteTrack) run together
across several real frames, to catch integration issues unit tests
(mocked tracker) can't — e.g. a boxmot API mismatch, or detections that
never actually produce a stable track in practice.

Requires real files that aren't part of the repo:
  - road_traffic.mp4 one level above the repo root (kept out of git)
  - models/detection/yolov8n.pt (kept out of git, see .gitignore)
Skips itself if either is missing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from src.detection.vehicle_detector import DEFAULT_TARGET_CLASSES, VehicleDetector
from src.tracking.tracker import VehicleTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = REPO_ROOT.parent / "road_traffic.mp4"
MODEL_PATH = REPO_ROOT / "models" / "detection" / "yolov8n.pt"
N_FRAMES = 15  # ByteTrack's default min_hits=3, so a few frames are needed
               # before a track is confirmed and actually returned.

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def frames():
    if not VIDEO_PATH.exists():
        pytest.skip(f"road_traffic.mp4 not found at {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    collected = []
    for _ in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        collected.append(frame)
    cap.release()
    if not collected:
        pytest.skip("Could not read frames from road_traffic.mp4")
    return collected


@pytest.fixture(scope="module")
def detector():
    if not MODEL_PATH.exists():
        pytest.skip(f"yolov8n.pt not found at {MODEL_PATH}")
    return VehicleDetector(model_path=str(MODEL_PATH))


def test_tracks_are_confirmed_and_persist_across_frames(frames, detector):
    tracker = VehicleTracker(tracker_class="ByteTrack")

    track_ids_per_frame: list[set[int]] = []
    for frame in frames:
        detections = detector.detect(frame)
        tracked = tracker.update(frame, detections)
        track_ids_per_frame.append({t.track_id for t in tracked})

    all_seen_ids = set().union(*track_ids_per_frame)
    assert len(all_seen_ids) > 0, "no track was ever confirmed across N_FRAMES"

    # At least one track ID must reappear in consecutive frames — this is
    # the actual point of tracking (a stable identity across time), not
    # just "did detection find something."
    persisted = any(
        track_ids_per_frame[i] & track_ids_per_frame[i + 1]
        for i in range(len(track_ids_per_frame) - 1)
    )
    assert persisted, "no track ID was shared between any two consecutive frames"


def test_tracked_objects_are_well_formed(frames, detector):
    tracker = VehicleTracker(tracker_class="ByteTrack")
    frame_h, frame_w = frames[0].shape[:2]

    seen_any = False
    for frame in frames:
        detections = detector.detect(frame)
        for tracked in tracker.update(frame, detections):
            seen_any = True
            assert tracked.track_id > 0
            assert tracked.bbox.class_name in DEFAULT_TARGET_CLASSES.values()
            assert 0 <= tracked.bbox.x1 < tracked.bbox.x2 <= frame_w
            assert 0 <= tracked.bbox.y1 < tracked.bbox.y2 <= frame_h
            assert tracked.hits >= 1
            assert tracked.frames_missing == 0  # only active tracks are returned

    assert seen_any, "no tracked objects were produced across N_FRAMES"
