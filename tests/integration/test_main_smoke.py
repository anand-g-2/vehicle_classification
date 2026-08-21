"""
Smoke test: main.process_video() run end-to-end with real detector,
tracker, and classifier over a short real clip. Doesn't exercise
calibrate_lanes() (interactive, GUI-blocking) or main() itself (CLI +
calibration UI) -- those are manually tested, per README.

Requires real files that aren't part of the repo:
  - road_traffic.mp4 one level above the repo root (kept out of git)
  - models/detection/yolov8n.pt (kept out of git, see .gitignore)
Skips itself if either is missing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from main import process_video
from src.classification.vehicle_classifier import HV, MV, SV, VehicleClassifier
from src.detection.vehicle_detector import VehicleDetector
from src.pipeline.data_types import LaneLine
from src.tracking.tracker import VehicleTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = REPO_ROOT.parent / "road_traffic.mp4"
MODEL_PATH = REPO_ROOT / "models" / "detection" / "yolov8n.pt"
N_FRAMES = 15

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def short_clip_path(tmp_path_factory):
    """A short temp video built from the first N_FRAMES of road_traffic.mp4
    -- keeps the smoke test fast without needing to process the full file."""
    if not VIDEO_PATH.exists():
        pytest.skip(f"road_traffic.mp4 not found at {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    for _ in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        pytest.skip("Could not read frames from road_traffic.mp4")

    out_path = tmp_path_factory.mktemp("clip") / "short_clip.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()

    return str(out_path), (w, h)


@pytest.fixture(scope="module")
def detector():
    if not MODEL_PATH.exists():
        pytest.skip(f"yolov8n.pt not found at {MODEL_PATH}")
    return VehicleDetector(model_path=str(MODEL_PATH))


def test_process_video_runs_and_writes_output(short_clip_path, detector, tmp_path):
    """Smoke test that process_video() runs end-to-end on a short real clip,
    with real detector, tracker, and classifier, and writes a non-empty output file.
    Also checks that the returned counts dict has the expected keys and non-negative values.
    """
    clip_path, (width, height) = short_clip_path
    lines = [LaneLine(start=(0, height // 2), end=(width, height // 2))]
    tracker = VehicleTracker(tracker_class="ByteTrack")
    classifier = VehicleClassifier(lines=lines)
    output_path = tmp_path / "annotated.mp4"

    counts = process_video(
        video_path=clip_path,
        lines=lines,
        detector=detector,
        tracker=tracker,
        classifier=classifier,
        output_path=str(output_path),
        display=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert set(counts.keys()) == {SV, MV, HV}
    assert all(v >= 0 for v in counts.values())
    # process_video() returns classifier.counts, not a copy that drifted
    assert counts == classifier.counts


def test_process_video_raises_on_missing_video(detector, tmp_path):
    """Tests that process_video() raises FileNotFoundError when the input video path does not exist."""
    tracker = VehicleTracker(tracker_class="ByteTrack")
    classifier = VehicleClassifier(lines=[])

    with pytest.raises(FileNotFoundError):
        process_video(
            video_path=str(tmp_path / "does_not_exist.mp4"),
            lines=[],
            detector=detector,
            tracker=tracker,
            classifier=classifier,
            output_path=str(tmp_path / "out.mp4"),
            display=False,
        )
