"""
main.py

CLI entry point: reads a video, lets you calibrate lane lines by drawing
on the first frame, then runs detection -> tracking -> classification ->
annotation per frame, displaying and/or saving the annotated output.

Structured port of vehicle_detection_template_v2_3.py's draw_line() +
process_video(): same click-drag-release calibration gesture and 'c' to
continue, same per-frame pipeline — split into testable pieces instead of
one top-level script.

One deliberate behaviour change from the original: v2_3.py reads the
video's first frame for calibration and then keeps processing from the
SAME cv2.VideoCapture, so that first frame is shown during calibration
but never actually written to the output video. Here, process_video()
reopens the video and processes every frame including the first — a
"process this video" function that silently skips frame 1 is a
surprising API, so calibration now reads its own throwaway frame instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import structlog

from src.classification.vehicle_classifier import VehicleClassifier
from src.detection.vehicle_detector import VehicleDetector
from src.pipeline.data_types import LaneLine
from src.tracking.tracker import VehicleTracker
from src.utils.annotator import annotate_frame

log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = REPO_ROOT.parent / "road_traffic.mp4"
DEFAULT_MODEL = REPO_ROOT / "models" / "detection" / "yolov8n.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect, track, and classify vehicles (SV/MV/HV) in a traffic video."
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to the input video.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO weights path.")
    parser.add_argument(
        "--tracker", default="ByteTrack", choices=["ByteTrack", "BotSort", "StrongSort"],
        help="Tracker backend. BotSort/StrongSort need ReID weights this repo doesn't ship.",
    )
    parser.add_argument("--output", default="vehicle_track.mp4", help="Path to write the annotated video.")
    parser.add_argument(
        "--intersection-threshold", type=float, default=50.0,
        help="Minimum percent of a calibration line a bbox must cross to trigger classification.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't open a preview window while processing (--output is still written).",
    )
    return parser.parse_args(argv)


def calibrate_lanes(first_frame: np.ndarray) -> list[LaneLine]:
    """
    Interactive calibration: click-drag to draw one line per lane on
    first_frame; press 'c' to finish.

    Convention carried over from the original script: if you draw 4 or
    more lines, the 4th one drawn is treated as the curved/merge lane
    (wider crossing tolerance) — see LaneLine.is_curved. This used to be
    a hardcoded `i == 3` check at classification time; it's now a real
    field set once, here, at calibration time.
    """
    points: list[tuple[int, int]] = []
    lines: list[LaneLine] = []
    drawing = False
    preview = first_frame.copy()

    def _redraw() -> np.ndarray:
        img = first_frame.copy()
        for line in lines:
            cv2.line(img, line.start, line.end, (0, 255, 0), 2)
        return img

    def on_mouse(event, x, y, flags, param) -> None:
        nonlocal drawing, points, preview
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            preview = _redraw()
            cv2.line(preview, points[0], (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            lines.append(LaneLine(start=points[0], end=(x, y)))
            log.info(
                "calibration.line_added",
                index=len(lines) - 1,
                start=points[0], end=(x, y),
                length=round(lines[-1].length, 1),
            )
            preview = _redraw()

    window = "Calibrate lanes -- draw a line per lane, press 'c' when done"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print("Draw one line per lane (click-drag-release). Press 'c' to continue.")
    while True:
        cv2.imshow(window, preview)
        if cv2.waitKey(1) & 0xFF == ord("c"):
            break
    cv2.destroyWindow(window)

    if len(lines) >= 4:
        lines[3].is_curved = True

    return lines


def process_video(
    video_path: str,
    lines: list[LaneLine],
    detector: VehicleDetector,
    tracker: VehicleTracker,
    classifier: VehicleClassifier,
    output_path: str,
    display: bool = True,
) -> dict[str, int]:
    """
    Runs detect -> track -> classify -> annotate over every frame of
    video_path, writing the annotated result to output_path. Returns the
    final SV/MV/HV counts.

    Deliberately separate from main() / calibrate_lanes() so it's
    callable and testable on its own, with pre-built components and
    already-calibrated lines, no interactive UI or CLI parsing involved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracked = tracker.update(frame, detections)
            classifier.update(tracked)

            labels = {t.track_id: classifier.label_for(t.track_id) for t in tracked}
            annotated = annotate_frame(frame, tracked, labels, classifier.counts, lines)

            out.write(annotated)
            if display:
                cv2.imshow("Processed Frames", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

    log.info("process_video.complete", output=output_path, counts=classifier.counts)
    return dict(classifier.counts)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read the first frame of {args.video}")

    lines = calibrate_lanes(first_frame)
    if not lines:
        raise RuntimeError("No lane lines were drawn -- nothing to classify vehicles against.")

    detector = VehicleDetector(model_path=args.model)
    tracker = VehicleTracker(tracker_class=args.tracker)
    classifier = VehicleClassifier(lines=lines, intersection_threshold=args.intersection_threshold)

    counts = process_video(
        video_path=args.video,
        lines=lines,
        detector=detector,
        tracker=tracker,
        classifier=classifier,
        output_path=args.output,
        display=not args.no_display,
    )
    print(f"Done. Counts: {counts}. Output saved to {args.output}")


if __name__ == "__main__":
    main()
