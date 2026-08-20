"""
src/utils/annotator.py

Draws the same visual overlay as vehicle_detection_template_v2_3.py's
process_video() loop — calibration lines, per-track box + label, and a
running SV/MV/HV count — as reusable, testable drawing functions instead
of inline per-frame code.

One cleanup vs. the original: v2_3.py redraws the lines and the current
track's box/label/count once per calibration line it checks (nested
inside `for i in range(len(lines))`), i.e. up to 4x per track per frame
for no visual difference — pure waste, not a behaviour a caller should
rely on. Here each element is drawn exactly once per frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.classification.vehicle_classifier import HV, MV, SV
from src.pipeline.data_types import LaneLine, TrackedObject

LINE_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 255)
LABEL_COLOR = (0, 0, 255)
COUNT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_lines(
    frame: np.ndarray,
    lines: list[LaneLine],
    color: tuple[int, int, int] = LINE_COLOR,
    thickness: int = 2,
) -> None:
    """Draws all calibration lines onto frame, in place."""
    for line in lines:
        cv2.line(frame, line.start, line.end, color, thickness)


def draw_tracked_object(
    frame: np.ndarray,
    tracked: TrackedObject,
    label: str,
    box_color: tuple[int, int, int] = BOX_COLOR,
    label_color: tuple[int, int, int] = LABEL_COLOR,
    box_thickness: int = 1,
) -> None:
    """Draws one track's bounding box, plus its classification label
    centered in the box if it has one (empty string draws no text —
    matches a not-yet-classified track), in place."""
    x1, y1, x2, y2 = map(int, (tracked.bbox.x1, tracked.bbox.y1, tracked.bbox.x2, tracked.bbox.y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
    if label:
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(frame, label, (center_x, center_y), FONT, 1, label_color, 2)


def draw_counts(
    frame: np.ndarray,
    counts: dict[str, int],
    position: tuple[int, int] = (10, 20),
    color: tuple[int, int, int] = COUNT_COLOR,
) -> None:
    """Draws the running SV/MV/HV tally, in place."""
    text = f"SV: {counts.get(SV, 0)}  MV: {counts.get(MV, 0)}  HV: {counts.get(HV, 0)}"
    cv2.putText(frame, text, position, FONT, 1, color, 2)


def annotate_frame(
    frame: np.ndarray,
    tracked_objects: list[TrackedObject],
    labels: dict[int, str],
    counts: dict[str, int],
    lines: list[LaneLine],
) -> np.ndarray:
    """
    Returns a new annotated copy of frame: calibration lines, one
    box(+label) per tracked object, and the running count. Does not
    mutate the input frame.
    """
    annotated = frame.copy()
    draw_lines(annotated, lines)
    for tracked in tracked_objects:
        draw_tracked_object(annotated, tracked, labels.get(tracked.track_id, ""))
    draw_counts(annotated, counts)
    return annotated
