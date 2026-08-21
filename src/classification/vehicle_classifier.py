"""
src/classification/vehicle_classifier.py

Ports the SV/MV/HV classification logic from vehicle_detection_template_v2_3.py:
a vehicle is classified once, the first time its tracked bbox crosses a
calibrated LaneLine by more than `intersection_threshold` percent, using
its on-screen size relative to that line's length.
"""
from __future__ import annotations

import structlog

from src.pipeline.data_types import BoundingBox, LaneLine, TrackedObject

log = structlog.get_logger(__name__)

SV = "SV"
MV = "MV"
HV = "HV"


def check_line_intersection(line: LaneLine, bbox: BoundingBox) -> tuple[bool, float]:
    """
    Does bbox cross `line`, and by what percentage of the line's
    horizontal span?

    A crossing requires: (a) the bbox's x-range overlaps the line's
    x-range, and (b) the line's start point sits within the bbox's
    y-range (widened by a tolerance for curved/angled lanes).

    Returns (False, 0.0) when there's no crossing.
    """
    (x1, y1), (x2, y2) = line.start, line.end
    horizontal_span = abs(x2 - x1)
    if horizontal_span == 0:
        return False, 0.0

    y_tolerance = 20 if line.is_curved else 5

    intersection_start = max(x1, bbox.x1)
    intersection_end = min(x2, bbox.x2)
    if intersection_end <= intersection_start:
        return False, 0.0

    intersection_pct = (intersection_end - intersection_start) / horizontal_span * 100

    if (bbox.y1 - y_tolerance) <= y1 <= (bbox.y2 + y_tolerance):
        return True, intersection_pct

    return False, 0.0


def classify_by_dimensions(bbox: BoundingBox, line: LaneLine) -> str:
    """
    SV / MV / HV from a vehicle's on-screen size relative to the
    crossing line's length. Height is checked first, then width.
    """
    if bbox.height > 0.8 * line.length:
        if bbox.width >= 0.8 * line.length:
            return HV
        return MV
    return SV


class VehicleClassifier:
    """
    Stateful: classifies each track_id exactly once, the first frame it
    crosses any calibrated line by more than intersection_threshold
    percent — same "first crossing wins, never re-classified" behaviour
    as v2_3.py's detected_vehicles dict.
    """

    def __init__(
        self,
        lines: list[LaneLine],
        intersection_threshold: float = 50.0,
    ) -> None:
        """
        Args:
            lines: calibrated LaneLines, checked in order — the first one
                a track crosses (by more than intersection_threshold) is
                the one used for its size classification.
            intersection_threshold: minimum crossing percentage (0-100)
                required before a track is classified.
        """
        self.lines = lines
        self.intersection_threshold = intersection_threshold
        self.classifications: dict[int, str] = {}
        self.counts: dict[str, int] = {SV: 0, MV: 0, HV: 0}

    def update(self, tracked_objects: list[TrackedObject]) -> dict[int, str]:
        """
        Classifies any not-yet-classified tracks that cross a line this
        frame. Returns only the classifications decided by this call
        (empty dict if none) — self.classifications holds the full
        history across all calls.
        """
        newly_classified: dict[int, str] = {}

        for obj in tracked_objects:
            if obj.track_id in self.classifications:
                continue

            for line in self.lines:
                crosses, pct = check_line_intersection(line, obj.bbox)
                if crosses and pct > self.intersection_threshold:
                    label = classify_by_dimensions(obj.bbox, line)
                    self.classifications[obj.track_id] = label
                    self.counts[label] += 1
                    newly_classified[obj.track_id] = label
                    log.info(
                        "vehicle_classifier.classified",
                        track_id=obj.track_id,
                        label=label,
                        intersection_pct=round(pct, 1),
                    )
                    break

        return newly_classified

    def label_for(self, track_id: int) -> str:
        """Empty string if track_id hasn't been classified yet — matches
        v2_3.py's `detected_vehicles.get(track_id, '')`."""
        return self.classifications.get(track_id, "")

    def reset(self) -> None:
        self.classifications.clear()
        self.counts = {SV: 0, MV: 0, HV: 0}
        log.info("vehicle_classifier.reset")
