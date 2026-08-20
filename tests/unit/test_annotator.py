"""
Unit tests for src.utils.annotator. Pure OpenCV drawing onto numpy arrays
— checked via pixel sampling rather than exact-render comparison, since
glyph rendering isn't something worth pinning pixel-for-pixel.
"""
from __future__ import annotations

import numpy as np

from src.pipeline.data_types import BoundingBox, LaneLine, TrackedObject
from src.utils.annotator import annotate_frame, draw_counts, draw_lines, draw_tracked_object

BLACK = (0, 0, 0)


def _blank_frame() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)


def _tracked(track_id, x1, y1, x2, y2) -> TrackedObject:
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, class_id=2, class_name="car")
    return TrackedObject(track_id=track_id, bbox=bbox, age=1, hits=1, frames_missing=0)


# ---- draw_lines ----

def test_draw_lines_paints_line_pixels():
    frame = _blank_frame()
    line = LaneLine(start=(10, 50), end=(190, 50))

    draw_lines(frame, [line], color=(255, 255, 255), thickness=2)

    assert tuple(frame[50, 100]) == (255, 255, 255)   # midpoint of the line
    assert tuple(frame[10, 100]) == BLACK              # well away from it


def test_draw_lines_is_in_place():
    frame = _blank_frame()
    line = LaneLine(start=(10, 50), end=(190, 50))

    result = draw_lines(frame, [line])

    assert result is None
    assert not np.array_equal(frame, _blank_frame())  # the passed-in frame itself changed


# ---- draw_tracked_object ----

def test_draw_tracked_object_paints_box_border_not_interior():
    frame = _blank_frame()
    tracked = _tracked(1, 20, 20, 80, 80)

    draw_tracked_object(frame, tracked, label="", box_color=(0, 0, 255), box_thickness=1)

    assert tuple(frame[20, 50]) == (0, 0, 255)   # top border
    assert tuple(frame[50, 50]) == BLACK          # interior — thickness=1, not filled


def test_draw_tracked_object_nonempty_label_adds_visible_text():
    box_only = _blank_frame()
    with_label = _blank_frame()
    tracked = _tracked(1, 20, 20, 80, 80)

    draw_tracked_object(box_only, tracked, label="")
    draw_tracked_object(with_label, tracked, label="HV")

    assert not np.array_equal(box_only, with_label)


# ---- draw_counts ----

def test_draw_counts_paints_near_position_and_leaves_rest_untouched():
    frame = _blank_frame()

    draw_counts(frame, {"SV": 1, "MV": 2, "HV": 3}, position=(10, 20))

    assert not np.array_equal(frame, _blank_frame())
    assert tuple(frame[190, 190]) == BLACK  # far corner, nowhere near the text


# ---- annotate_frame ----

def test_annotate_frame_does_not_mutate_input():
    frame = _blank_frame()
    original = frame.copy()
    line = LaneLine(start=(10, 50), end=(190, 50))
    tracked = _tracked(1, 20, 20, 80, 80)

    annotate_frame(frame, [tracked], {1: "HV"}, {"SV": 0, "MV": 0, "HV": 1}, [line])

    assert np.array_equal(frame, original)


def test_annotate_frame_returns_annotated_copy():
    frame = _blank_frame()
    line = LaneLine(start=(10, 50), end=(190, 50))
    tracked = _tracked(1, 20, 20, 80, 80)

    annotated = annotate_frame(frame, [tracked], {1: "HV"}, {"SV": 0, "MV": 0, "HV": 1}, [line])

    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)


def test_annotate_frame_draws_unclassified_tracks_with_no_label_text():
    frame = _blank_frame()
    tracked = _tracked(1, 20, 20, 80, 80)

    # no entry for track_id 1 in labels — matches a not-yet-classified track
    annotated = annotate_frame(frame, [tracked], {}, {"SV": 0, "MV": 0, "HV": 0}, [])

    assert tuple(annotated[20, 50]) == (0, 0, 255)  # box is still drawn
