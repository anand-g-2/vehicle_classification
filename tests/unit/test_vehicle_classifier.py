"""
Unit tests for src.classification.vehicle_classifier — pure geometry/logic,
no model or tracker involved, so these run instantly.
"""
from __future__ import annotations

import pytest

from src.classification.vehicle_classifier import (
    HV,
    MV,
    SV,
    VehicleClassifier,
    check_line_intersection,
    classify_by_dimensions,
)
from src.pipeline.data_types import BoundingBox, LaneLine, TrackedObject


def _bbox(x1, y1, x2, y2, class_id=2, class_name="car", confidence=0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence,
                        class_id=class_id, class_name=class_name)


def _tracked(track_id, bbox) -> TrackedObject:
    return TrackedObject(track_id=track_id, bbox=bbox, age=1, hits=1, frames_missing=0)


# ---- check_line_intersection ----

def test_intersection_true_when_overlapping_and_within_y_tolerance():
    line = LaneLine(start=(0, 300), end=(200, 300))
    bbox = _bbox(50, 280, 170, 320)  # x-overlap [50,170] = 120 / 200 span = 60%

    crosses, pct = check_line_intersection(line, bbox)

    assert crosses is True
    assert pct == pytest.approx(60.0)


def test_intersection_false_when_no_x_overlap():
    line = LaneLine(start=(0, 300), end=(200, 300))
    bbox = _bbox(250, 280, 350, 320)  # entirely to the right of the line

    crosses, pct = check_line_intersection(line, bbox)

    assert crosses is False
    assert pct == 0.0


def test_intersection_false_when_y_outside_tolerance():
    line = LaneLine(start=(0, 300), end=(200, 300))
    bbox = _bbox(50, 320, 150, 340)  # x overlaps, but y range [320,340] misses y=300 by >5

    crosses, pct = check_line_intersection(line, bbox)

    assert crosses is False


def test_curved_line_widens_y_tolerance():
    straight = LaneLine(start=(0, 300), end=(200, 300), is_curved=False)
    curved = LaneLine(start=(0, 300), end=(200, 300), is_curved=True)
    bbox = _bbox(50, 320, 150, 340)  # misses tol=5, but within tol=20

    crosses_straight, _ = check_line_intersection(straight, bbox)
    crosses_curved, _ = check_line_intersection(curved, bbox)

    assert crosses_straight is False
    assert crosses_curved is True


def test_intersection_zero_length_line_does_not_crash():
    vertical_line = LaneLine(start=(50, 0), end=(50, 300))  # x1 == x2
    bbox = _bbox(0, 0, 100, 300)

    crosses, pct = check_line_intersection(vertical_line, bbox)

    assert crosses is False
    assert pct == 0.0


# ---- classify_by_dimensions ----

def test_classify_wide_and_tall_is_hv():
    line = LaneLine(start=(0, 0), end=(100, 0))  # length 100
    bbox = _bbox(0, 0, 90, 90)  # width=90 (>80), height=90 (>=80)
    assert classify_by_dimensions(bbox, line) == HV


def test_classify_wide_but_short_is_mv():
    line = LaneLine(start=(0, 0), end=(100, 0))
    bbox = _bbox(0, 0, 90, 50)  # width=90 (>80), height=50 (<80)
    assert classify_by_dimensions(bbox, line) == MV


def test_classify_narrow_is_sv():
    line = LaneLine(start=(0, 0), end=(100, 0))
    bbox = _bbox(0, 0, 70, 90)  # width=70 (<=80) — SV regardless of height
    assert classify_by_dimensions(bbox, line) == SV


def test_classify_width_exactly_at_threshold_is_sv():
    # original check is strict `>`, so exactly 0.8 * line.length doesn't qualify
    line = LaneLine(start=(0, 0), end=(100, 0))
    bbox = _bbox(0, 0, 80, 90)  # width == 80 exactly
    assert classify_by_dimensions(bbox, line) == SV


def test_classify_height_exactly_at_threshold_is_hv():
    # inner check is `>=`, so exactly 0.8 * line.length does qualify
    line = LaneLine(start=(0, 0), end=(100, 0))
    bbox = _bbox(0, 0, 90, 80)  # width=90 (>80), height==80 exactly
    assert classify_by_dimensions(bbox, line) == HV


# ---- VehicleClassifier ----

def _single_line_classifier(threshold=50.0) -> VehicleClassifier:
    return VehicleClassifier(lines=[LaneLine(start=(0, 300), end=(200, 300))],
                              intersection_threshold=threshold)


def test_update_classifies_crossing_track():
    clf = _single_line_classifier()
    bbox = _bbox(50, 280, 170, 320)  # 60% crossing — enough to trigger classification

    result = clf.update([_tracked(1, bbox)])

    assert 1 in result
    assert clf.label_for(1) == result[1]
    assert clf.counts[result[1]] == 1


def test_update_does_not_reclassify_existing_track():
    clf = _single_line_classifier()
    bbox = _bbox(50, 280, 170, 320)

    first = clf.update([_tracked(1, bbox)])
    second = clf.update([_tracked(1, bbox)])

    assert 1 in first
    assert second == {}  # already classified — not returned again
    assert sum(clf.counts.values()) == 1  # counted exactly once


def test_update_ignores_track_that_never_crosses():
    clf = _single_line_classifier()
    bbox = _bbox(250, 280, 350, 320)  # no x-overlap with the line at all

    result = clf.update([_tracked(1, bbox)])

    assert result == {}
    assert clf.label_for(1) == ""
    assert sum(clf.counts.values()) == 0


def test_update_respects_intersection_threshold():
    # 60% crossing passes a 50 threshold but not a 70 threshold
    bbox = _bbox(50, 280, 170, 320)
    lenient = _single_line_classifier(threshold=50.0)
    strict = _single_line_classifier(threshold=70.0)

    assert lenient.update([_tracked(1, bbox)]) != {}
    assert strict.update([_tracked(1, bbox)]) == {}


def test_update_uses_first_matching_line_in_order():
    line_a = LaneLine(start=(0, 300), end=(200, 300))
    line_b = LaneLine(start=(0, 300), end=(200, 300))  # identical — both would match
    clf = VehicleClassifier(lines=[line_a, line_b], intersection_threshold=50.0)
    bbox = _bbox(50, 280, 170, 320)

    clf.update([_tracked(1, bbox)])

    # only counted once, against whichever line came first
    assert sum(clf.counts.values()) == 1


def test_reset_clears_classifications_and_counts():
    clf = _single_line_classifier()
    bbox = _bbox(50, 280, 170, 320)
    clf.update([_tracked(1, bbox)])

    clf.reset()

    assert clf.classifications == {}
    assert clf.counts == {SV: 0, MV: 0, HV: 0}
    assert clf.label_for(1) == ""
