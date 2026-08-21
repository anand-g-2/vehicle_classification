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
    """Tests that check_line_intersection returns True and the correct percentage
    when a bounding box overlaps a lane line in both x and y (within tolerance)."""
    line = LaneLine(start=(0, 300), end=(200, 300))
    bbox = _bbox(50, 280, 170, 320)  # x-overlap [50,170] = 120 / 200 span = 60%

    crosses, pct = check_line_intersection(line, bbox)

    assert crosses is True
    assert pct == pytest.approx(60.0)


def test_intersection_false_when_no_x_overlap():
    """Tests that check_line_intersection returns False and 0.0 percentage
    when a bounding box does not overlap a lane line in the x dimension."""
    line = LaneLine(start=(0, 300), end=(200, 300))
    bbox = _bbox(250, 280, 350, 320)  # entirely to the right of the line

    crosses, pct = check_line_intersection(line, bbox)

    assert crosses is False
    assert pct == 0.0


def test_intersection_false_when_y_outside_tolerance():
    """Tests that check_line_intersection returns False and 0.0 percentage
    when a bounding box overlaps a lane line in x but is outside the y tolerance."""
    line = LaneLine(start=(0, 300), end=(200, 300))
    bbox = _bbox(50, 320, 150, 340)  # x overlaps, but y range [320,340] misses y=300 by >5

    crosses, pct = check_line_intersection(line, bbox)

    assert crosses is False


def test_curved_line_widens_y_tolerance():
    """Tests that a curved LaneLine increases the y-tolerance for intersection checks,
    allowing a bounding box that would otherwise miss to be considered crossing."""
    straight = LaneLine(start=(0, 300), end=(200, 300), is_curved=False)
    curved = LaneLine(start=(0, 300), end=(200, 300), is_curved=True)
    bbox = _bbox(50, 320, 150, 340)  # misses tol=5, but within tol=20

    crosses_straight, _ = check_line_intersection(straight, bbox)
    crosses_curved, _ = check_line_intersection(curved, bbox)

    assert crosses_straight is False
    assert crosses_curved is True


def test_intersection_zero_length_line_does_not_crash():
    """Tests that check_line_intersection handles a zero-length line gracefully, 
    returning False and 0.0 percentage."""
    vertical_line = LaneLine(start=(50, 0), end=(50, 300))  # x1 == x2
    bbox = _bbox(0, 0, 100, 300)

    crosses, pct = check_line_intersection(vertical_line, bbox)

    assert crosses is False
    assert pct == 0.0


# ---- classify_by_dimensions ----

def test_classify_tall_and_wide_is_hv():
    """Tests that a bounding box that is both tall and wide is classified as HV."""
    line = LaneLine(start=(0, 0), end=(100, 0))  # length 100
    bbox = _bbox(0, 0, 90, 90)  # height=90 (>80), width=90 (>=80)
    assert classify_by_dimensions(bbox, line) == HV


def test_classify_tall_but_narrow_is_mv():
    """Tests that a bounding box that is tall but not wide is classified as MV."""
    line = LaneLine(start=(0, 0), end=(100, 0))
    bbox = _bbox(0, 0, 50, 90)  # height=90 (>80), width=50 (<80)
    assert classify_by_dimensions(bbox, line) == MV


def test_classify_short_is_sv():
    """Tests that a bounding box that is short is classified as SV."""
    line = LaneLine(start=(0, 0), end=(100, 0))
    bbox = _bbox(0, 0, 90, 70)  # height=70 (<=80) — SV regardless of width
    assert classify_by_dimensions(bbox, line) == SV


def test_classify_height_exactly_at_threshold_is_sv():
    """Tests that a bounding box with height exactly at the threshold is classified as SV,
    since the height check uses a strict greater-than comparison."""
    # outer check is strict `>`, so exactly 0.8 * line.length doesn't qualify
    line = LaneLine(start=(0, 0), end=(100, 0))
    bbox = _bbox(0, 0, 90, 80)  # height == 80 exactly
    assert classify_by_dimensions(bbox, line) == SV


def test_classify_width_exactly_at_threshold_is_hv():
    """Tests that a bounding box with width exactly at the threshold is classified as HV,
    since the width check uses a inclusive greater-than-or-equal comparison."""
    # inner check is `>=`, so exactly 0.8 * line.length does qualify
    line = LaneLine(start=(0, 0), end=(100, 0))
    bbox = _bbox(0, 0, 80, 90)  # height=90 (>80), width==80 exactly
    assert classify_by_dimensions(bbox, line) == HV


# ---- VehicleClassifier ----

def _single_line_classifier(threshold=50.0) -> VehicleClassifier:
    return VehicleClassifier(lines=[LaneLine(start=(0, 300), end=(200, 300))],
                              intersection_threshold=threshold)


def test_update_classifies_crossing_track():
    """Tests that VehicleClassifier.update() classifies a track that 
    crosses a line by more than the intersection threshold, and returns the classification."""
    clf = _single_line_classifier()
    bbox = _bbox(50, 280, 170, 320)  # 60% crossing — enough to trigger classification

    result = clf.update([_tracked(1, bbox)])

    assert 1 in result
    assert clf.label_for(1) == result[1]
    assert clf.counts[result[1]] == 1


def test_update_does_not_reclassify_existing_track():
    """Tests that VehicleClassifier.update() does not reclassify a track 
    that has already been classified, even if the track crosses the line again."""
    clf = _single_line_classifier()
    bbox = _bbox(50, 280, 170, 320)

    first = clf.update([_tracked(1, bbox)])
    second = clf.update([_tracked(1, bbox)])

    assert 1 in first
    assert second == {}  # already classified — not returned again
    assert sum(clf.counts.values()) == 1  # counted exactly once


def test_update_ignores_track_that_never_crosses():
    """Tests that VehicleClassifier.update() ignores a track 
    that never crosses a line by more than the intersection threshold."""
    clf = _single_line_classifier()
    bbox = _bbox(250, 280, 350, 320)  # no x-overlap with the line at all

    result = clf.update([_tracked(1, bbox)])

    assert result == {}
    assert clf.label_for(1) == ""
    assert sum(clf.counts.values()) == 0


def test_update_respects_intersection_threshold():
    """Tests that VehicleClassifier.update() respects the intersection_threshold parameter, 
    classifying a track only if it crosses a line by more than the specified percentage."""
    # 60% crossing passes a 50 threshold but not a 70 threshold
    bbox = _bbox(50, 280, 170, 320)
    lenient = _single_line_classifier(threshold=50.0)
    strict = _single_line_classifier(threshold=70.0)

    assert lenient.update([_tracked(1, bbox)]) != {}
    assert strict.update([_tracked(1, bbox)]) == {}


def test_update_uses_first_matching_line_in_order():
    """Tests that VehicleClassifier.update() uses the first line in the list 
    that a track crosses, even if multiple lines would match."""
    line_a = LaneLine(start=(0, 300), end=(200, 300))
    line_b = LaneLine(start=(0, 300), end=(200, 300))  # identical — both would match
    clf = VehicleClassifier(lines=[line_a, line_b], intersection_threshold=50.0)
    bbox = _bbox(50, 280, 170, 320)

    clf.update([_tracked(1, bbox)])

    # only counted once, against whichever line came first
    assert sum(clf.counts.values()) == 1


def test_reset_clears_classifications_and_counts():
    """Tests that VehicleClassifier.reset() clears all classifications and counts,
    allowing the classifier to start fresh."""
    clf = _single_line_classifier()
    bbox = _bbox(50, 280, 170, 320)
    clf.update([_tracked(1, bbox)])

    clf.reset()

    assert clf.classifications == {}
    assert clf.counts == {SV: 0, MV: 0, HV: 0}
    assert clf.label_for(1) == ""
