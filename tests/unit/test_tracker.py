"""
Unit tests for VehicleTracker. The boxmot tracker is faked out entirely —
no real ByteTrack/motion-matching runs, so these are fast and deterministic.
For a check against a real tracker, see tests/integration.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.data_types import BoundingBox
from src.tracking.tracker import (
    VehicleTracker,
    _filter_accepted_kwargs,
    _translate_kwargs,
)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _detection(x1, y1, x2, y2, class_id=2, class_name="car", confidence=0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence,
                        class_id=class_id, class_name=class_name)


def _fake_raw_tracks(rows: list[tuple[float, float, float, float, int, float, int]]):
    """Stand-in for boxmot's TrackResults: only the attributes/shape
    VehicleTracker.update() actually touches (.size, .xyxy, .id, .conf, .cls)."""
    raw = MagicMock()
    raw.size = len(rows) * 8
    raw.xyxy = np.array([r[0:4] for r in rows], dtype=np.float32) if rows else np.empty((0, 4))
    raw.id = np.array([r[4] for r in rows], dtype=np.float32)
    raw.conf = np.array([r[5] for r in rows], dtype=np.float32)
    raw.cls = np.array([r[6] for r in rows], dtype=np.float32)
    return raw


def _make_tracker(update_side_effect) -> VehicleTracker:
    """Builds a VehicleTracker whose underlying boxmot tracker is a mock
    with the given .update() behaviour (a list of return values, one per
    call, or a single value reused every call)."""
    fake_boxmot_tracker = MagicMock()
    fake_boxmot_tracker.update.side_effect = update_side_effect
    with patch.dict("src.tracking.tracker.TRACKER_CLASSES",
                     {"ByteTrack": MagicMock(return_value=fake_boxmot_tracker)}):
        tracker = VehicleTracker(tracker_class="ByteTrack")
    return tracker


def test_unknown_tracker_class_raises():
    with pytest.raises(ValueError):
        VehicleTracker(tracker_class="NotARealTracker")


def test_update_returns_empty_list_when_no_tracks():
    tracker = _make_tracker(update_side_effect=[_fake_raw_tracks([])])
    result = tracker.update(FRAME, [_detection(0, 0, 50, 50)])
    assert result == []


def test_update_returns_tracked_object_with_correct_fields():
    raw = _fake_raw_tracks([(10, 20, 110, 220, 1, 0.87, 2)])  # car
    tracker = _make_tracker(update_side_effect=[raw])

    result = tracker.update(FRAME, [_detection(10, 20, 110, 220)])

    assert len(result) == 1
    tracked = result[0]
    assert tracked.track_id == 1
    assert tracked.bbox.class_name == "car"
    assert tracked.bbox.confidence == pytest.approx(0.87)
    assert (tracked.bbox.x1, tracked.bbox.y1, tracked.bbox.x2, tracked.bbox.y2) == (10, 20, 110, 220)
    # first sighting: age/hits start at 1, nothing missing yet
    assert tracked.age == 1
    assert tracked.hits == 1
    assert tracked.frames_missing == 0


def test_track_state_accumulates_across_frames():
    raw = _fake_raw_tracks([(0, 0, 50, 50, 1, 0.9, 2)])
    tracker = _make_tracker(update_side_effect=[raw, raw, raw])

    tracker.update(FRAME, [_detection(0, 0, 50, 50)])
    tracker.update(FRAME, [_detection(0, 0, 50, 50)])
    third = tracker.update(FRAME, [_detection(0, 0, 50, 50)])

    assert third[0].age == 3
    assert third[0].hits == 3
    assert third[0].frames_missing == 0


def test_missing_track_increments_frames_missing_and_is_not_returned():
    seen = _fake_raw_tracks([(0, 0, 50, 50, 1, 0.9, 2)])
    empty = _fake_raw_tracks([])
    tracker = _make_tracker(update_side_effect=[seen, empty])

    tracker.update(FRAME, [_detection(0, 0, 50, 50)])
    result = tracker.update(FRAME, [])

    assert result == []
    assert tracker.stats["active_tracks"] == 1  # state kept, just not active this frame


def test_track_reappearing_resumes_state():
    seen = _fake_raw_tracks([(0, 0, 50, 50, 1, 0.9, 2)])
    empty = _fake_raw_tracks([])
    tracker = _make_tracker(update_side_effect=[seen, empty, seen])

    tracker.update(FRAME, [_detection(0, 0, 50, 50)])
    tracker.update(FRAME, [])
    third = tracker.update(FRAME, [_detection(0, 0, 50, 50)])

    assert third[0].track_id == 1
    assert third[0].frames_missing == 0  # reset on re-match
    assert third[0].hits == 2  # matched twice total (frame 1 and frame 3)


def test_unknown_class_id_falls_back_to_string_label():
    raw = _fake_raw_tracks([(0, 0, 50, 50, 1, 0.9, 999)])  # not in target_classes
    tracker = _make_tracker(update_side_effect=[raw])

    result = tracker.update(FRAME, [_detection(0, 0, 50, 50, class_id=999)])

    assert result[0].bbox.class_name == "999"


def test_multiple_tracks_sorted_by_track_id():
    raw = _fake_raw_tracks([
        (100, 100, 150, 150, 5, 0.8, 5),
        (0, 0, 50, 50, 1, 0.9, 2),
    ])
    tracker = _make_tracker(update_side_effect=[raw])

    result = tracker.update(FRAME, [_detection(100, 100, 150, 150), _detection(0, 0, 50, 50)])

    assert [t.track_id for t in result] == [1, 5]


def test_reset_clears_track_state():
    raw = _fake_raw_tracks([(0, 0, 50, 50, 1, 0.9, 2)])
    tracker = _make_tracker(update_side_effect=[raw])
    tracker.update(FRAME, [_detection(0, 0, 50, 50)])

    tracker.reset()

    assert tracker.stats["active_tracks"] == 0


# ---- _translate_kwargs / _filter_accepted_kwargs ----

def test_translate_kwargs_bytetrack_is_identity():
    # ByteTrack IS the canonical vocabulary — its alias table is empty.
    raw = {"track_thresh": 0.5, "min_conf": 0.1}
    assert _translate_kwargs("ByteTrack", raw) == raw


def test_translate_kwargs_botsort_renames_known_params():
    raw = {"track_thresh": 0.5, "min_conf": 0.1}
    translated = _translate_kwargs("BotSort", raw)
    assert translated == {"track_high_thresh": 0.5, "track_low_thresh": 0.1}


def test_translate_kwargs_leaves_unaliased_params_untouched():
    # frame_rate isn't in any tracker's alias table — passes through as-is.
    raw = {"track_thresh": 0.5, "frame_rate": 30}
    translated = _translate_kwargs("BotSort", raw)
    assert translated["frame_rate"] == 30
    assert translated["track_high_thresh"] == 0.5


def test_filter_accepted_kwargs_passes_everything_when_target_has_kwargs():
    class _HasCatchAll:
        def __init__(self, known=1, **kwargs):
            pass

    accepted, dropped = _filter_accepted_kwargs(_HasCatchAll, {"known": 2, "unknown": 3})
    assert accepted == {"known": 2, "unknown": 3}
    assert dropped == []


def test_filter_accepted_kwargs_drops_unrecognised_params_without_catch_all():
    class _NoCatchAll:
        def __init__(self, known=1):
            pass

    accepted, dropped = _filter_accepted_kwargs(_NoCatchAll, {"known": 2, "unknown": 3})
    assert accepted == {"known": 2}
    assert dropped == ["unknown"]


def test_vehicle_tracker_passes_translated_kwargs_to_constructor():
    # A real (non-mock) class, so _filter_accepted_kwargs introspects an
    # actual signature rather than guessing at MagicMock's dunder behaviour.
    captured: dict = {}

    class _FakeBotSort:
        def __init__(self, track_high_thresh=0.5, **kwargs):
            captured["received"] = {"track_high_thresh": track_high_thresh, **kwargs}

    with patch.dict("src.tracking.tracker.TRACKER_CLASSES", {"BotSort": _FakeBotSort}):
        VehicleTracker(tracker_class="BotSort", track_thresh=0.7)

    # "track_thresh" (canonical) must have arrived as "track_high_thresh" (native)
    assert captured["received"] == {"track_high_thresh": 0.7}
