"""
src/tracking/tracker.py

Wraps boxmot trackers behind a common interface: BoundingBox detections
in, persistent-ID TrackedObjects out. tracker_class is a config choice
("ByteTrack" | "BotSort" | "StrongSort") — only ByteTrack is exercised
so far; BotSort/StrongSort need ReID weights this repo doesn't ship yet.
"""
from __future__ import annotations

import inspect
import sys
from typing import Any

import numpy as np
import structlog

from src.detection.vehicle_detector import DEFAULT_TARGET_CLASSES
from src.pipeline.data_types import BoundingBox, TrackedObject

log = structlog.get_logger(__name__)

# boxmot's internal modules do `import lap`. On Windows, lap has no wheel;
# lapx is the drop-in replacement. Patch sys.modules so boxmot finds it.
try:
    import lap  # noqa: F401
except ModuleNotFoundError:
    import lapx as lap  # noqa: F401
    sys.modules["lap"] = lap
    log.debug("vehicle_tracker.lapx_shim_applied")

from boxmot.trackers import BotSort, ByteTrack, StrongSort  # noqa: E402

TRACKER_CLASSES: dict[str, type] = {
    "ByteTrack": ByteTrack,
    "BotSort": BotSort,
    "StrongSort": StrongSort,
}

# Canonical config names -> native parameter names per tracker. Lets call
# sites use one consistent vocabulary (track_thresh, min_conf, match_thresh,
# track_buffer — ByteTrack's own names) regardless of which tracker_class
# is actually chosen. Verified against boxmot==19.0.0's real constructors:
#   ByteTrack  : min_conf, track_thresh, match_thresh, track_buffer
#   BotSort    : track_high_thresh, track_low_thresh, match_thresh, track_buffer
#   StrongSort : min_conf, max_iou_dist  (no track_buffer-equivalent;
#                still aliased to "max_age" for interface symmetry — lands
#                in StrongSort's **kwargs today and is a no-op there)
_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "ByteTrack": {},
    "BotSort": {"track_thresh": "track_high_thresh", "min_conf": "track_low_thresh"},
    "StrongSort": {"match_thresh": "max_iou_dist", "track_buffer": "max_age"},
}


def _translate_kwargs(tracker_name: str, raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    aliases = _PARAM_ALIASES.get(tracker_name, {})
    return {aliases.get(k, k): v for k, v in raw_kwargs.items()}


def _filter_accepted_kwargs(cls: type, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop kwargs the target __init__ doesn't accept — surfaced as a
    warning rather than a silent no-op or a TypeError. Skipped entirely
    when the target has its own **kwargs catch-all (true for all three
    boxmot trackers today), since there's nothing meaningful to filter."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return kwargs, []
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []
    accepted_names = set(params.keys())
    accepted = {k: v for k, v in kwargs.items() if k in accepted_names}
    dropped = [k for k in kwargs if k not in accepted_names]
    return accepted, dropped


class _TrackState:
    __slots__ = ("age", "hits", "frames_missing")

    def __init__(self) -> None:
        self.age = 1
        self.hits = 1
        self.frames_missing = 0

    def on_match(self) -> None:
        self.age += 1
        self.hits += 1
        self.frames_missing = 0

    def on_miss(self) -> None:
        self.age += 1
        self.frames_missing += 1


class VehicleTracker:
    def __init__(
        self,
        tracker_class: str = "ByteTrack",
        target_classes: dict[int, str] | None = None,
        **tracker_kwargs: Any,
    ) -> None:
        """
        Args:
            tracker_class: "ByteTrack" | "BotSort" | "StrongSort".
            target_classes: COCO class_id -> label for naming tracked
                boxes. Defaults to the same set VehicleDetector uses.
            **tracker_kwargs: canonical-named options (ByteTrack's own
                param names — track_thresh, min_conf, match_thresh,
                track_buffer, ...), translated to whichever native names
                the chosen tracker actually expects. See _PARAM_ALIASES.
        """
        chosen_class = TRACKER_CLASSES.get(tracker_class)
        if chosen_class is None:
            raise ValueError(
                f"Unknown tracker_class '{tracker_class}'. "
                f"Available: {list(TRACKER_CLASSES.keys())}"
            )

        self._tracker_class = tracker_class
        self.target_classes = target_classes or dict(DEFAULT_TARGET_CLASSES)
        self._track_states: dict[int, _TrackState] = {}

        translated = _translate_kwargs(tracker_class, tracker_kwargs)
        tracker_params, dropped = _filter_accepted_kwargs(chosen_class, translated)
        if dropped:
            log.warning(
                "vehicle_tracker.params_dropped",
                tracker=tracker_class,
                dropped=dropped,
                hint="check the tracker's constructor, or update _PARAM_ALIASES",
            )

        self.tracker = chosen_class(**tracker_params)
        log.info("vehicle_tracker.initialised", tracker=tracker_class, params=tracker_params)

    def update(self, frame: np.ndarray, detections: list[BoundingBox]) -> list[TrackedObject]:
        """
        Advances the tracker by one frame.

        Returns all currently active TrackedObjects (empty list if none
        are active this frame — note ByteTrack requires a detection to be
        matched across several frames, via min_hits, before it appears
        here at all).
        """
        if detections:
            det_array = np.array(
                [[d.x1, d.y1, d.x2, d.y2, d.confidence, float(d.class_id)] for d in detections],
                dtype=np.float32,
            )
        else:
            det_array = np.empty((0, 6), dtype=np.float32)

        try:
            raw_tracks = self.tracker.update(det_array, np.ascontiguousarray(frame))
        except Exception as exc:
            log.error("vehicle_tracker.update_error", tracker=self._tracker_class, error=str(exc))
            return []

        if raw_tracks is None or raw_tracks.size == 0:
            self._age_all_tracks()
            return []

        tracked_objects: list[TrackedObject] = []
        active_ids: set[int] = set()

        xyxy = raw_tracks.xyxy
        ids = raw_tracks.id.astype(int)
        confs = raw_tracks.conf
        cls_ids = raw_tracks.cls.astype(int)

        for (x1, y1, x2, y2), track_id, conf, cls_id in zip(xyxy, ids, confs, cls_ids):
            track_id = int(track_id)
            cls_id = int(cls_id)

            bbox = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(conf),
                class_id=cls_id,
                class_name=self.target_classes.get(cls_id, str(cls_id)),
            )

            if track_id not in self._track_states:
                self._track_states[track_id] = _TrackState()
            else:
                self._track_states[track_id].on_match()

            active_ids.add(track_id)
            state = self._track_states[track_id]

            tracked_objects.append(TrackedObject(
                track_id=track_id,
                bbox=bbox,
                age=state.age,
                hits=state.hits,
                frames_missing=state.frames_missing,
            ))

        for tid, state in self._track_states.items():
            if tid not in active_ids:
                state.on_miss()

        self._prune_stale_states()
        tracked_objects.sort(key=lambda t: t.track_id)
        return tracked_objects

    def reset(self) -> None:
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        self._track_states.clear()
        log.info("vehicle_tracker.reset", tracker=self._tracker_class)

    @property
    def stats(self) -> dict:
        return {
            "tracker": self._tracker_class,
            "active_tracks": len(self._track_states),
        }

    def _age_all_tracks(self) -> None:
        for state in self._track_states.values():
            state.on_miss()

    def _prune_stale_states(self, max_missing: int = 300) -> None:
        stale = [tid for tid, s in self._track_states.items() if s.frames_missing > max_missing]
        for tid in stale:
            del self._track_states[tid]
        if stale:
            log.debug("vehicle_tracker.pruned_stale", count=len(stale))
