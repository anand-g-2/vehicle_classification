"""
Canonical data types for inter-stage communication.
Adding new fields is backward compatible; removing is a breaking change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Detection contract

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

# Tracking contract

@dataclass
class TrackedObject:
    track_id: int
    bbox: BoundingBox
    age: int = 0                # frames since first detection
    hits: int = 0                # total confirmed detections
    frames_missing: int = 0      # consecutive frames without detection

# Calibration / classification contract

@dataclass
class LaneLine:
    """A calibration line drawn across a lane, used both to detect a
    vehicle crossing it and as the size reference for SV/MV/HV
    classification (a vehicle's on-screen size relative to this line's
    length)."""
    start: tuple[int, int]
    end: tuple[int, int]
    is_curved: bool = False   # widens the crossing tolerance for an angled lane

    @property
    def length(self) -> float:
        (x1, y1), (x2, y2) = self.start, self.end
        return math.hypot(x2 - x1, y2 - y1)
