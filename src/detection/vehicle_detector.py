"""
src/detection/vehicle_detector.py

Takes model path and target classes as input, and provides a 
`detect` method to detect vehicles in a single frame.
"""
from __future__ import annotations

import numpy as np
import structlog
from ultralytics import YOLO

from src.pipeline.data_types import BoundingBox

log = structlog.get_logger(__name__)

# COCO class IDs -> label, for the vehicle classes we care about.
DEFAULT_TARGET_CLASSES: dict[int, str] = {2: "car", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(
        self,
        model_path: str = "models/detection/yolov8n.pt",
        target_classes: dict[int, str] | None = None,
        min_area_ratio: float = 0.001,
    ) -> None:
        """
        Args:
            model_path: Path to the YOLO model file (.pt).
            target_classes: COCO class_id -> label for classes to report.
                Defaults to {2: "car", 5: "bus", 7: "truck"}.
            min_area_ratio: Minimum bbox area as a fraction of frame area;
                smaller detections are dropped as noise. Matches the
                `w * h > frame_area * 0.001` check in the original script.
        """
        self.target_classes = target_classes or dict(DEFAULT_TARGET_CLASSES)
        self.min_area_ratio = min_area_ratio

        self.model = YOLO(model_path, task="detect")
        log.info("vehicle_detector.model_loaded", path=model_path)

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """
        Detects vehicles in a single frame.

        Returns detections as BoundingBox instances. Empty list if none
        found or the frame is invalid.
        """
        if frame is None or frame.size == 0:
            log.warning("vehicle_detector.empty_frame")
            return []

        frame_area = frame.shape[0] * frame.shape[1]
        detections: list[BoundingBox] = []

        result = self.model(frame, verbose=False)[0]
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in self.target_classes:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if (x2 - x1) * (y2 - y1) <= frame_area * self.min_area_ratio:
                continue

            detections.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=confidence,
                class_id=class_id,
                class_name=self.target_classes[class_id],
            ))

        return detections
