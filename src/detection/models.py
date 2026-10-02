"""
Detection Domain Models & Data Structures
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DetectionBox:
    cls_id: int
    cls_name: str
    conf: float
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def feet_point(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int(y2))


@dataclass
class AlertEvent:
    severity: str  # "DANGER", "WARNING", "SAFE", "IDLE"
    toddler_box: Optional[DetectionBox]
    hazard_box: Optional[DetectionBox]
    hazard_name: str
    distance_px: float
    message: str
