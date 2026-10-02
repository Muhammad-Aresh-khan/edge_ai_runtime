"""
Detection Schemas (Pydantic Models)
"""

from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionItem(BaseModel):
    cls_id: int
    cls_name: str
    conf: float
    box: BoundingBox
    center: Tuple[int, int]
    feet_point: Tuple[int, int]


class SingleModelDetectionResponse(BaseModel):
    success: bool = True
    model_name: str
    detections: List[DetectionItem]
    count: int
    latency_ms: float


class ModelInfoResponse(BaseModel):
    toddler_model: Dict[str, Any]
    hazard_model: Dict[str, Any]
    vlm_guardrail: Dict[str, Any]
