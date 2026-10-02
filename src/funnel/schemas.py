"""
Funnel Schemas (Pydantic Models)
"""

from typing import List, Tuple, Optional
from pydantic import BaseModel, Field

from src.detection.schemas import DetectionItem, BoundingBox


class SpatialDetail(BaseModel):
    hazard_name: str
    distance_px: float
    is_overlapping: bool
    child_feet: Tuple[int, int]
    hazard_box: BoundingBox


class VLMVerificationDetail(BaseModel):
    enabled: bool
    success: bool
    child_detected: bool
    is_real_danger: bool
    verified_severity: str
    verified_hazard: str
    explanation: str
    extra_hazards: List[str] = []
    latency_sec: float


class LatencyBreakdown(BaseModel):
    toddler_detector_ms: float
    hazard_detector_ms: float
    spatial_fusion_ms: float
    vlm_guard_ms: float
    total_pipeline_ms: float


class FunnelAnalysisResponse(BaseModel):
    success: bool = True
    severity: str = Field(..., description="Overall severity: DANGER, WARNING, SAFE, IDLE")
    child_name: str
    child_detected: bool
    primary_hazard: Optional[str] = None
    alert_message: str
    tts_text: str
    toddler_count: int
    hazard_count: int
    toddlers: List[DetectionItem] = []
    hazards: List[DetectionItem] = []
    spatial_details: List[SpatialDetail] = []
    vlm_guardrail: Optional[VLMVerificationDetail] = None
    latencies: LatencyBreakdown
    annotated_image_base64: Optional[str] = None
