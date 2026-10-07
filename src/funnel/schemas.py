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
    child_verified: bool
    explanation: str
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


class HazardZone(BaseModel):
    id: str = "zone_1"
    label: str
    risk: str = "high"
    box: List[int]  # [x1, y1, x2, y2]
    box_normalized: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] (0 to 1000)


class RoomSetupConfirmRequest(BaseModel):
    camera_id: str = "default_camera"
    child_name: str = "Toddler"
    hazards: List[HazardZone]


class RoomSetupResponse(BaseModel):
    success: bool = True
    camera_id: str
    child_name: str
    updated_at: float
    formatted_time: str
    hazard_count: int
    hazards: List[HazardZone]
    message: str
