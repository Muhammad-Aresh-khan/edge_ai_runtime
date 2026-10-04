"""
Funnel Router: Endpoints for Full 5-Stage Pipeline Execution
"""

import time
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends

from src.config import DEFAULT_WARNING_BUFFER_PX, DEFAULT_CONF_THRESHOLD
from src.funnel.schemas import FunnelAnalysisResponse
from src.funnel.service import FunnelService, get_funnel_service
from src.detection.service import ToddlerSafetyEngine, GroqVLMGuard, get_detection_engine, get_vlm_guard

router = APIRouter(prefix="/funnel", tags=["Full Funnel Pipeline"])


@router.post("/analyze-image", response_model=FunnelAnalysisResponse)
def analyze_image(
    file: UploadFile = File(..., description="Target image file (JPEG/PNG)"),
    child_name: str = Form("Toddler", description="Child's name for personalized alert"),
    vlm_guardrail: bool = Form(True, description="Enable Qwen Vision LLM Cognitive Guardrail"),
    warning_buffer_px: int = Form(DEFAULT_WARNING_BUFFER_PX, description="Proximity threshold in pixels"),
    return_annotated_image: bool = Form(True, description="Return base64 encoded annotated image"),
    service: FunnelService = Depends(get_funnel_service),
):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    return service.process_image(
        frame=frame,
        child_name=child_name,
        vlm_guardrail=vlm_guardrail,
        warning_buffer_px=warning_buffer_px,
        return_annotated_image=return_annotated_image,
    )


@router.post("/analyze-frame")
def analyze_frame_fast(
    file: UploadFile = File(..., description="Streaming video frame"),
    child_name: str = Form("Toddler"),
    vlm_on_danger_only: bool = Form(True, description="Only invoke VLM when YOLO detects DANGER/WARNING"),
    engine: ToddlerSafetyEngine = Depends(get_detection_engine),
    vlm: GroqVLMGuard = Depends(get_vlm_guard),
):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid frame buffer")

    t0 = time.time()
    toddlers = engine.detect_toddlers(frame, conf_thresh=DEFAULT_CONF_THRESHOLD)
    hazards = engine.detect_hazards(frame, conf_thresh=DEFAULT_CONF_THRESHOLD)
    severity, alert_events = engine.evaluate_safety(toddlers, hazards)
    yolo_ms = (time.time() - t0) * 1000

    vlm_result = None
    if vlm_on_danger_only and severity in ("DANGER", "WARNING"):
        hazard_name = alert_events[0].hazard_name if alert_events else "hazard"
        vlm_result = vlm.verify_safety(frame, child_name, severity, hazard_name)
        if not vlm_result.get("is_real_danger", True):
            severity = "SAFE"

    return {
        "severity": severity,
        "toddler_count": len(toddlers),
        "hazard_count": len(hazards),
        "alert_events": [e.message for e in alert_events],
        "vlm_verification": vlm_result,
        "latency_ms": round(yolo_ms, 1),
    }
