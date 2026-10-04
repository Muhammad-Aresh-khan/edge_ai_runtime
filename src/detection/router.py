"""
Detection Router: Modular Single-Model Endpoints
"""

import time
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends

from src.config import DEFAULT_CONF_THRESHOLD, DEFAULT_VLM_MODEL
from src.detection.models import DetectionBox
from src.detection.schemas import (
    SingleModelDetectionResponse,
    DetectionItem,
    BoundingBox,
    ModelInfoResponse,
)
from src.detection.service import (
    ToddlerSafetyEngine,
    GroqVLMGuard,
    get_detection_engine,
    get_vlm_guard,
)

router = APIRouter(prefix="/models", tags=["Modular Detection Models"])


def _box_to_item(box: DetectionBox) -> DetectionItem:
    return DetectionItem(
        cls_id=box.cls_id,
        cls_name=box.cls_name,
        conf=round(box.conf, 3),
        box=BoundingBox(x1=box.box[0], y1=box.box[1], x2=box.box[2], y2=box.box[3]),
        center=box.center,
        feet_point=box.feet_point,
    )


@router.post("/toddler/detect", response_model=SingleModelDetectionResponse)
async def detect_toddler_only(
    file: UploadFile = File(..., description="Image to test toddler detection"),
    conf_thresh: float = Form(DEFAULT_CONF_THRESHOLD, ge=0.01, le=1.0),
    engine: ToddlerSafetyEngine = Depends(get_detection_engine),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    t0 = time.time()
    toddlers = engine.detect_toddlers(frame, conf_thresh=conf_thresh)
    latency_ms = (time.time() - t0) * 1000

    return SingleModelDetectionResponse(
        success=True,
        model_name="ToddlerDetection_YOLOv8n",
        detections=[_box_to_item(t) for t in toddlers],
        count=len(toddlers),
        latency_ms=round(latency_ms, 1),
    )


@router.post("/hazard/detect", response_model=SingleModelDetectionResponse)
async def detect_hazards_only(
    file: UploadFile = File(..., description="Image to test danger zone detection"),
    conf_thresh: float = Form(DEFAULT_CONF_THRESHOLD, ge=0.01, le=1.0),
    engine: ToddlerSafetyEngine = Depends(get_detection_engine),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    t0 = time.time()
    hazards = engine.detect_hazards(frame, conf_thresh=conf_thresh)
    latency_ms = (time.time() - t0) * 1000

    return SingleModelDetectionResponse(
        success=True,
        model_name="DangerZone_Hazard_YOLO",
        detections=[_box_to_item(h) for h in hazards],
        count=len(hazards),
        latency_ms=round(latency_ms, 1),
    )


@router.post("/vlm/verify")
async def verify_vlm_only(
    file: UploadFile = File(..., description="Image to evaluate with VLM Guardrail"),
    child_name: str = Form("Toddler"),
    candidate_hazard: str = Form("none"),
    vlm: GroqVLMGuard = Depends(get_vlm_guard),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = vlm.verify_safety(
        frame_bgr=frame,
        child_name=child_name,
        yolo_severity="DANGER" if candidate_hazard != "none" else "IDLE",
        yolo_hazard=candidate_hazard,
    )
    return result


@router.get("/info", response_model=ModelInfoResponse)
async def get_models_info(engine: ToddlerSafetyEngine = Depends(get_detection_engine)):
    return ModelInfoResponse(
        toddler_model={
            "path": engine.toddler_model_path,
            "classes": engine.model_toddler.names if engine.model_toddler else {"0": "non_toddler", "1": "toddler"},
            "status": "LOADED",
        },
        hazard_model={
            "path": engine.danger_model_path,
            "classes": engine.model_danger.names if engine.model_danger else {"0": "door", "1": "socket", "2": "stairs", "3": "stove", "4": "window"},
            "status": "LOADED",
        },
        vlm_guardrail={
            "model": DEFAULT_VLM_MODEL,
            "provider": "Groq LPU (Vision API)",
            "status": "READY",
        },
    )
