"""
Funnel Service: Business logic orchestrating the 5-Stage SafeChild Vision Pipeline
"""

import time
import base64
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from src.config import DEFAULT_CONF_THRESHOLD, DEFAULT_WARNING_BUFFER_PX
from src.detection.models import DetectionBox
from src.detection.schemas import DetectionItem, BoundingBox
from src.detection.service import (
    ToddlerSafetyEngine,
    GroqVLMGuard,
    get_detection_engine,
    get_vlm_guard,
    build_tts_message,
)
from src.funnel.schemas import (
    FunnelAnalysisResponse,
    SpatialDetail,
    VLMVerificationDetail,
    LatencyBreakdown,
)
from src.funnel.zone_manager import get_room_zones


def _box_to_item(box: DetectionBox) -> DetectionItem:
    return DetectionItem(
        cls_id=box.cls_id,
        cls_name=box.cls_name,
        conf=round(box.conf, 3),
        box=BoundingBox(x1=box.box[0], y1=box.box[1], x2=box.box[2], y2=box.box[3]),
        center=box.center,
        feet_point=box.feet_point,
    )


class FunnelService:
    def __init__(
        self,
        engine: Optional[ToddlerSafetyEngine] = None,
        vlm: Optional[GroqVLMGuard] = None,
    ):
        self.engine = engine or get_detection_engine()
        self.vlm = vlm or get_vlm_guard()
        self.last_annotated_jpeg: Optional[bytes] = None

    def process_image(
        self,
        frame: np.ndarray,
        child_name: str = "Toddler",
        vlm_guardrail: bool = True,
        warning_buffer_px: int = DEFAULT_WARNING_BUFFER_PX,
        camera_id: str = "default_camera",
    ) -> FunnelAnalysisResponse:
        total_t0 = time.time()
        self.engine.warning_buffer_px = warning_buffer_px

        # Stage 1: Edge Toddler Inference
        t0_toddler = time.time()
        toddlers = self.engine.detect_toddlers(frame, conf_thresh=DEFAULT_CONF_THRESHOLD)
        t_toddler_ms = (time.time() - t0_toddler) * 1000

        # Stage 1.5: Toddler Arbiter Validation (VLM only verifies & refines child candidate)
        vlm_ms = 0.0
        arbiter_reason = ""
        is_child_verified = len(toddlers) > 0
        if vlm_guardrail and toddlers:
            t0_vlm = time.time()
            cand_box = toddlers[0].box
            arbiter_res = self.vlm.verify_and_refine_toddler(frame, cand_box)
            vlm_ms = (time.time() - t0_vlm) * 1000
            if not arbiter_res.get("is_child", True):
                # Discard false alarm detection
                toddlers = []
                is_child_verified = False
                arbiter_reason = "VLM Arbiter: Candidate is not a real toddler (false alarm filtered)"
            else:
                is_child_verified = True
                arbiter_reason = arbiter_res.get("reason", "Toddler verified by VLM arbiter")
                if "refined_box" in arbiter_res and arbiter_res["refined_box"]:
                    rb = arbiter_res["refined_box"]
                    toddlers[0].box = tuple(rb)

        # Stage 2: Static Room Hazards (from Setup) or Dynamic YOLO Fallback
        t0_hazard = time.time()
        active_zones = get_room_zones(camera_id)
        if active_zones and active_zones.get("hazards"):
            hazards = [
                DetectionBox(
                    cls_id=i + 1,
                    cls_name=z.get("label", "hazard"),
                    conf=1.0,
                    box=tuple(z["box"]) if isinstance(z.get("box"), (list, tuple)) else (0, 0, 0, 0),
                )
                for i, z in enumerate(active_zones["hazards"])
            ]
        else:
            hazards = self.engine.detect_hazards(frame, conf_thresh=DEFAULT_CONF_THRESHOLD)
        t_hazard_ms = (time.time() - t0_hazard) * 1000

        # Stage 3: Ground-Plane Spatial Fusion (DANGER CALCULATED 100% BY CODE MATHS)
        t0_spatial = time.time()
        severity, alert_events = self.engine.evaluate_safety(toddlers, hazards)
        t_spatial_ms = (time.time() - t0_spatial) * 1000

        primary_hazard = alert_events[0].hazard_name if alert_events else (hazards[0].cls_name if hazards else None)

        # Pure Code-Generated Deterministic Alert Message (No LLM Overwrite)
        if not toddlers:
            alert_msg = "No toddler detected in camera view"
        elif severity == "DANGER":
            alert_msg = f"DANGER: {child_name} is in critical proximity to {primary_hazard.upper()}!"
        elif severity == "WARNING":
            dist = alert_events[0].distance_px if alert_events else self.engine.warning_buffer_px
            alert_msg = f"WARNING: {child_name} is approaching {primary_hazard.upper()} ({int(dist)}px away)"
        else:
            alert_msg = f"SAFE: {child_name} is playing safely away from hazard zones"

        # Spatial Details
        spatial_details = []
        for ev in alert_events:
            if ev.toddler_box and ev.hazard_box:
                spatial_details.append(
                    SpatialDetail(
                        hazard_name=ev.hazard_name,
                        distance_px=ev.distance_px,
                        is_overlapping=ev.distance_px == 0.0,
                        child_feet=ev.toddler_box.feet_point,
                        hazard_box=BoundingBox(
                            x1=ev.hazard_box.box[0],
                            y1=ev.hazard_box.box[1],
                            x2=ev.hazard_box.box[2],
                            y2=ev.hazard_box.box[3],
                        ),
                    )
                )

        # Stage 4: VLM Toddler Arbiter Detail (Reports ONLY Toddler Validation)
        vlm_detail = VLMVerificationDetail(
            enabled=vlm_guardrail,
            child_verified=is_child_verified,
            explanation=arbiter_reason or ("Child verified by VLM" if toddlers else "No child detected"),
            latency_sec=round(vlm_ms / 1000, 2),
        )

        final_severity = severity

        # Stage 5: Alert, TTS & Snapshot
        tts_sentence = build_tts_message(final_severity, child_name, primary_hazard or "")

        if final_severity == "DANGER" and self.engine.save_alerts:
            self.engine._save_alert_snapshot(frame, primary_hazard or "hazard")

        # Draw HUD & cache frame for the /latest-image endpoint (0ms overhead)
        annotated_frame = self.engine.draw_hud_and_annotations(
            frame=frame,
            toddlers=toddlers,
            hazards=hazards,
            overall_severity=final_severity,
            alert_events=alert_events,
            fps=0.0,
        )
        _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.last_annotated_jpeg = buffer.tobytes()

        total_pipeline_ms = (time.time() - total_t0) * 1000

        return FunnelAnalysisResponse(
            success=True,
            severity=final_severity,
            child_name=child_name,
            child_detected=len(toddlers) > 0 or (vlm_detail.child_detected if vlm_detail else False),
            primary_hazard=primary_hazard,
            alert_message=alert_msg,
            tts_text=tts_sentence,
            toddler_count=len(toddlers),
            hazard_count=len(hazards),
            toddlers=[_box_to_item(t) for t in toddlers],
            hazards=[_box_to_item(h) for h in hazards],
            spatial_details=spatial_details,
            vlm_guardrail=vlm_detail,
            latencies=LatencyBreakdown(
                toddler_detector_ms=round(t_toddler_ms, 1),
                hazard_detector_ms=round(t_hazard_ms, 1),
                spatial_fusion_ms=round(t_spatial_ms, 1),
                vlm_guard_ms=round(vlm_ms, 1),
                total_pipeline_ms=round(total_pipeline_ms, 1),
            ),
        )


_funnel_service_instance: Optional[FunnelService] = None


def get_funnel_service() -> FunnelService:
    global _funnel_service_instance
    if _funnel_service_instance is None:
        _funnel_service_instance = FunnelService()
    return _funnel_service_instance
