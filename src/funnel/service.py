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

    def process_image(
        self,
        frame: np.ndarray,
        child_name: str = "Toddler",
        vlm_guardrail: bool = True,
        warning_buffer_px: int = DEFAULT_WARNING_BUFFER_PX,
        return_annotated_image: bool = True,
    ) -> FunnelAnalysisResponse:
        total_t0 = time.time()
        self.engine.warning_buffer_px = warning_buffer_px

        # Stage 1 & 2: Edge YOLO Inference
        t0_toddler = time.time()
        toddlers = self.engine.detect_toddlers(frame, conf_thresh=DEFAULT_CONF_THRESHOLD)
        t_toddler_ms = (time.time() - t0_toddler) * 1000

        t0_hazard = time.time()
        hazards = self.engine.detect_hazards(frame, conf_thresh=DEFAULT_CONF_THRESHOLD)
        t_hazard_ms = (time.time() - t0_hazard) * 1000

        # Stage 3: Ground-Plane Spatial Fusion
        t0_spatial = time.time()
        severity, alert_events = self.engine.evaluate_safety(toddlers, hazards)
        t_spatial_ms = (time.time() - t0_spatial) * 1000

        primary_hazard = alert_events[0].hazard_name if alert_events else (hazards[0].cls_name if hazards else None)
        alert_msg = alert_events[0].message if alert_events else "All areas safe"

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

        # Stage 4: Cognitive Vision LLM Guardrail
        vlm_detail = None
        vlm_ms = 0.0
        final_severity = severity

        if vlm_guardrail:
            t0_vlm = time.time()
            vlm_res = self.vlm.verify_safety(
                frame_bgr=frame,
                child_name=child_name,
                yolo_severity=severity,
                yolo_hazard=primary_hazard or "none",
            )
            vlm_ms = (time.time() - t0_vlm) * 1000

            vlm_detail = VLMVerificationDetail(
                enabled=True,
                success=vlm_res.get("success", False),
                child_detected=vlm_res.get("child_detected", len(toddlers) > 0),
                is_real_danger=vlm_res.get("is_real_danger", severity == "DANGER"),
                verified_severity=vlm_res.get("verified_severity", severity),
                verified_hazard=vlm_res.get("verified_hazard", primary_hazard or "none"),
                explanation=vlm_res.get("explanation", ""),
                extra_hazards=vlm_res.get("extra_hazards", []),
                latency_sec=vlm_res.get("latency_sec", round(vlm_ms / 1000, 2)),
            )

            # Reconcile final severity
            if not vlm_res.get("child_detected", True):
                final_severity = "IDLE"
                alert_msg = "No child detected in camera view"
            elif severity == "DANGER" and not vlm_res.get("is_real_danger", True):
                final_severity = "SAFE"
                alert_msg = f"False Alarm Filtered: {child_name} is playing safely away from hazard"
            elif vlm_res.get("verified_severity") in ("DANGER", "WARNING"):
                final_severity = vlm_res.get("verified_severity")
                alert_msg = f"{final_severity}: {vlm_res.get('explanation', alert_msg)}"
        else:
            vlm_detail = VLMVerificationDetail(
                enabled=False,
                success=True,
                child_detected=len(toddlers) > 0,
                is_real_danger=severity == "DANGER",
                verified_severity=severity,
                verified_hazard=primary_hazard or "none",
                explanation="VLM Guardrail bypassed by user toggle.",
                extra_hazards=[],
                latency_sec=0.0,
            )

        # Stage 5: Alert, TTS & Snapshot
        tts_sentence = build_tts_message(final_severity, child_name, primary_hazard or "")

        if final_severity == "DANGER" and self.engine.save_alerts:
            self.engine._save_alert_snapshot(frame, primary_hazard or "hazard")

        annotated_base64 = None
        if return_annotated_image:
            annotated_frame = self.engine.draw_hud_and_annotations(
                frame=frame,
                toddlers=toddlers,
                hazards=hazards,
                overall_severity=final_severity,
                alert_events=alert_events,
                fps=0.0,
            )
            _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_base64 = base64.b64encode(buffer).decode("utf-8")

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
            annotated_image_base64=annotated_base64,
        )


_funnel_service_instance: Optional[FunnelService] = None


def get_funnel_service() -> FunnelService:
    global _funnel_service_instance
    if _funnel_service_instance is None:
        _funnel_service_instance = FunnelService()
    return _funnel_service_instance
