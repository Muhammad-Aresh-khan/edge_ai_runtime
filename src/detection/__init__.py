"""
Detection Module Package
"""

from .models import DetectionBox, AlertEvent
from .schemas import DetectionItem, SingleModelDetectionResponse, ModelInfoResponse
from .service import (
    ToddlerSafetyEngine,
    GroqVLMGuard,
    get_detection_engine,
    get_vlm_guard,
    speak,
    build_tts_message,
)
from .router import router

__all__ = [
    "DetectionBox",
    "AlertEvent",
    "DetectionItem",
    "SingleModelDetectionResponse",
    "ModelInfoResponse",
    "ToddlerSafetyEngine",
    "GroqVLMGuard",
    "get_detection_engine",
    "get_vlm_guard",
    "speak",
    "build_tts_message",
    "router",
]
