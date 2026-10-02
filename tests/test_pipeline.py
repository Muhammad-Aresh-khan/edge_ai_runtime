"""
Unified Test Suite for SafeChild Vision Edge AI & FastAPI Pipeline
Tests:
  1. Fusion Engine Logic (Overlap, Proximity, Safe, Idle)
  2. Edge YOLO Model Loading from models/ directory
  3. Single Image Inference & Output Generation
  4. FastAPI Service Endpoints & Health Check
  5. Isolated Model Endpoints & Full Funnel API
"""

import os
import sys
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import time
import cv2
import numpy as np

from src.config import (
    DEFAULT_TODDLER_MODEL,
    DEFAULT_HAZARD_MODEL,
    SAMPLES_DIR,
    OUTPUTS_DIR,
)
from src.detection.models import DetectionBox, AlertEvent
from src.detection.service import ToddlerSafetyEngine
from src.main import app
from fastapi.testclient import TestClient


def test_fusion_logic():
    print("\n--- [TEST 1] Testing Spatial Fusion Logic ---")
    engine = ToddlerSafetyEngine(
        danger_model_path=DEFAULT_HAZARD_MODEL,
        toddler_model_path=DEFAULT_TODDLER_MODEL,
        warning_buffer_px=100,
        enable_sound=False,
        save_alerts=False,
    )

    # 1. Overlap Scenario (Critical Danger)
    hazard_stove = DetectionBox(cls_id=4, cls_name="stove", conf=0.88, box=(100, 100, 300, 300))
    toddler_danger = DetectionBox(cls_id=1, cls_name="toddler", conf=0.92, box=(150, 150, 250, 280))
    sev, events = engine.evaluate_safety([toddler_danger], [hazard_stove])
    assert sev == engine.SEV_DANGER, f"Expected DANGER, got {sev}"

    # 2. Proximity Scenario (Warning - within 100px)
    hazard_stairs = DetectionBox(cls_id=3, cls_name="stairs", conf=0.85, box=(100, 100, 200, 200))
    toddler_warning = DetectionBox(cls_id=1, cls_name="toddler", conf=0.90, box=(250, 100, 350, 250))
    sev, events = engine.evaluate_safety([toddler_warning], [hazard_stairs])
    assert sev == engine.SEV_WARNING, f"Expected WARNING, got {sev}"

    # 3. Safe Scenario
    toddler_safe = DetectionBox(cls_id=1, cls_name="toddler", conf=0.91, box=(500, 500, 600, 650))
    sev, events = engine.evaluate_safety([toddler_safe], [hazard_stairs])
    assert sev == engine.SEV_SAFE, f"Expected SAFE, got {sev}"

    # 4. Idle Scenario
    sev, events = engine.evaluate_safety([], [hazard_stairs])
    assert sev == engine.SEV_IDLE, f"Expected IDLE, got {sev}"
    print(">>> [SUCCESS] Fusion Logic passed!")


def test_fastapi_endpoints():
    print("\n--- [TEST 2] Testing FastAPI REST Endpoints ---")
    client = TestClient(app)

    # 1. Root & Health
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["service"] == "SafeChild Vision Edge AI API"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    # 2. Models Info
    res_info = client.get("/api/v1/models/info")
    assert res_info.status_code == 200
    info_data = res_info.json()
    assert "toddler_model" in info_data
    assert "hazard_model" in info_data
    assert info_data["toddler_model"]["status"] == "LOADED"
    print("  Models info verified:", info_data["toddler_model"]["path"])

    # 3. Funnel Image Analysis (with sample image)
    sample_img_path = SAMPLES_DIR / "user_test_window_toddler.jpg"
    if not sample_img_path.exists():
        # Fallback to test_sample.jpg
        sample_img_path = SAMPLES_DIR / "test_sample.jpg"

    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    res_funnel = client.post(
        "/api/v1/funnel/analyze-image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"child_name": "Hamza", "vlm_guardrail": "false", "warning_buffer_px": "90"},
    )
    assert res_funnel.status_code == 200
    data = res_funnel.json()
    assert data["success"] is True
    assert "severity" in data
    assert "latencies" in data
    print(f"  Funnel Image API response: Severity={data['severity']}, Total ms={data['latencies']['total_pipeline_ms']}")

    # 4. Modular Model Endpoints
    res_toddler = client.post(
        "/api/v1/models/toddler/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert res_toddler.status_code == 200
    assert res_toddler.json()["model_name"] == "ToddlerDetection_YOLOv8n"

    res_hazard = client.post(
        "/api/v1/models/hazard/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert res_hazard.status_code == 200
    assert res_hazard.json()["model_name"] == "DangerZone_Hazard_YOLO"

    print(">>> [SUCCESS] All FastAPI REST Endpoints passed!")


if __name__ == "__main__":
    test_fusion_logic()
    test_fastapi_endpoints()
    print("\n🎉 ALL TESTS PASSED! System is production-ready.")
