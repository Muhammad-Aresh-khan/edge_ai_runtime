"""
Room Setup Router: Endpoints for One-Time Room Danger Zone Grounding & Persistence
Handles:
  1. POST /api/v1/setup/room      - Upload room photo, detect hazard zones with VLM, save & replace
  2. POST /api/v1/setup/confirm   - Confirm/edit room danger zones, overwrite disk & RAM cache
  3. GET /api/v1/setup/zones      - View currently active danger zones
  4. DELETE /api/v1/setup/zones   - Reset/wipe active zones (clean slate)
"""

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Optional

from src.detection.service import GroqVLMGuard, get_vlm_guard
from src.funnel.schemas import HazardZone, RoomSetupConfirmRequest, RoomSetupResponse
from src.funnel.zone_manager import save_room_zones, get_room_zones, clear_room_zones

setup_router = APIRouter(prefix="/setup", tags=["Room Danger Zone Setup"])


@setup_router.post("/room", response_model=RoomSetupResponse)
def setup_room(
    file: UploadFile = File(..., description="Reference room photo (empty room or wide view)"),
    camera_id: str = Form("default_camera", description="Unique identifier for camera or room"),
    child_name: str = Form("Toddler", description="Child's name for personalized monitoring"),
    vlm: GroqVLMGuard = Depends(get_vlm_guard),
):
    """
    One-time Room Setup:
    1. Parent uploads a room reference photo.
    2. Vision LLM visual grounding identifies static hazards (heaters, stairs, stove, cables, pool) with pixel boxes.
    3. Atomically overwrites data/room_zones.json and updates fast RAM cache.
    4. Guarantees old setup is completely replaced.
    """
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid room image format. Please upload JPEG or PNG.")

    h, w = frame.shape[:2]
    resolution = {"width": w, "height": h}

    # Run VLM Visual Grounding
    detected_hazards = vlm.detect_room_hazard_zones(frame)

    # Save and replace in RAM cache & disk atomically
    saved_config = save_room_zones(
        camera_id=camera_id,
        child_name=child_name,
        hazards=detected_hazards,
        resolution=resolution,
    )

    return RoomSetupResponse(
        success=True,
        camera_id=saved_config["camera_id"],
        child_name=saved_config["child_name"],
        updated_at=saved_config["updated_at"],
        formatted_time=saved_config["formatted_time"],
        hazard_count=saved_config["hazard_count"],
        hazards=[HazardZone(**hz) for hz in saved_config["hazards"]],
        message=f"Room setup successfully saved. {len(detected_hazards)} danger zones active.",
    )


@setup_router.post("/confirm", response_model=RoomSetupResponse)
def confirm_room_setup(
    body: RoomSetupConfirmRequest,
):
    """
    Parent Confirmation / Manual Adjustment:
    Accepts parent-verified or edited danger zones.
    Completely replaces previous room setup on disk and in memory cache.
    """
    hazards_dict_list = [h.model_dump() for h in body.hazards]

    saved_config = save_room_zones(
        camera_id=body.camera_id,
        child_name=body.child_name,
        hazards=hazards_dict_list,
    )

    return RoomSetupResponse(
        success=True,
        camera_id=saved_config["camera_id"],
        child_name=saved_config["child_name"],
        updated_at=saved_config["updated_at"],
        formatted_time=saved_config["formatted_time"],
        hazard_count=saved_config["hazard_count"],
        hazards=[HazardZone(**hz) for hz in saved_config["hazards"]],
        message="Room danger zones confirmed and active for monitoring.",
    )


@setup_router.get("/zones")
def get_active_zones(camera_id: str = "default_camera"):
    """
    Returns active static hazard zones for the given camera.
    Reads from fast RAM cache first, falls back to disk.
    """
    zones = get_room_zones(camera_id)
    if not zones:
        raise HTTPException(
            status_code=404,
            detail=f"No active danger zones found for camera '{camera_id}'. Please run /api/v1/setup/room first.",
        )
    return {
        "success": True,
        "camera_id": zones.get("camera_id", camera_id),
        "child_name": zones.get("child_name", "Toddler"),
        "updated_at": zones.get("updated_at"),
        "formatted_time": zones.get("formatted_time"),
        "hazard_count": zones.get("hazard_count", len(zones.get("hazards", []))),
        "hazards": zones.get("hazards", []),
    }


@setup_router.delete("/zones")
def delete_active_zones(camera_id: Optional[str] = None):
    """
    Wipes danger zones from RAM cache and removes data/room_zones.json.
    Ensures clean slate when re-calibrating or moving cameras.
    """
    clear_room_zones(camera_id)
    return {
        "success": True,
        "message": f"Danger zones successfully cleared for {camera_id or 'all cameras'}. Fresh setup required.",
    }
