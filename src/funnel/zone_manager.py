"""
Room Danger Zone Persistence Manager
Handles saving, updating, caching, and loading static room hazard zones.
Ensures new setup calls cleanly replace old configurations in file & RAM cache.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from src.config import ROOM_ZONES_FILE, DATA_DIR


# In-memory fast cache to avoid disk reads on every live video frame
_ZONE_CACHE: Dict[str, Dict[str, Any]] = {}


def save_room_zones(
    camera_id: str,
    child_name: str,
    hazards: List[Dict[str, Any]],
    resolution: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Saves and completely replaces room zones for a camera.
    Overwrites disk persistence and updates in-memory cache immediately.
    """
    now = time.time()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

    config = {
        "camera_id": camera_id or "default_camera",
        "child_name": child_name.strip() or "Toddler",
        "updated_at": now,
        "formatted_time": formatted_time,
        "reference_resolution": resolution or {"width": 1280, "height": 720},
        "hazard_count": len(hazards),
        "hazards": hazards,
    }

    # 1. Update in-memory cache immediately (Zero latency for subsequent live frames)
    _ZONE_CACHE[config["camera_id"]] = config

    # 2. Persist to disk (overwriting previous configuration cleanly)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = str(ROOM_ZONES_FILE) + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_file, str(ROOM_ZONES_FILE))

    print(f"[ZoneManager] Replaced room zones for '{config['camera_id']}' with {len(hazards)} hazards at {formatted_time}")
    return config


def get_room_zones(camera_id: str = "default_camera") -> Optional[Dict[str, Any]]:
    """
    Retrieves the active danger zones for a camera.
    Reads from fast in-memory cache first, falls back to disk if cache is empty.
    """
    # 1. Fast Cache Hit
    if camera_id in _ZONE_CACHE:
        return _ZONE_CACHE[camera_id]

    # 2. Disk Read on cold start
    if ROOM_ZONES_FILE.exists():
        try:
            with open(str(ROOM_ZONES_FILE), "r", encoding="utf-8") as f:
                data = json.load(f)
            _ZONE_CACHE[data.get("camera_id", camera_id)] = data
            return data
        except Exception as e:
            print(f"[ZoneManager] Error loading {ROOM_ZONES_FILE}: {e}")
            return None

    return None


def clear_room_zones(camera_id: Optional[str] = None):
    """
    Clears active room zones in cache and deletes persistence file.
    """
    global _ZONE_CACHE
    if camera_id:
        _ZONE_CACHE.pop(camera_id, None)
    else:
        _ZONE_CACHE.clear()

    if ROOM_ZONES_FILE.exists():
        try:
            os.remove(str(ROOM_ZONES_FILE))
            print(f"[ZoneManager] Deleted {ROOM_ZONES_FILE}")
        except Exception:
            pass
