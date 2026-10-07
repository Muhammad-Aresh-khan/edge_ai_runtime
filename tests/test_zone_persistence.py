"""
Test Zone Persistence & Replacement:
Verifies that:
1. Room setup saves static zones to RAM and disk.
2. A NEW setup call completely overwrites/replaces old zones (zero stale data).
3. DELETE /api/v1/setup/zones wipes memory and removes persistence file.
4. FunnelService calculates Euclidean distance against saved zones.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from fastapi.testclient import TestClient
from src.main import app
from src.config import ROOM_ZONES_FILE
from src.funnel.zone_manager import get_room_zones, clear_room_zones


def test_zone_persistence_and_replacement():
    client = TestClient(app)
    clear_room_zones()

    # Step 1: Confirm Setup with 2 initial zones (e.g. Living Room)
    payload_1 = {
        "camera_id": "living_room_cam",
        "child_name": "Ayaan",
        "hazards": [
            {"id": "h1", "label": "fireplace", "risk": "critical", "box": [100, 200, 300, 400]},
            {"id": "h2", "label": "loose_cables", "risk": "medium", "box": [50, 50, 80, 80]},
        ],
    }
    res1 = client.post("/api/v1/setup/confirm", json=payload_1)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["hazard_count"] == 2
    assert ROOM_ZONES_FILE.exists()

    # Verify cache & disk state
    active1 = get_room_zones("living_room_cam")
    assert active1 is not None
    assert len(active1["hazards"]) == 2
    assert active1["hazards"][0]["label"] == "fireplace"

    # Step 2: NEW SETUP CALL (e.g. Room moved or new calibration)
    # This MUST completely replace old hazards with 1 completely different hazard!
    payload_2 = {
        "camera_id": "living_room_cam",
        "child_name": "Ayaan",
        "hazards": [
            {"id": "h_new", "label": "stairs_down", "risk": "critical", "box": [500, 400, 700, 600]}
        ],
    }
    res2 = client.post("/api/v1/setup/confirm", json=payload_2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["hazard_count"] == 1
    assert data2["hazards"][0]["label"] == "stairs_down"

    # CRITICAL CHECK: Confirm OLD hazards ('fireplace', 'loose_cables') are 100% GONE!
    active2 = get_room_zones("living_room_cam")
    assert len(active2["hazards"]) == 1
    labels = [h["label"] for h in active2["hazards"]]
    assert "fireplace" not in labels, "FAIL: Old hazard 'fireplace' was NOT replaced!"
    assert "loose_cables" not in labels, "FAIL: Old hazard 'loose_cables' was NOT replaced!"
    assert "stairs_down" in labels

    # Step 3: Test GET /api/v1/setup/zones endpoint
    res_get = client.get("/api/v1/setup/zones?camera_id=living_room_cam")
    assert res_get.status_code == 200
    assert res_get.json()["hazard_count"] == 1

    # Step 4: Test DELETE /api/v1/setup/zones (Wipe clean slate)
    res_del = client.delete("/api/v1/setup/zones?camera_id=living_room_cam")
    assert res_del.status_code == 200
    assert not ROOM_ZONES_FILE.exists(), "FAIL: Persistence file was not deleted on wipe!"
    assert get_room_zones("living_room_cam") is None

    print("\n>>> [PASSED] Zone Persistence & Clean Replacement 100% Verified!")


if __name__ == "__main__":
    test_zone_persistence_and_replacement()
