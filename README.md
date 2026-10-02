# 🛡️ SafeChild Vision AI: Hybrid Edge AI & Cognitive Vision Runtime

SafeChild Vision is a state-of-the-art hybrid computer vision system engineered for real-time indoor toddler danger zone monitoring. It fuses **dual lightweight YOLO edge detectors**, a **3D ground-plane spatial geometric solver**, and a **cloud-based Vision LLM (Qwen 3.8-27B on Groq LPUs)** to eliminate 2D perspective false alarms and guarantee immediate child safety.

---

## 📁 Industry-Standard Feature-Module Directory Structure

```text
edge_ai_runtime/
├── src/                          # Core application source package
│   ├── config.py                 # Central configuration (paths, thresholds, keys)
│   ├── main.py                   # FastAPI App definition, mounts feature routers
│   │
│   ├── funnel/                   # Feature 1: End-to-end SafeChild Vision funnel
│   │   ├── router.py             # Full funnel endpoints (/funnel/analyze-image, /analyze-frame)
│   │   ├── schemas.py            # Pydantic schemas (FunnelAnalysisResponse, etc.)
│   │   └── service.py            # Funnel business logic (Detection + Spatial + VLM + TTS)
│   │
│   ├── detection/                # Feature 2: Models & computer vision engines
│   │   ├── router.py             # Isolated model endpoints (/models/toddler, /hazard, /vlm)
│   │   ├── schemas.py            # Pydantic schemas for single model detections
│   │   ├── models.py             # Domain models (DetectionBox, AlertEvent)
│   │   └── service.py            # ToddlerSafetyEngine, GroqVLMGuard, SAPI voice synthesizer
│   │
│   └── training/                 # Feature 3: Asynchronous YOLO training & fine-tuning
│       ├── router.py             # Training endpoints (/train/trigger, /train/status)
│       ├── schemas.py            # Pydantic schemas for training jobs
│       └── service.py            # Background training worker & job tracker
│
├── models/                       # Active YOLO weights (No duplicates!)
│   ├── toddler_detector.pt       # YOLOv8n toddler detection model (6.25 MB)
│   └── hazard_detector.pt        # Custom YOLO danger zone detector (20.3 MB)
│
├── data/                         # Runtime media & assets
│   ├── samples/                  # Input test assets (images, clips)
│   ├── outputs/                  # Exported HUD annotated results
│   └── alerts/                   # Saved snapshot incident captures
│
├── tests/                        # Verification & test suite
│   └── test_pipeline.py          # Unified pipeline integration test
│
├── cli.py                        # Master CLI Controller (FastAPI, Streamlit, Both)
├── app.py                        # Streamlit Interactive Web Dashboard
├── edge_runtime.py               # Edge device CLI runner
├── requirements.txt              # Standardized dependencies
├── .env.example                  # Environment configuration template
└── README.md                     # Documentation
```

---

## 🔄 The 5-Stage Hybrid Funnel Pipeline

```text
[Input Frame / Stream]
         │
         ├───► [Stage 1: Toddler Detector (YOLOv8n)] ───► Toddler BBoxes (Conf >= 0.12)
         │
         └───► [Stage 2: Hazard Detector (Custom YOLO)] ─► Window, Stove, Socket, Stairs (Conf >= 0.10)
                                    │
                                    ▼
         [Stage 3: Spatial Ground-Plane Geometric Fusion]
         ├── Child feet contact (cx, y2)
         ├── Substantial body intersection (>= 15%)
         └── Euclidean proximity buffer (default: 90px)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
     [Candidate Safe]                              [Candidate DANGER / WARNING]
            │                                               │
            │                     ┌─────────────────────────┘
            │                     ▼
            │        [Stage 4: Cognitive VLM Guardrail]
            │        (Qwen 3.8-27b on Groq LPU: ~1.2s)
            │        ├── Filters 2D background false alarms
            │        ├── Discovers open-ended hazards (knives, wires)
            │        └── Recovers missed toddlers if edge YOLO outputs IDLE
            │                     │
            └──────────────┬──────┘
                           ▼
         [Stage 5: Decision & Multi-Modal Action]
         ├── Visual HUD Overlays & Connector Lines
         ├── Personalized Voice Warning ("Hamza is near the window!")
         └── Incident Snapshot Storage (data/alerts/)
```

---

## 🚀 Quickstart

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Master CLI Runner (`cli.py`)
Run any service or both with a single command:
```powershell
# Run BOTH FastAPI and Streamlit simultaneously!
python cli.py all

# Run FastAPI Backend only (http://localhost:8000/docs)
python cli.py api

# Run Streamlit Web Dashboard only (http://localhost:8501)
python cli.py ui

# Run Automated Test Suite
python cli.py test

# Interactive Mode (shows interactive menu to pick)
python cli.py
```

### 4. Run Edge Device CLI
```powershell
# Analyze an image
python edge_runtime.py --mode image --source data/samples/user_test_window_toddler.jpg

# Real-time webcam monitoring
python edge_runtime.py --mode webcam --cam-id 0
```

---

## 📡 Complete FastAPI REST API Documentation

The backend service runs on **`http://localhost:8000`** with interactive documentation available at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### Endpoint Directory

| Category | Method | Endpoint | Description |
|---|---|---|---|
| **Health** | `GET` | `/` | Service metadata & available endpoints catalog |
| **Health** | `GET` | `/health` | Uptime & health check |
| **Funnel** | `POST` | `/api/v1/funnel/analyze-image` | Full 5-stage funnel execution on image file |
| **Funnel** | `POST` | `/api/v1/funnel/analyze-frame` | Low-latency endpoint for live streaming video frames |
| **Models** | `POST` | `/api/v1/models/toddler/detect` | Isolated toddler detection (YOLOv8n) |
| **Models** | `POST` | `/api/v1/models/hazard/detect` | Isolated multi-hazard detection (window, stove, socket, etc.) |
| **Models** | `POST` | `/api/v1/models/vlm/verify` | Isolated Qwen 3.8-27B Cognitive Guardrail verification |
| **Models** | `GET` | `/api/v1/models/info` | Loaded weights, classes, and device status |
| **Training**| `POST` | `/api/v1/train/trigger` | Trigger asynchronous YOLO model training / fine-tuning |
| **Training**| `GET` | `/api/v1/train/status/{job_id}` | Query progress and weights path for a training job |
| **Training**| `GET` | `/api/v1/train/jobs` | List all past and active training jobs |

---

### 1. Full Funnel Endpoints (`/api/v1/funnel`)

#### `POST /api/v1/funnel/analyze-image`
Executes the complete 5-stage SafeChild Vision funnel on an uploaded image:
1. Toddler detector (YOLOv8n)
2. Danger zone detector (Custom multi-hazard YOLO)
3. Spatial ground-plane geometric fusion (feet contact + Euclidean distance buffer)
4. Cloud Cognitive VLM Guardrail (Groq Qwen 3.8-27b)
5. Decision logic & personalized SAPI voice speech generation

**Request Type:** `multipart/form-data`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | **Yes** | — | Target room image (`.jpg`, `.png`, `.webp`) |
| `child_name` | string | No | `"Toddler"` | Child's name for personalized alert audio and text |
| `vlm_guardrail`| boolean| No | `true` | Enable/bypass Cloud VLM Cognitive Guardrail |
| `warning_buffer_px` | int | No | `90` | Proximity warning threshold in pixels |
| `return_annotated_image` | boolean | No | `true` | Return base64-encoded JPEG with HUD bounding boxes |

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/funnel/analyze-image" \
  -F "file=@data/samples/user_test_window_toddler.jpg" \
  -F "child_name=Ali" \
  -F "vlm_guardrail=true" \
  -F "warning_buffer_px=90" \
  -F "return_annotated_image=true"
```

**Response JSON (`200 OK`):**
```json
{
  "success": true,
  "severity": "DANGER",
  "child_name": "Ali",
  "child_detected": true,
  "primary_hazard": "window",
  "alert_message": "CRITICAL DANGER: Toddler inside WINDOW!",
  "tts_text": "Warning! Warning! Ali is in a danger zone! Ali is near the window! Please check immediately!",
  "toddler_count": 1,
  "hazard_count": 1,
  "toddlers": [
    {
      "cls_id": 1,
      "cls_name": "toddler",
      "conf": 0.892,
      "box": { "x1": 320, "y1": 210, "x2": 450, "y2": 490 },
      "center": [385, 350],
      "feet_point": [385, 490]
    }
  ],
  "hazards": [
    {
      "cls_id": 5,
      "cls_name": "window",
      "conf": 0.915,
      "box": { "x1": 280, "y1": 150, "x2": 520, "y2": 480 },
      "center": [400, 315],
      "feet_point": [400, 480]
    }
  ],
  "spatial_details": [
    {
      "hazard_name": "window",
      "distance_px": 0.0,
      "is_overlapping": true,
      "child_feet": [385, 490],
      "hazard_box": { "x1": 280, "y1": 150, "x2": 520, "y2": 480 }
    }
  ],
  "vlm_guardrail": {
    "enabled": true,
    "success": true,
    "child_detected": true,
    "is_real_danger": true,
    "verified_severity": "DANGER",
    "verified_hazard": "window",
    "explanation": "The child is climbing onto the windowsill with an open window sash, posing an immediate fall hazard.",
    "extra_hazards": [],
    "latency_sec": 1.12
  },
  "latencies": {
    "toddler_detector_ms": 115.4,
    "hazard_detector_ms": 182.1,
    "spatial_fusion_ms": 0.8,
    "vlm_guard_ms": 1120.0,
    "total_pipeline_ms": 1418.3
  },
  "annotated_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

---

#### `POST /api/v1/funnel/analyze-frame`
Optimized for live video pipelines, webcam loops, or RTSP cameras. Returns lightweight JSON and conditionally triggers the VLM guardrail only when edge models detect candidate hazards.

**Request Type:** `multipart/form-data`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | **Yes** | — | Raw video frame |
| `child_name` | string | No | `"Toddler"` | Child name |
| `vlm_on_danger_only` | boolean | No | `true` | Only invoke VLM if YOLO flags DANGER/WARNING |

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/funnel/analyze-frame" \
  -F "file=@data/samples/test_sample.jpg" \
  -F "child_name=Fatima"
```

**Response JSON (`200 OK`):**
```json
{
  "severity": "SAFE",
  "toddler_count": 1,
  "hazard_count": 0,
  "alert_events": [],
  "vlm_verification": null,
  "latency_ms": 64.2
}
```

---

### 2. Isolated Model Endpoints (`/api/v1/models`)

#### `POST /api/v1/models/toddler/detect`
Runs **ONLY** the YOLOv8n Toddler Detection model on an image.

**Request Type:** `multipart/form-data`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | **Yes** | — | Image file |
| `conf_thresh` | float | No | `0.15` | Minimum confidence threshold [0.01 - 1.0] |

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/models/toddler/detect" \
  -F "file=@data/samples/user_test_window_toddler.jpg" \
  -F "conf_thresh=0.20"
```

**Response JSON (`200 OK`):**
```json
{
  "success": true,
  "model_name": "ToddlerDetection_YOLOv8n",
  "count": 1,
  "latency_ms": 78.4,
  "detections": [
    {
      "cls_id": 1,
      "cls_name": "toddler",
      "conf": 0.89,
      "box": { "x1": 320, "y1": 210, "x2": 450, "y2": 490 },
      "center": [385, 350],
      "feet_point": [385, 490]
    }
  ]
}
```

---

#### `POST /api/v1/models/hazard/detect`
Runs **ONLY** the custom multi-hazard danger zone model (windows, stoves, electrical sockets, stairs, doors).

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/models/hazard/detect" \
  -F "file=@data/samples/user_test_window_toddler.jpg" \
  -F "conf_thresh=0.15"
```

**Response JSON (`200 OK`):**
```json
{
  "success": true,
  "model_name": "DangerZone_Hazard_YOLO",
  "count": 1,
  "latency_ms": 142.1,
  "detections": [
    {
      "cls_id": 5,
      "cls_name": "window",
      "conf": 0.915,
      "box": { "x1": 280, "y1": 150, "x2": 520, "y2": 480 },
      "center": [400, 315],
      "feet_point": [400, 480]
    }
  ]
}
```

---

#### `POST /api/v1/models/vlm/verify`
Directly queries the Qwen 3.8-27B Vision LLM on Groq LPU without running YOLO.

**Request Type:** `multipart/form-data`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | **Yes** | — | Room image |
| `child_name` | string | No | `"Toddler"` | Name of child |
| `candidate_hazard` | string | No | `"none"` | Preliminary hazard name |

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/models/vlm/verify" \
  -F "file=@data/samples/user_test_window_toddler.jpg" \
  -F "child_name=Hamza" \
  -F "candidate_hazard=window"
```

**Response JSON (`200 OK`):**
```json
{
  "success": true,
  "child_detected": true,
  "is_real_danger": true,
  "verified_severity": "DANGER",
  "verified_hazard": "window",
  "explanation": "Hamza is standing directly on the open window frame without supervision.",
  "extra_hazards": [],
  "latency_sec": 1.05,
  "model": "qwen/qwen3.8-27b"
}
```

---

#### `GET /api/v1/models/info`
Returns loaded model architecture, paths, classes, and status.

**Example cURL:**
```bash
curl -X GET "http://localhost:8000/api/v1/models/info"
```

**Response JSON (`200 OK`):**
```json
{
  "toddler_model": {
    "path": "C:\\edge_ai_runtime\\models\\toddler_detector.pt",
    "classes": { "0": "non_toddler", "1": "toddler" },
    "status": "LOADED"
  },
  "hazard_model": {
    "path": "C:\\edge_ai_runtime\\models\\hazard_detector.pt",
    "classes": {
      "0": "closed door",
      "1": "open door",
      "2": "socket",
      "3": "stairs",
      "4": "stove",
      "5": "window"
    },
    "status": "LOADED"
  },
  "vlm_guardrail": {
    "model": "qwen/qwen3.8-27b",
    "provider": "Groq LPU (Vision API)",
    "status": "READY"
  }
}
```

---

### 3. Model Training & Fine-Tuning API (`/api/v1/train`)

#### `POST /api/v1/train/trigger`
Kicks off an asynchronous training/fine-tuning worker in a background thread using Ultralytics YOLO.

**Request Type:** `application/json`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model_type` | string | **Yes** | — | Either `"toddler"` or `"hazard"` |
| `data_yaml` | string | **Yes** | — | Absolute or relative path to `data.yaml` |
| `epochs` | int | No | `30` | Number of training epochs (1-300) |
| `batch_size` | int | No | `16` | Batch size (1-128) |
| `imgsz` | int | No | `640` | Image resolution (320-1280) |
| `base_weights` | string | No | `null` | Optional custom starting weights path |

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/train/trigger" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "hazard",
    "data_yaml": "dataset/data.yaml",
    "epochs": 50,
    "batch_size": 16,
    "imgsz": 640
  }'
```

**Response JSON (`200 OK`):**
```json
{
  "job_id": "train_hazard_20260927_140015_a1b2",
  "model_type": "hazard",
  "status": "QUEUED",
  "start_time": "2026-09-27T14:00:15.123456",
  "epochs": 50,
  "progress": "0/50 epochs",
  "message": "Training job queued in background",
  "weights_path": null,
  "error": null
}
```

---

#### `GET /api/v1/train/status/{job_id}`
Checks the live status, progress, and output weights path of a training job.

**Example cURL:**
```bash
curl -X GET "http://localhost:8000/api/v1/train/status/train_hazard_20260927_140015_a1b2"
```

**Response JSON (`200 OK`):**
```json
{
  "job_id": "train_hazard_20260927_140015_a1b2",
  "model_type": "hazard",
  "status": "COMPLETED",
  "start_time": "2026-09-27T14:00:15.123456",
  "epochs": 50,
  "progress": "100% (50/50 epochs)",
  "message": "Training complete! Weights saved to runs/train/weights/best.pt",
  "weights_path": "C:\\edge_ai_runtime\\runs\\train\\weights\\best.pt",
  "error": null
}
```

---

#### `GET /api/v1/train/jobs`
Returns a list of all queued, active, and completed training jobs.

---

### 4. Health & System Endpoints

- **`GET /`**: Returns system name, version, status, and API directory.
- **`GET /health`**: Returns `{"status": "ok", "timestamp": ...}` for load-balancer health checks.

---

## 🐍 Python Client Code Example

Here is how you can consume the Full Funnel API from any external Python service, camera, or mobile app backend:

```python
import requests

url = "http://localhost:8000/api/v1/funnel/analyze-image"

with open("data/samples/user_test_window_toddler.jpg", "rb") as img:
    files = {"file": img}
    data = {
        "child_name": "Ali",
        "vlm_guardrail": "true",
        "warning_buffer_px": 90,
        "return_annotated_image": "false",
    }
    response = requests.post(url, files=files, data=data)

result = response.json()
print("Severity:    ", result["severity"])
print("Alert:       ", result["alert_message"])
print("Voice Text:  ", result["tts_text"])
print("VLM Verdict: ", result["vlm_guardrail"]["explanation"])
print("Pipeline ms: ", result["latencies"]["total_pipeline_ms"])
```

---

## 🧪 Automated Testing
Run the comprehensive test suite:
```powershell
python cli.py test
```
This tests:
1. Spatial ground-plane geometric fusion (overlap, proximity, safe, idle)
2. Edge YOLO model inference from `models/`
3. All FastAPI endpoints (`/`, `/health`, `/api/v1/models/info`, `/api/v1/funnel/analyze-image`, `/api/v1/models/toddler/detect`)

