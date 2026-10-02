"""
Centralized Configuration for Toddler Safety AI Runtime.
Defines directory paths, model weights, API keys, and default thresholds.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Base Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"
ALERTS_DIR = DATA_DIR / "alerts"
TESTS_DIR = BASE_DIR / "tests"

# Ensure runtime directories exist
ALERTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Standard Model Weights Paths
DEFAULT_TODDLER_MODEL = str(MODELS_DIR / "toddler_detector.pt")
DEFAULT_HAZARD_MODEL = str(MODELS_DIR / "hazard_detector.pt")

# Fallback paths for backward compatibility if ever placed in root
if not os.path.exists(DEFAULT_TODDLER_MODEL) and os.path.exists(str(BASE_DIR / "ToddlerDetection.pt")):
    DEFAULT_TODDLER_MODEL = str(BASE_DIR / "ToddlerDetection.pt")

if not os.path.exists(DEFAULT_HAZARD_MODEL) and os.path.exists(str(BASE_DIR / "current_best_window_merged.pt")):
    DEFAULT_HAZARD_MODEL = str(BASE_DIR / "current_best_window_merged.pt")

# Cognitive Vision Guardrail (Groq LPU)
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_VLM_MODEL = os.getenv("VLM_MODEL", "qwen/qwen3.8-27b")

# Edge Detection & Spatial Fusion Thresholds
DEFAULT_CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.15"))
DEFAULT_WARNING_BUFFER_PX = int(os.getenv("WARNING_BUFFER_PX", "90"))
DEFAULT_ALERT_COOLDOWN_SEC = float(os.getenv("ALERT_COOLDOWN_SEC", "2.0"))

# FastAPI Server Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
