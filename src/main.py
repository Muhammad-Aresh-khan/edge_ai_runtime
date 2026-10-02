"""
FastAPI Main Application
Includes feature module routers: Funnel, Detection, Training
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.detection.service import get_detection_engine, get_vlm_guard
from src.detection.router import router as detection_router
from src.funnel.router import router as funnel_router
from src.training.router import router as training_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[FastAPI] Pre-warming YOLO models and VLM guardrail...")
    get_detection_engine()
    get_vlm_guard()
    print("[FastAPI] Models pre-warmed and ready to serve!")
    yield
    print("[FastAPI] Shutting down SafeChild Vision API server.")


app = FastAPI(
    title="SafeChild Vision Edge AI API",
    description="""
    ## Modular Enterprise FastAPI Service for Toddler Safety AI
    
    ### Available Modules:
    - **/api/v1/funnel**: End-to-end 5-stage pipeline (Image, Video frame, Stream)
    - **/api/v1/models**: Single model inference (Toddler, Hazard, VLM, Info)
    - **/api/v1/train**: Background asynchronous YOLO training & fine-tuning
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount feature routers under /api/v1
app.include_router(funnel_router, prefix="/api/v1")
app.include_router(detection_router, prefix="/api/v1")
app.include_router(training_router, prefix="/api/v1")


@app.get("/", tags=["Health & Metadata"])
async def root():
    return {
        "service": "SafeChild Vision Edge AI API",
        "version": "2.0.0",
        "status": "HEALTHY",
        "documentation": "/docs",
        "interactive_redoc": "/redoc",
        "endpoints": {
            "funnel_image": "/api/v1/funnel/analyze-image",
            "funnel_frame": "/api/v1/funnel/analyze-frame",
            "models_toddler": "/api/v1/models/toddler/detect",
            "models_hazard": "/api/v1/models/hazard/detect",
            "models_vlm": "/api/v1/models/vlm/verify",
            "models_info": "/api/v1/models/info",
            "train_trigger": "/api/v1/train/trigger",
            "train_jobs": "/api/v1/train/jobs",
        },
    }


@app.get("/health", tags=["Health & Metadata"])
async def health_check():
    return {"status": "ok", "timestamp": time.time()}
