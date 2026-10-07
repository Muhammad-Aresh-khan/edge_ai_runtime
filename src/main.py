"""
FastAPI Main Application
Includes feature module routers: Funnel, Detection, Training
"""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Optimize PyTorch and C runtime memory allocations for low-RAM cloud instances (Render 512MB)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
try:
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    pass

from src.detection.service import get_detection_engine, get_vlm_guard
from src.detection.router import router as detection_router
from src.funnel.router import router as funnel_router
from src.funnel.setup_router import setup_router
from src.training.router import router as training_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[FastAPI] Server booted with lightweight memory profile (models lazy-loaded).")
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

# Mount feature routers under /api/v1 (Only Main Setup and Full Funnel visible in Swagger docs)
app.include_router(setup_router, prefix="/api/v1")
app.include_router(funnel_router, prefix="/api/v1")


@app.head("/", include_in_schema=False)
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "SafeChild Vision Edge AI API",
        "version": "2.0.0",
        "status": "HEALTHY",
        "documentation": "/docs",
        "endpoints": {
            "setup_room": "/api/v1/setup/room",
            "setup_confirm": "/api/v1/setup/confirm",
            "setup_zones": "/api/v1/setup/zones",
            "funnel_image": "/api/v1/funnel/analyze-image",
            "funnel_frame": "/api/v1/funnel/analyze-frame",
            "latest_image": "/api/v1/funnel/latest-image",
            "viewer": "/viewer",
        },
    }


@app.head("/health", include_in_schema=False)
@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/viewer", response_class=HTMLResponse, include_in_schema=False)
async def interactive_image_viewer():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>SafeChild HUD Image Viewer</title>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: #0b0f19; color: #f8fafc; padding: 24px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
        .container { max-width: 900px; width: 100%; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        h1 { font-size: 1.5rem; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
        .badge { background: #1e293b; color: #94a3b8; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; }
        .card { background: #131d31; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
        .btn-row { display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
        button { background: #2563eb; color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-size: 0.9rem; }
        button:hover { background: #1d4ed8; }
        button.secondary { background: #334155; }
        button.secondary:hover { background: #475569; }
        textarea { width: 100%; height: 90px; background: #0b0f19; border: 1px solid #334155; border-radius: 8px; color: #cbd5e1; padding: 10px; font-size: 0.85rem; resize: vertical; margin-bottom: 12px; font-family: monospace; }
        .img-wrapper { background: #080c14; border: 1px solid #1e293b; border-radius: 12px; padding: 12px; display: flex; justify-content: center; align-items: center; min-height: 350px; }
        img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        .placeholder { color: #64748b; font-size: 0.95rem; text-align: center; }
      </style>
    </head>
    <body>
      <div class="container">
        <header>
          <h1>🛡️ SafeChild Vision HUD Viewer</h1>
          <a href="/docs" style="color: #38bdf8; text-decoration: none; font-size: 0.9rem;">← Back to Swagger Docs</a>
        </header>

        <div class="card">
          <div class="btn-row">
            <button onclick="loadLatestImage()">🔄 Load Latest Analyzed Image</button>
            <button class="secondary" onclick="renderFromInput()">👁️ Render Pasted Base64</button>
            <button class="secondary" onclick="clearViewer()">🗑️ Clear</button>
          </div>
          <textarea id="base64Input" placeholder="Paste your base64 string here (e.g. data:image/jpeg;base64,... or raw /9j/...)" oninput="renderFromInput()"></textarea>
          
          <div class="img-wrapper">
            <img id="displayImg" src="/api/v1/funnel/latest-image" onerror="handleImgError()" alt="Annotated Child HUD Frame">
            <div id="placeholderText" class="placeholder" style="display: none;">No image loaded yet. Click 'Load Latest' or paste Base64 text above.</div>
          </div>
        </div>
      </div>

      <script>
        function renderFromInput() {
          let val = document.getElementById('base64Input').value.trim();
          if (!val) return;
          if (!val.startsWith('data:image')) {
            val = 'data:image/jpeg;base64,' + val;
          }
          const img = document.getElementById('displayImg');
          const placeholder = document.getElementById('placeholderText');
          img.src = val;
          img.style.display = 'block';
          placeholder.style.display = 'none';
        }

        function loadLatestImage() {
          const img = document.getElementById('displayImg');
          const placeholder = document.getElementById('placeholderText');
          img.src = '/api/v1/funnel/latest-image?t=' + new Date().getTime();
          img.style.display = 'block';
          placeholder.style.display = 'none';
        }

        function handleImgError() {
          const img = document.getElementById('displayImg');
          const placeholder = document.getElementById('placeholderText');
          img.style.display = 'none';
          placeholder.style.display = 'block';
        }

        function clearViewer() {
          document.getElementById('base64Input').value = '';
          handleImgError();
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
