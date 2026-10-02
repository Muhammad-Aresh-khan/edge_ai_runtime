"""
Toddler Safety AI Monitor — Streamlit Dashboard
Features:
  - User name input (TTS calls child by name)
  - Mode 1: Image Upload & Analysis
  - Mode 2: Video File Upload & Analysis
  - Mode 3: Live Webcam Real-Time Monitoring
  - pyttsx3 Text-to-Speech personalized voice alerts
  - Real-time alert log with timestamps
"""

import os
import sys
import time
import threading
import tempfile
import queue
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src.config import (
    DEFAULT_TODDLER_MODEL,
    DEFAULT_HAZARD_MODEL,
    ALERTS_DIR,
    DEFAULT_GROQ_API_KEY,
)
from src.detection.service import GroqVLMGuard
DEFAULT_GROQ_KEY = DEFAULT_GROQ_API_KEY

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toddler Safety AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e0e0e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12002b 0%, #1a0a3b 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.3);
}
section[data-testid="stSidebar"] * {
    color: #e2d9f3 !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #4338ca);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
}

/* Status cards */
.status-card {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: 0.05em;
    animation: pulse 2s infinite;
}
.status-danger {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 2px solid #ef4444;
    color: #fecaca;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.4);
}
.status-warning {
    background: linear-gradient(135deg, #78350f, #92400e);
    border: 2px solid #f59e0b;
    color: #fde68a;
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.4);
}
.status-safe {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 2px solid #10b981;
    color: #a7f3d0;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
}
.status-idle {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 2px solid #6366f1;
    color: #c7d2fe;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
}

/* Alert log */
.alert-entry {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 6px 0;
    font-size: 0.88rem;
    border-left: 4px solid;
}
.alert-danger  { background: rgba(239,68,68,0.12); border-color: #ef4444; }
.alert-warning { background: rgba(245,158,11,0.12); border-color: #f59e0b; }
.alert-safe    { background: rgba(16,185,129,0.10); border-color: #10b981; }
.alert-idle    { background: rgba(99,102,241,0.10); border-color: #6366f1; }

/* Metric boxes */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 12px;
    padding: 12px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
}

h1, h2, h3 { color: #e0d7ff !important; }

.logo-text {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0;
}
.logo-sub {
    text-align: center;
    color: #7c6fa0;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(167,139,250,0.2);
}

/* VLM Guardrail Card */
.guard-card {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(79, 70, 229, 0.18));
    border: 1px solid rgba(167, 139, 250, 0.4);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 15px 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
.guard-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
    letter-spacing: 0.05em;
}
.hazard-pill {
    display: inline-block;
    background: rgba(239, 68, 68, 0.2);
    border: 1px solid rgba(239, 68, 68, 0.45);
    color: #fca5a5;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── TTS Engine (Native Windows SAPI via win32com + CoInitialize) ────────────
def speak(text: str):
    """
    Native Windows SAPI voice synthesizer.
    Runs in a daemon thread with pythoncom.CoInitialize() for instant,
    zero-latency, crash-free speech on Windows.
    """
    if not text:
        return

    def _worker():
        # Method 1: Direct native Windows SAPI COM (instant, ~0ms latency)
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = 1
            speaker.Volume = 100
            speaker.Speak(text)
            return
        except Exception:
            pass
        finally:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

        # Method 2: Fallback to System.Speech via PowerShell
        try:
            import subprocess
            safe_text = text.replace("'", " ").replace('"', " ").replace("`", " ")
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Speak('{safe_text}');"
            )
            subprocess.run(
                ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
            )
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def build_tts_message(severity: str, child_name: str, hazard_name: str) -> str:
    name = child_name.strip() or "the child"
    h = hazard_name.replace("_", " ")
    if severity == "DANGER":
        return (
            f"Warning! Warning! {name} is in a danger zone! "
            f"{name} is near the {h}! Please check immediately!"
        )
    elif severity == "WARNING":
        return f"Caution! {name} is approaching the {h}. Please keep an eye on {name}."
    return ""


# ─── Engine loader (cached so it loads only once) ───────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading AI models — please wait...")
def load_engine(danger_model: str, toddler_model: str, buffer: int):
    from src.detection.service import ToddlerSafetyEngine
    return ToddlerSafetyEngine(
        danger_model_path=danger_model,
        toddler_model_path=toddler_model,
        warning_buffer_px=buffer,
        enable_sound=False,   # We use pyttsx3 TTS instead
        save_alerts=True,
        alerts_dir=str(ALERTS_DIR),
    )


@st.cache_resource
def load_vlm_guard():
    return GroqVLMGuard(api_key=DEFAULT_GROQ_KEY)


# ─── Alert log helpers ───────────────────────────────────────────────────────
def _sev_class(sev: str) -> str:
    return {"DANGER": "alert-danger", "WARNING": "alert-warning",
            "SAFE": "alert-safe"}.get(sev, "alert-idle")

def _sev_emoji(sev: str) -> str:
    return {"DANGER": "🔴", "WARNING": "🟡", "SAFE": "🟢"}.get(sev, "⚪")

def add_log(sev: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"time": ts, "severity": sev, "message": msg}
    st.session_state.alert_log.insert(0, entry)
    if len(st.session_state.alert_log) > 50:
        st.session_state.alert_log.pop()


def render_status_card(severity: str, child_name: str, hazard: str):
    name = child_name or "Toddler"
    if severity == "DANGER":
        st.markdown(
            f'<div class="status-card status-danger">🚨 CRITICAL DANGER — {name} near {hazard.upper()}!</div>',
            unsafe_allow_html=True,
        )
    elif severity == "WARNING":
        st.markdown(
            f'<div class="status-card status-warning">⚠️ WARNING — {name} approaching {hazard.upper()}</div>',
            unsafe_allow_html=True,
        )
    elif severity == "SAFE":
        st.markdown(
            f'<div class="status-card status-safe">✅ SAFE — {name} is in a safe zone</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-card status-idle">👁️ MONITORING — No toddler detected</div>',
            unsafe_allow_html=True,
        )


# ─── Session state init ──────────────────────────────────────────────────────
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "last_severity" not in st.session_state:
    st.session_state.last_severity = "IDLE"
if "last_hazard" not in st.session_state:
    st.session_state.last_hazard = ""
if "last_tts_time" not in st.session_state:
    st.session_state.last_tts_time = 0.0
if "stats" not in st.session_state:
    st.session_state.stats = {"frames": 0, "danger": 0, "warning": 0, "safe": 0}


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">🛡️ SafeWatch AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Toddler Danger Zone Monitor</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">👶 Child Profile</div>', unsafe_allow_html=True)
    child_name = st.text_input(
        "Child's Name",
        placeholder="Enter child's name...",
        help="AI will call this name in voice alerts",
        key="child_name_input",
    )

    # Standardized model paths from models/
    danger_model_path = DEFAULT_HAZARD_MODEL
    toddler_model_path = DEFAULT_TODDLER_MODEL
    conf_threshold = 0.15  # Always High (most sensitive) — best for child safety

    st.markdown("---")
    st.markdown('<div class="section-header">🔊 Voice Alert Settings</div>', unsafe_allow_html=True)
    tts_enabled = st.toggle("Enable Voice Alerts", value=True)
    tts_cooldown = st.slider("Repeat Alert Every (sec)", 2, 15, 5,
                             help="Minimum seconds between repeated voice alerts", disabled=not tts_enabled)
    if st.button("🔊 Test Voice Alert", disabled=not tts_enabled, key="btn_test_tts"):
        sample_name = child_name.strip() or "Ali"
        test_msg = build_tts_message("DANGER", sample_name, "window")
        speak(test_msg)
        st.toast(f"🔊 Playing Voice Alert: '{test_msg}'", icon="🔊")

    st.markdown("---")
    st.markdown('<div class="section-header">🤖 AI Vision Guardrail</div>', unsafe_allow_html=True)
    vlm_enabled = st.toggle("Enable Qwen 3.8-27B Guardrail", value=True,
                            help="Cloud VLM on Groq LPU: filters 2D false alarms & detects extra hazards")
    if vlm_enabled:
        st.caption("⚡ Powered by `qwen/qwen3.8-27b` on Groq (~1.5s)")

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Session Stats</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("🔴 Danger", st.session_state.stats["danger"])
        st.metric("✅ Safe", st.session_state.stats["safe"])
    with col_s2:
        st.metric("🟡 Warning", st.session_state.stats["warning"])
        st.metric("📷 Frames", st.session_state.stats["frames"])

    if st.button("🗑️ Clear Alert Log"):
        st.session_state.alert_log = []
        st.session_state.stats = {"frames": 0, "danger": 0, "warning": 0, "safe": 0}
        st.rerun()


# ─── MAIN CONTENT ────────────────────────────────────────────────────────────
st.markdown("## 🛡️ Toddler Safety Monitoring System")
st.caption("Powered by Dual YOLO Models + Spatial Fusion Engine")

# Load engine (all settings hardcoded — optimized for child safety)
try:
    engine = load_engine(danger_model_path, toddler_model_path, 90)
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()


# ─── TABS ────────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_webcam, tab_log = st.tabs([
    "🖼️ Image Upload",
    "🎬 Video Upload",
    "📹 Live Webcam",
    "📋 Alert Log",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE MODE
# ═══════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown("### 🖼️ Upload an Image for Danger Zone Analysis")
    st.caption("Upload any JPG/PNG photo. Both models will analyse it and give a safety verdict.")

    uploaded_file = st.file_uploader(
        "Drop image here or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_uploader",
    )

    if uploaded_file:
        col_orig, col_result = st.columns(2)

        # Read image
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with col_orig:
            st.markdown("**📸 Original Image**")
            st.image(img_rgb, use_container_width=True)

        with st.spinner("🤖 Analysing with AI models (Multi-scale TTA)..."):
            annotated_bgr, severity, events = engine.process_frame(
                img_bgr, conf_toddler=conf_threshold, conf_danger=conf_threshold,
                augment=True
            )
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col_result:
            st.markdown("**🔍 AI Detection Result**")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("---")

        # Pick the most critical event (highest severity) for TTS & status card
        priority = {"DANGER": 3, "WARNING": 2, "SAFE": 1, "IDLE": 0}
        critical_events = [e for e in events if e.severity in ("DANGER", "WARNING")]
        all_events_sorted = sorted(events, key=lambda e: priority.get(e.severity, 0), reverse=True)
        top_event = all_events_sorted[0] if all_events_sorted else None
        hazard = top_event.hazard_name if top_event else ""

        # AI Vision Guardrail (Qwen 3.8-27B on Groq)
        vlm_res = None
        yolo_missed_child = (severity == "IDLE" or len(events) == 0)

        if vlm_enabled:
            with st.spinner("🤖 Consulting Qwen 3.8-27B Vision Guardrail on Groq..."):
                try:
                    guard = load_vlm_guard()
                    vlm_res = guard.verify_safety(img_bgr, child_name, severity, hazard)
                except Exception as e:
                    vlm_res = {"success": False, "explanation": str(e)}

            if vlm_res and vlm_res.get("success"):
                vlm_sev = vlm_res.get("verified_severity", severity)
                child_found = vlm_res.get("child_detected", True)

                # Case 1: YOLO completely missed the child, but AI Guard spotted the child
                if yolo_missed_child and child_found:
                    severity = vlm_sev
                    hazard = vlm_res.get("verified_hazard", "hazard")
                # Case 2: YOLO triggered false alarm, but AI Guard verified child is safe
                elif not vlm_res.get("is_real_danger") and severity in ("DANGER", "WARNING"):
                    severity = "SAFE"
                    hazard = "none"
                # Case 3: AI Guard confirmed real danger
                elif vlm_res.get("is_real_danger"):
                    severity = vlm_sev
                    hazard = vlm_res.get("verified_hazard", hazard)

        # Status card (based on verified severity)
        render_status_card(severity, child_name, hazard)

        # Render Guardrail Assessment Card
        if vlm_res and vlm_res.get("success"):
            st.markdown("#### 🤖 AI Vision Guardrail Assessment (Qwen 3.8-27B on Groq)")
            latency = vlm_res.get("latency_sec", 0.0)
            expl = vlm_res.get("explanation", "")
            extras = vlm_res.get("extra_hazards", [])
            extras_html = "".join([f'<span class="hazard-pill">⚠️ {x}</span>' for x in extras]) if extras else '<span style="opacity:0.6;font-size:0.85rem">None</span>'

            if yolo_missed_child and vlm_res.get("child_detected"):
                badge_msg = "⚠️ CHILD RECOVERED BY AI GUARD"
                badge_color = "#f59e0b"
                status_override_badge = '<div style="color:#fbbf24;font-weight:700;margin-bottom:6px">⚠️ Toddler Recovered: Edge YOLO missed the child, but Qwen Vision Guard detected them and evaluated safety.</div>'
            elif vlm_res.get("is_real_danger"):
                badge_msg = "🚨 DANGER CONFIRMED"
                badge_color = "#ef4444"
                status_override_badge = '<div style="color:#f87171;font-weight:700;margin-bottom:6px">🚨 Danger Confirmed: Physical risk validated by Vision LLM.</div>'
            elif not vlm_res.get("is_real_danger") and critical_events:
                badge_msg = "🛡️ FALSE ALARM FILTERED"
                badge_color = "#10b981"
                status_override_badge = '<div style="color:#34d399;font-weight:700;margin-bottom:6px">🛡️ False Alarm Filtered: 2D perspective overlap dismissed — Child is physically safe.</div>'
            else:
                badge_msg = "✅ SAFE CONFIRMED"
                badge_color = "#10b981"
                status_override_badge = ""

            st.markdown(
                f'<div class="guard-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<span class="guard-badge">⚡ Groq LPU • {latency}s</span>'
                f'<span style="color:{badge_color};font-weight:700;font-size:0.85rem">{badge_msg}</span>'
                f'</div>'
                f'{status_override_badge}'
                f'<div style="font-size:0.95rem;margin:6px 0;"><b>Guard Reasoning:</b> {expl}</div>'
                f'<div style="margin-top:8px;font-size:0.88rem;"><b>Extra Hazards Spotted:</b> {extras_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Events detail — only show DANGER/WARNING entries (filter false positive SAFE events)
        display_events = critical_events if critical_events else ([top_event] if top_event else [])
        if display_events:
            st.markdown("#### 🔎 Detection Details")
            for ev in display_events:
                icon = _sev_emoji(ev.severity)
                dist_str = "Direct overlap" if ev.distance_px == 0 else f"{int(ev.distance_px)}px away"
                h_name = ev.hazard_name.replace("_", " ").title()
                st.markdown(
                    f'<div class="alert-entry {_sev_class(ev.severity)}">'
                    f'{icon} <b>{ev.message}</b><br>'
                    f'<span style="opacity:0.7;font-size:0.82rem">'
                    f'Hazard: <b>{h_name}</b> &nbsp;|&nbsp; Distance: <b>{dist_str}</b>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

        # TTS
        if tts_enabled and severity in ("DANGER", "WARNING"):
            h = hazard
            tts_msg = build_tts_message(severity, child_name, h)
            speak(tts_msg)
            st.toast(f"🔊 Voice Alert: {tts_msg}", icon="🚨")

        # Log it
        msg = events[0].message if events else f"Image scanned — {severity}"
        add_log(severity, msg)
        st.session_state.stats["frames"] += 1
        st.session_state.stats[severity.lower()] = \
            st.session_state.stats.get(severity.lower(), 0) + 1

        # Download button
        _, dl_buf = cv2.imencode(".jpg", annotated_bgr)
        st.download_button(
            "⬇️ Download Annotated Result",
            data=dl_buf.tobytes(),
            file_name=f"safewatch_result_{int(time.time())}.jpg",
            mime="image/jpeg",
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO MODE
# ═══════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("### 🎬 Upload a Video for Danger Zone Analysis")
    st.caption("Upload any MP4/AVI file. Frames are analysed and annotated in real-time.")

    video_file = st.file_uploader(
        "Drop video here or click to browse",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_uploader",
    )

    if video_file:
        # Save to temp file
        suffix = os.path.splitext(video_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        col_va, col_vb = st.columns([2, 1])
        with col_va:
            skip_frames = st.slider("Analyse every N frames", 1, 10, 3,
                                    help="Higher = faster but may miss events")
        with col_vb:
            analyse_btn = st.button("▶️ Start Video Analysis", key="vid_analyse")

        if analyse_btn:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_idx = 0

            progress_bar = st.progress(0, text="Analysing video...")
            status_placeholder = st.empty()
            preview_placeholder = st.empty()

            danger_count = 0
            warning_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                progress = frame_idx / max(total_frames, 1)
                progress_bar.progress(min(progress, 1.0),
                                      text=f"Frame {frame_idx}/{total_frames}")

                if frame_idx % skip_frames != 0:
                    continue

                annotated, severity, events = engine.process_frame(
                    frame, conf_toddler=conf_threshold, conf_danger=conf_threshold,
                    fps=fps_vid
                )
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(annotated_rgb, use_container_width=True,
                                          caption=f"Frame {frame_idx} | Severity: {severity}")

                hazard = events[0].hazard_name if events else ""
                if severity in ("DANGER", "WARNING"):
                    if severity == "DANGER":
                        danger_count += 1
                    else:
                        warning_count += 1
                    add_log(severity, events[0].message if events else f"{severity} detected")
                    if tts_enabled:
                        now = time.time()
                        if now - st.session_state.last_tts_time > tts_cooldown:
                            speak(build_tts_message(severity, child_name, hazard))
                            st.session_state.last_tts_time = now

                st.session_state.stats["frames"] += 1

            cap.release()
            progress_bar.empty()

            st.success(f"✅ Analysis complete! Danger frames: {danger_count} | Warning frames: {warning_count}")
            render_status_card(st.session_state.last_severity, child_name,
                               st.session_state.last_hazard)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — LIVE WEBCAM
# ═══════════════════════════════════════════════════════════════════
with tab_webcam:
    st.markdown("### 📹 Live Webcam Monitoring")
    st.caption("Real-time danger detection using your laptop or external USB camera.")

    col_wc1, col_wc2, col_wc3 = st.columns(3)
    with col_wc1:
        cam_id = st.selectbox("Camera Device", [0, 1, 2], index=0)
    with col_wc2:
        max_frames = st.number_input("Max frames to capture (0 = unlimited)", 0, 9999, 300)
    with col_wc3:
        st.markdown("<br>", unsafe_allow_html=True)
        start_cam = st.button("▶️ Start Live Camera", key="cam_start")

    if not child_name:
        st.warning("⚠️ Please enter the child's name in the sidebar before starting the camera.")

    if start_cam:
        if not child_name:
            st.error("Please enter the child's name first!")
        else:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                st.error(f"❌ Could not open camera {cam_id}. Try a different device index.")
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                frame_slot = st.empty()
                status_slot = st.empty()
                stop_btn = st.button("⏹️ Stop Camera", key="cam_stop")
                info_cols = st.columns(4)

                frame_count = 0
                prev_time = time.time()

                while cap.isOpened():
                    if stop_btn:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break

                    now = time.time()
                    fps_live = 1.0 / max(now - prev_time, 1e-5)
                    prev_time = now
                    frame_count += 1

                    annotated, severity, events = engine.process_frame(
                        frame,
                        conf_toddler=conf_threshold,
                        conf_danger=conf_threshold,
                        fps=fps_live,
                    )
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_slot.image(annotated_rgb, use_container_width=True)

                    hazard = events[0].hazard_name if events else ""
                    with status_slot.container():
                        render_status_card(severity, child_name, hazard)

                    # TTS alert with cooldown
                    if tts_enabled and severity in ("DANGER", "WARNING"):
                        now2 = time.time()
                        if now2 - st.session_state.last_tts_time > tts_cooldown:
                            tts_msg = build_tts_message(severity, child_name, hazard)
                            speak(tts_msg)
                            st.session_state.last_tts_time = now2
                            add_log(severity, events[0].message if events else severity)

                    st.session_state.last_severity = severity
                    st.session_state.last_hazard = hazard
                    st.session_state.stats["frames"] += 1
                    st.session_state.stats[severity.lower()] = \
                        st.session_state.stats.get(severity.lower(), 0) + 1

                    if max_frames and frame_count >= max_frames:
                        break

                cap.release()
                st.success("✅ Camera session ended.")


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — ALERT LOG
# ═══════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown("### 📋 Alert History Log")

    if not st.session_state.alert_log:
        st.info("No alerts recorded yet. Run an analysis from the other tabs.")
    else:
        for entry in st.session_state.alert_log:
            sev = entry["severity"]
            css = _sev_class(sev)
            emoji = _sev_emoji(sev)
            st.markdown(
                f'<div class="alert-entry {css}">'
                f'<span style="opacity:0.6;font-size:0.8rem">{entry["time"]}</span>&nbsp;&nbsp;'
                f'{emoji} <b>[{sev}]</b> {entry["message"]}</div>',
                unsafe_allow_html=True,
            )
