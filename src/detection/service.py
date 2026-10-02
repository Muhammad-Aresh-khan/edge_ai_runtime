"""
Detection Service Layer:
- Dual-YOLO loading & inference (Toddler & Multi-Hazard)
- Ground-plane spatial geometric fusion
- Cognitive Vision LLM Guardrail (Groq Qwen 3.8-27B)
- Native Windows SAPI voice synthesizer
"""

import os
import time
import math
import base64
import json
import threading
import subprocess
from typing import List, Tuple, Dict, Optional, Any
import cv2
import numpy as np
from groq import Groq

from src.config import (
    DEFAULT_TODDLER_MODEL,
    DEFAULT_HAZARD_MODEL,
    DEFAULT_WARNING_BUFFER_PX,
    DEFAULT_ALERT_COOLDOWN_SEC,
    DEFAULT_GROQ_API_KEY,
    DEFAULT_VLM_MODEL,
    ALERTS_DIR,
)
from src.detection.models import DetectionBox, AlertEvent

# Optional winsound for Windows audio alert
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ─── TTS Utility ────────────────────────────────────────────────────────────
def build_tts_message(severity: str, child_name: str, hazard_name: str) -> str:
    name = child_name.strip() or "the child"
    h = hazard_name.replace("_", " ") if hazard_name else "hazard"
    if severity == "DANGER":
        return (
            f"Warning! Warning! {name} is in a danger zone! "
            f"{name} is near the {h}! Please check immediately!"
        )
    elif severity == "WARNING":
        return f"Caution! {name} is approaching the {h}. Please keep an eye on {name}."
    return ""


def speak(text: str):
    if not text:
        return

    def _worker():
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

        try:
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
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=10,
            )
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


# ─── Cognitive Vision LLM Guardrail ─────────────────────────────────────────
class GroqVLMGuard:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or DEFAULT_GROQ_API_KEY
        self.model = model or DEFAULT_VLM_MODEL
        self.client = Groq(api_key=self.api_key)

    def _frame_to_base64(self, frame_bgr: np.ndarray, max_dim: int = 1024) -> str:
        h, w = frame_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame_bgr

        _, buffer = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def verify_safety(
        self,
        frame_bgr: np.ndarray,
        child_name: str,
        yolo_severity: str,
        yolo_hazard: str,
    ) -> Dict[str, Any]:
        name = child_name.strip() or "the child"
        base64_img = self._frame_to_base64(frame_bgr)

        prompt = f"""You are an expert AI Child Safety Guardrail working alongside an edge YOLO object detector.
The edge detector analyzed this room and flagged:
- Child Name: "{name}"
- Preliminary Status: "{yolo_severity}" (Note: if "IDLE", the edge detector could not detect any child)
- Candidate Hazard: "{yolo_hazard}"

Your responsibilities:
1. CHILD DETECTION & RECOVERY:
   - Check if there is a toddler / young child ({name}) in the image.
   - If the edge detector returned IDLE but you see a child in the photo, identify where they are and evaluate their physical safety.
2. SPATIAL REALITY & DANGER CHECK:
   - Is {name} in immediate physical danger?
   - If {name} is sitting/playing safely on the floor and the hazard (door, stairs, window) is in the background or far away, this is a 2D FALSE ALARM. Mark as SAFE.
   - If {name} is touching, climbing, leaning out of, or reaching into a hazard (socket, hot stove, high window, edge of stairs), mark as DANGER or WARNING.
3. OPEN HAZARD DISCOVERY:
   - Spot any other dangerous objects not caught by YOLO (e.g. electrical sockets, sharp knives, boiling water, heaters, loose cables, toxic medicines, tripping rugs).

Return ONLY a valid JSON object matching this schema:
{{
  "child_detected": true or false,
  "is_real_danger": true or false,
  "verified_severity": "DANGER" or "WARNING" or "SAFE" or "IDLE",
  "verified_hazard": "hazard name or none",
  "explanation": "concise 1-2 sentence assessment with physical reasoning",
  "extra_hazards": ["list of any other safety concerns in the room"]
}}
"""
        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - t0
            raw_text = response.choices[0].message.content
            data = json.loads(raw_text)

            child_found = data.get("child_detected", True)
            verified_sev = data.get("verified_severity", yolo_severity).upper()
            if not child_found:
                verified_sev = "IDLE"

            return {
                "success": True,
                "child_detected": child_found,
                "is_real_danger": data.get("is_real_danger", False),
                "verified_severity": verified_sev,
                "verified_hazard": data.get("verified_hazard", yolo_hazard),
                "explanation": data.get("explanation", ""),
                "extra_hazards": data.get("extra_hazards", []),
                "latency_sec": round(elapsed, 2),
                "model": self.model,
            }
        except Exception as e:
            return {
                "success": False,
                "is_real_danger": yolo_severity == "DANGER",
                "verified_severity": yolo_severity,
                "verified_hazard": yolo_hazard,
                "explanation": f"Guardrail unavailable: {str(e)[:100]}",
                "extra_hazards": [],
                "latency_sec": round(time.time() - t0, 2),
                "model": self.model,
            }


# ─── Toddler Safety Edge Engine ─────────────────────────────────────────────
class ToddlerSafetyEngine:
    SEV_DANGER = "DANGER"
    SEV_WARNING = "WARNING"
    SEV_SAFE = "SAFE"
    SEV_IDLE = "IDLE"

    HIGH_RISK_HAZARDS = {"stove", "stairs", "window", "socket", "open door"}

    def __init__(
        self,
        danger_model_path: Optional[str] = None,
        toddler_model_path: Optional[str] = None,
        warning_buffer_px: int = DEFAULT_WARNING_BUFFER_PX,
        alert_cooldown_sec: float = DEFAULT_ALERT_COOLDOWN_SEC,
        save_alerts: bool = True,
        alerts_dir: Optional[str] = None,
        enable_sound: bool = True,
    ):
        self.danger_model_path = danger_model_path or DEFAULT_HAZARD_MODEL
        self.toddler_model_path = toddler_model_path or DEFAULT_TODDLER_MODEL
        self.warning_buffer_px = warning_buffer_px
        self.alert_cooldown_sec = alert_cooldown_sec
        self.save_alerts = save_alerts
        self.alerts_dir = alerts_dir or str(ALERTS_DIR)
        self.enable_sound = enable_sound

        self.last_audio_alert_time = 0.0
        self.last_snapshot_time = 0.0

        if self.save_alerts:
            os.makedirs(self.alerts_dir, exist_ok=True)

        self.model_danger = None
        self.model_toddler = None
        self._load_models()

    def _load_models(self):
        from ultralytics import YOLO
        print(f"[Engine] Loading Danger Zone model: {self.danger_model_path}...")
        self.model_danger = YOLO(self.danger_model_path)
        print(f"[Engine] Loading Toddler Detection model: {self.toddler_model_path}...")
        self.model_toddler = YOLO(self.toddler_model_path)
        print("[Engine] Both models loaded successfully!")

    def _trigger_audio_alert(self, freq: int = 1200, duration_ms: int = 400):
        if not self.enable_sound or not HAS_WINSOUND:
            return
        now = time.time()
        if now - self.last_audio_alert_time < self.alert_cooldown_sec:
            return
        self.last_audio_alert_time = now

        def _beep():
            try:
                winsound.Beep(freq, duration_ms)
            except Exception:
                pass
        threading.Thread(target=_beep, daemon=True).start()

    def _save_alert_snapshot(self, frame: np.ndarray, hazard_name: str):
        if not self.save_alerts:
            return
        now = time.time()
        if now - self.last_snapshot_time < self.alert_cooldown_sec:
            return
        self.last_snapshot_time = now

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        clean_hazard = hazard_name.replace(" ", "_")
        filename = f"alert_{timestamp}_{clean_hazard}.jpg"
        filepath = os.path.join(self.alerts_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"[Engine] Alert snapshot saved to: {filepath}")

    @staticmethod
    def calculate_box_distance(
        toddler_box: Tuple[int, int, int, int],
        hazard_box: Tuple[int, int, int, int],
    ) -> Tuple[float, bool]:
        tx1, ty1, tx2, ty2 = toddler_box
        hx1, hy1, hx2, hy2 = hazard_box

        t_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        ix1 = max(tx1, hx1)
        iy1 = max(ty1, hy1)
        ix2 = min(tx2, hx2)
        iy2 = min(ty2, hy2)

        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            inter_ratio = inter_area / t_area
            cx = (tx1 + tx2) / 2
            cy = (ty1 + ty2) / 2
            feet_x = cx
            feet_y = ty2

            center_inside = (hx1 <= cx <= hx2) and (hy1 <= cy <= hy2)
            feet_inside = (hx1 <= feet_x <= hx2) and (hy1 <= feet_y <= hy2)

            if inter_ratio >= 0.15 or center_inside or feet_inside:
                return 0.0, True

        cx = (tx1 + tx2) / 2
        feet_y = ty2
        dx = max(0, hx1 - cx, cx - hx2)
        dy = max(0, hy1 - feet_y, feet_y - hy2)
        feet_dist = math.hypot(dx, dy)

        edge_dx = max(0, tx1 - hx2, hx1 - tx2)
        edge_dy = max(0, ty1 - hy2, hy1 - ty2)
        edge_dist = math.hypot(edge_dx, edge_dy)

        dist = min(feet_dist, edge_dist if edge_dist > 0 else feet_dist)
        return float(dist), False

    @staticmethod
    def _box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def detect_hazards(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.25,
        imgsz: Optional[int] = None,
        augment: bool = False,
    ) -> List[DetectionBox]:
        h, w = frame.shape[:2]
        target_imgsz = imgsz if imgsz is not None else (1024 if max(h, w) >= 900 else 800)
        base_thresh = min(conf_thresh, 0.10)
        results = self.model_danger(
            frame, imgsz=target_imgsz, conf=base_thresh, augment=augment, verbose=False
        )[0]

        CLASS_MIN_CONF = {
            "socket": 0.10,
            "stove": min(conf_thresh, 0.12),
            "stairs": min(conf_thresh, 0.12),
            "open door": min(conf_thresh, 0.15),
            "closed door": min(conf_thresh, 0.15),
            "window": min(conf_thresh, 0.15),
        }

        hazards = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = results.names.get(cls_id, str(cls_id)).lower()
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            min_required = CLASS_MIN_CONF.get(cls_name, conf_thresh)
            if conf >= min_required:
                hazards.append(DetectionBox(cls_id=cls_id, cls_name=cls_name, conf=conf, box=(x1, y1, x2, y2)))
        return hazards

    def detect_toddlers(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.25,
        imgsz: Optional[int] = None,
        augment: bool = False,
    ) -> List[DetectionBox]:
        target_imgsz = imgsz if imgsz is not None else 640
        base_thresh = min(conf_thresh, 0.10)
        results = self.model_toddler(
            frame, imgsz=target_imgsz, conf=base_thresh, augment=augment, verbose=False
        )[0]

        toddlers = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = results.names.get(cls_id, str(cls_id)).lower()
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if "toddler" in cls_name and "non" not in cls_name:
                if conf >= min(conf_thresh, 0.12):
                    toddlers.append(DetectionBox(cls_id=cls_id, cls_name="toddler", conf=conf, box=(x1, y1, x2, y2)))

        final_toddlers = []
        for t in sorted(toddlers, key=lambda x: x.conf, reverse=True):
            if not any(self._box_iou(t.box, ft.box) > 0.5 for ft in final_toddlers):
                final_toddlers.append(t)
        return final_toddlers

    def evaluate_safety(
        self,
        toddlers: List[DetectionBox],
        hazards: List[DetectionBox],
    ) -> Tuple[str, List[AlertEvent]]:
        if not toddlers:
            return self.SEV_IDLE, []

        overall_severity = self.SEV_SAFE
        alert_events = []

        for toddler in toddlers:
            worst_for_toddler = self.SEV_SAFE
            closest_hazard = None
            min_dist = float("inf")
            is_overlap = False

            for hazard in hazards:
                dist, overlap = self.calculate_box_distance(toddler.box, hazard.box)
                if overlap:
                    worst_for_toddler = self.SEV_DANGER
                    closest_hazard = hazard
                    min_dist = 0.0
                    is_overlap = True
                    break
                elif dist < self.warning_buffer_px:
                    if dist < min_dist:
                        min_dist = dist
                        closest_hazard = hazard
                        worst_for_toddler = self.SEV_WARNING
                elif dist < min_dist:
                    min_dist = dist
                    closest_hazard = hazard

            if is_overlap and closest_hazard:
                event = AlertEvent(
                    severity=self.SEV_DANGER,
                    toddler_box=toddler,
                    hazard_box=closest_hazard,
                    hazard_name=closest_hazard.cls_name,
                    distance_px=0.0,
                    message=f"CRITICAL DANGER: Toddler inside {closest_hazard.cls_name.upper()}!",
                )
                alert_events.append(event)
                overall_severity = self.SEV_DANGER
            elif worst_for_toddler == self.SEV_WARNING and closest_hazard:
                event = AlertEvent(
                    severity=self.SEV_WARNING,
                    toddler_box=toddler,
                    hazard_box=closest_hazard,
                    hazard_name=closest_hazard.cls_name,
                    distance_px=round(min_dist, 1),
                    message=f"WARNING: Toddler approaching {closest_hazard.cls_name.upper()} ({int(min_dist)}px)",
                )
                alert_events.append(event)
                if overall_severity != self.SEV_DANGER:
                    overall_severity = self.SEV_WARNING
            else:
                event = AlertEvent(
                    severity=self.SEV_SAFE,
                    toddler_box=toddler,
                    hazard_box=closest_hazard,
                    hazard_name=closest_hazard.cls_name if closest_hazard else "none",
                    distance_px=round(min_dist, 1) if closest_hazard else -1.0,
                    message="Toddler in safe zone",
                )
                alert_events.append(event)

        return overall_severity, alert_events

    def draw_hud_and_annotations(
        self,
        frame: np.ndarray,
        toddlers: List[DetectionBox],
        hazards: List[DetectionBox],
        overall_severity: str,
        alert_events: List[AlertEvent],
        fps: float = 0.0,
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for h_box in hazards:
            x1, y1, x2, y2 = h_box.box
            color = (0, 140, 255)
            if h_box.cls_name == "closed door":
                color = (180, 180, 180)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{h_box.cls_name} {h_box.conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        for event in alert_events:
            t = event.toddler_box
            if not t:
                continue
            tx1, ty1, tx2, ty2 = t.box

            if event.severity == self.SEV_DANGER:
                box_color = (0, 0, 255)
                status_text = "DANGER!"
            elif event.severity == self.SEV_WARNING:
                box_color = (0, 215, 255)
                status_text = "NEAR HAZARD"
            else:
                box_color = (0, 255, 0)
                status_text = "SAFE"

            cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), box_color, 3)
            label = f"TODDLER [{status_text}] {t.conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (tx1, max(25, ty1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
                cv2.LINE_AA,
            )

            if event.hazard_box and event.severity in (self.SEV_DANGER, self.SEV_WARNING):
                tc = t.center
                hc = event.hazard_box.center
                line_color = (0, 0, 255) if event.severity == self.SEV_DANGER else (0, 215, 255)
                cv2.line(annotated, tc, hc, line_color, 2, cv2.LINE_AA)
                mid_x = int((tc[0] + hc[0]) / 2)
                mid_y = int((tc[1] + hc[1]) / 2)
                dist_str = f"{int(event.distance_px)}px" if event.distance_px > 0 else "OVERLAP"
                cv2.putText(
                    annotated,
                    dist_str,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    line_color,
                    2,
                    cv2.LINE_AA,
                )

        banner_h = 55
        overlay = annotated.copy()
        if overall_severity == self.SEV_DANGER:
            banner_color = (0, 0, 200)
            banner_title = "CRITICAL ALERT: TODDLER IN DANGER ZONE!"
        elif overall_severity == self.SEV_WARNING:
            banner_color = (0, 160, 220)
            banner_title = "CAUTION: TODDLER APPROACHING HAZARD"
        elif overall_severity == self.SEV_SAFE:
            banner_color = (0, 130, 0)
            banner_title = "STATUS: SAFE - NO HAZARDS NEARBY"
        else:
            banner_color = (40, 40, 40)
            banner_title = "STATUS: MONITORING (NO TODDLER DETECTED)"

        cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        cv2.putText(
            annotated,
            banner_title,
            (20, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        stats_str = f"FPS: {fps:.1f} | Toddlers: {len(toddlers)} | Hazards: {len(hazards)}"
        cv2.putText(
            annotated,
            stats_str,
            (max(20, w - 380), 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def process_frame(
        self,
        frame: np.ndarray,
        conf_toddler: float = 0.25,
        conf_danger: float = 0.25,
        fps: float = 0.0,
        augment: bool = False,
    ) -> Tuple[np.ndarray, str, List[AlertEvent]]:
        toddlers = self.detect_toddlers(frame, conf_thresh=conf_toddler, augment=augment)
        hazards = self.detect_hazards(frame, conf_thresh=conf_danger, augment=augment)
        severity, alert_events = self.evaluate_safety(toddlers, hazards)

        if severity == self.SEV_DANGER:
            self._trigger_audio_alert(freq=1500, duration_ms=500)
            hazard_name = alert_events[0].hazard_name if alert_events else "hazard"
            self._save_alert_snapshot(frame, hazard_name)
        elif severity == self.SEV_WARNING:
            self._trigger_audio_alert(freq=900, duration_ms=250)

        annotated = self.draw_hud_and_annotations(
            frame=frame,
            toddlers=toddlers,
            hazards=hazards,
            overall_severity=severity,
            alert_events=alert_events,
            fps=fps,
        )
        return annotated, severity, alert_events


# ─── Singleton Engine Instances ─────────────────────────────────────────────
_engine_instance: Optional[ToddlerSafetyEngine] = None
_vlm_instance: Optional[GroqVLMGuard] = None


def get_detection_engine() -> ToddlerSafetyEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ToddlerSafetyEngine(save_alerts=True, enable_sound=False)
    return _engine_instance


def get_vlm_guard() -> GroqVLMGuard:
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = GroqVLMGuard()
    return _vlm_instance
