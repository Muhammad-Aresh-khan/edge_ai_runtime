"""
Edge AI Runtime CLI for Toddler Danger Zone Monitoring
Supports 3 Modes:
  1. Image Mode:  py edge_runtime.py --mode image --source path/to/image.jpg
  2. Video Mode:  py edge_runtime.py --mode video --source path/to/video.mp4 [--save]
  3. Webcam Mode: py edge_runtime.py --mode webcam [--cam-id 0]
"""

import os
import sys
import time
import argparse
import cv2
from src.config import (
    DEFAULT_TODDLER_MODEL,
    DEFAULT_HAZARD_MODEL,
    SAMPLES_DIR,
    OUTPUTS_DIR,
)
from src.detection.service import ToddlerSafetyEngine


def run_image_mode(engine: ToddlerSafetyEngine, source_path: str, save_output: bool = True, show_display: bool = True):
    print(f"\n==========================================")
    print(f"  [MODE 1: IMAGE TEST] Processing: {source_path}")
    print(f"==========================================")

    if not os.path.exists(source_path):
        print(f"[Error] Image file not found: {source_path}")
        return

    frame = cv2.imread(source_path)
    if frame is None:
        print(f"[Error] Failed to read image from: {source_path}")
        return

    t0 = time.time()
    annotated_frame, severity, alert_events = engine.process_frame(frame)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"[Result] Inference time: {elapsed_ms:.1f}ms")
    print(f"[Result] Overall Severity: {severity}")
    print(f"[Result] Active Alerts: {len(alert_events)}")
    for i, event in enumerate(alert_events, 1):
        print(f"  {i}. [{event.severity}] {event.message}")

    if save_output:
        base_name = os.path.basename(source_path)
        out_path = str(OUTPUTS_DIR / f"output_{base_name}")
        cv2.imwrite(out_path, annotated_frame)
        print(f"[Saved] Annotated output saved to: {out_path}")

    # Show window if enabled
    if show_display:
        print("\n[Display] Press any key on image window to close...")
        cv2.imshow(f"Toddler Danger Monitor - {os.path.basename(source_path)}", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_mode(engine: ToddlerSafetyEngine, source_path: str, save_video: bool = False):
    print(f"\n==========================================")
    print(f"  [MODE 2: VIDEO TEST] Processing: {source_path}")
    print(f"  Controls: 'q' = Quit, 'p' = Pause, 's' = Save snapshot")
    print(f"==========================================")

    if not os.path.exists(source_path):
        print(f"[Error] Video file not found: {source_path}")
        return

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"[Error] Unable to open video: {source_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if save_video:
        out_path = f"output_{os.path.basename(source_path)}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, input_fps, (width, height))
        print(f"[Video Writer] Output will be saved to: {out_path}")

    prev_time = time.time()
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[Video Finished] End of video stream.")
                break

            now = time.time()
            fps = 1.0 / max(1e-5, (now - prev_time))
            prev_time = now

            annotated, severity, events = engine.process_frame(frame, fps=fps)

            if writer:
                writer.write(annotated)

            cv2.imshow("Toddler Danger Zone Monitor [Video Mode]", annotated)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord('q'):
            print("\n[Quit] User requested stop.")
            break
        elif key == ord('p'):
            paused = not paused
            print("[Control] Paused" if paused else "[Control] Resumed")
        elif key == ord('s'):
            snap_file = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_file, annotated)
            print(f"[Control] Snapshot manually saved: {snap_file}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def run_webcam_mode(engine: ToddlerSafetyEngine, cam_id: int = 0):
    print(f"\n==========================================")
    print(f"  [MODE 3: LIVE WEBCAM] Starting Camera ID: {cam_id}")
    print(f"  Controls: 'q' = Quit, 's' = Save snapshot")
    print(f"==========================================")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {cam_id}. Checking other indices...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[Error] No active webcam found! Make sure camera permissions are enabled.")
            return

    # Optimize resolution for Edge AI runtime
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    print("[Live] Camera running. Press 'q' in video window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to read frame from camera.")
            break

        now = time.time()
        fps = 1.0 / max(1e-5, (now - prev_time))
        prev_time = now

        annotated, severity, events = engine.process_frame(frame, fps=fps)
        cv2.imshow("Toddler Danger Zone Monitor [Live Camera]", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[Quit] Exiting live camera.")
            break
        elif key == ord('s'):
            snap_file = f"webcam_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_file, annotated)
            print(f"[Control] Snapshot saved: {snap_file}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Edge AI Toddler Danger Zone Detection Runtime")
    parser.add_argument("--mode", type=str, choices=["image", "video", "webcam"], default="image",
                        help="Operation mode: 'image', 'video', or 'webcam'")
    parser.add_argument("--source", type=str, default=str(SAMPLES_DIR / "test_sample.jpg"),
                        help="Path to input image or video file")
    parser.add_argument("--cam-id", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--danger-model", type=str, default=DEFAULT_HAZARD_MODEL,
                        help="Path to Danger Zone YOLO model (.pt)")
    parser.add_argument("--toddler-model", type=str, default=DEFAULT_TODDLER_MODEL,
                        help="Path to Toddler YOLO model (.pt)")
    parser.add_argument("--buffer", type=int, default=90,
                        help="Warning proximity buffer in pixels (default: 90px)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated video output")
    parser.add_argument("--no-sound", action="store_true",
                        help="Disable audio buzzer")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable GUI window popup (useful for headless / automated runs)")

    args = parser.parse_args()

    # Initialize Edge AI Engine
    engine = ToddlerSafetyEngine(
        danger_model_path=args.danger_model,
        toddler_model_path=args.toddler_model,
        warning_buffer_px=args.buffer,
        enable_sound=not args.no_sound,
    )

    if args.mode == "image":
        run_image_mode(engine, args.source, show_display=not args.no_display)
    elif args.mode == "video":
        run_video_mode(engine, args.source, save_video=args.save)
    elif args.mode == "webcam":
        run_webcam_mode(engine, cam_id=args.cam_id)


if __name__ == "__main__":
    main()
