"""
Master CLI Controller for SafeChild Vision AI
Allows running FastAPI, Streamlit, Both simultaneously, or the Test Suite.

Usage:
  python cli.py api          # Start FastAPI Backend Server
  python cli.py ui           # Start Streamlit Web Dashboard
  python cli.py all          # Start BOTH (FastAPI + Streamlit) simultaneously
  python cli.py test         # Run Automated Test Suite
  python cli.py edge [args]  # Run Edge CLI Runtime
  python cli.py              # Interactive Menu
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PYTHON_EXEC = sys.executable


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the production FastAPI backend server."""
    print("\n=======================================================")
    print("  [*] Starting SafeChild Vision FastAPI Server")
    print(f"  [+] API Docs (Swagger): http://localhost:{port}/docs")
    print(f"  [+] Redoc:              http://localhost:{port}/redoc")
    print(f"  [+] Health Check:       http://localhost:{port}/health")
    print("=======================================================\n")
    import uvicorn
    uvicorn.run("src.main:app", host=host, port=port, reload=False)


def run_ui(port: int = 8501):
    """Start the Streamlit Web UI dashboard."""
    print("\n=======================================================")
    print("  [*] Starting SafeChild Vision Streamlit Dashboard")
    print(f"  [+] Web UI: http://localhost:{port}")
    print("=======================================================\n")
    cmd = [
        PYTHON_EXEC,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_both(api_port: int = 8000, ui_port: int = 8501):
    """Start both FastAPI and Streamlit concurrently with graceful shutdown."""
    print("\n=======================================================")
    print("  [*] Starting SafeChild Vision AI — FULL STACK")
    print(f"  [+] FastAPI Backend:   http://localhost:{api_port}/docs")
    print(f"  [+] Streamlit Web UI:  http://localhost:{ui_port}")
    print("  [!] Press CTRL+C anytime to stop both services")
    print("=======================================================\n")

    api_cmd = [
        PYTHON_EXEC,
        "-m",
        "uvicorn",
        "src.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(api_port),
    ]

    ui_cmd = [
        PYTHON_EXEC,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(ui_port),
        "--server.headless",
        "true",
    ]

    p_api = subprocess.Popen(api_cmd, cwd=str(BASE_DIR))
    time.sleep(1.5)  # Let FastAPI initialize first
    p_ui = subprocess.Popen(ui_cmd, cwd=str(BASE_DIR))

    def cleanup(sig=None, frame=None):
        print("\n[*] Shutting down SafeChild Vision services...")
        try:
            p_api.terminate()
        except Exception:
            pass
        try:
            p_ui.terminate()
        except Exception:
            pass
        print("[*] All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            # Check if any process terminated unexpectedly
            if p_api.poll() is not None:
                print(f"[!] FastAPI server exited with code {p_api.returncode}")
                cleanup()
            if p_ui.poll() is not None:
                print(f"[!] Streamlit exited with code {p_ui.returncode}")
                cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


def run_tests():
    """Run the test suite."""
    print("\n=======================================================")
    print("  [*] Running SafeChild Vision Automated Test Suite")
    print("=======================================================\n")
    cmd = [PYTHON_EXEC, "tests/test_pipeline.py"]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_edge(extra_args):
    """Run edge CLI runtime."""
    cmd = [PYTHON_EXEC, "edge_runtime.py"] + extra_args
    subprocess.run(cmd, cwd=str(BASE_DIR))


def interactive_menu():
    """Interactive command selector."""
    print("\n" + "=" * 62)
    print("  🛡️  SafeChild Vision AI — Control Center")
    print("=" * 62)
    print("  [1] Run FastAPI Backend Server    (http://localhost:8000/docs)")
    print("  [2] Run Streamlit Web Dashboard   (http://localhost:8501)")
    print("  [3] Run BOTH (FastAPI + Streamlit Simultaneously)")
    print("  [4] Run Unified Test Suite")
    print("  [5] Run Edge CLI Runner")
    print("  [6] Exit")
    print("=" * 62)

    try:
        choice = input("Select an option [1-6]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    if choice == "1":
        run_api()
    elif choice == "2":
        run_ui()
    elif choice == "3":
        run_both()
    elif choice == "4":
        run_tests()
    elif choice == "5":
        run_edge([])
    elif choice == "6":
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice.")


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower().strip()
        if cmd in ("api", "server", "fastapi"):
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            run_api(port=port)
        elif cmd in ("ui", "streamlit", "web"):
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8501
            run_ui(port=port)
        elif cmd in ("all", "both", "full"):
            run_both()
        elif cmd in ("test", "tests"):
            run_tests()
        elif cmd == "edge":
            run_edge(sys.argv[2:])
        elif cmd in ("-h", "--help", "help"):
            print(__doc__)
        else:
            print(f"Unknown command: '{cmd}'. Use 'api', 'ui', 'all', 'test', or 'edge'.")
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
