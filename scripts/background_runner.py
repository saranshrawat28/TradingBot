"""
Headless Autonomous Supervisor for ApexTrade.
Runs Streamlit UI (localhost:8501) and Paper Lab 24/7 scheduler invisibly in the background.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "storage" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_LOG = LOGS_DIR / "autonomous_service.log"

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}\n"
    with open(SERVICE_LOG, "a", encoding="utf-8") as f:
        f.write(entry)
    try:
        print(entry, end="")
    except Exception:
        pass

def main():
    log("=== ApexTrade Autonomous Background Service Started ===")
    python_exe = sys.executable

    # 1. Start Streamlit Dashboard in Background
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    log("Starting Streamlit Dashboard on http://localhost:8501...")
    streamlit_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )

    # 2. Start Paper Lab Scheduler in Background
    scheduler_cmd = [
        python_exe, "paper_lab_run.py", "--run-scheduler"
    ]
    log("Starting Paper Lab 24/7 Scheduler...")
    scheduler_proc = subprocess.Popen(
        scheduler_cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )

    log(f"Service running successfully. Streamlit PID: {streamlit_proc.pid} | Scheduler PID: {scheduler_proc.pid}")

    try:
        while True:
            time.sleep(30)
            # Health check & Auto-restart if a process died
            if streamlit_proc.poll() is not None:
                log("Streamlit process stopped unexpectedly. Restarting...")
                streamlit_proc = subprocess.Popen(
                    streamlit_cmd,
                    cwd=str(BASE_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )

            if scheduler_proc.poll() is not None:
                log("Scheduler process stopped unexpectedly. Restarting...")
                scheduler_proc = subprocess.Popen(
                    scheduler_cmd,
                    cwd=str(BASE_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
    except (KeyboardInterrupt, SystemExit):
        log("Stopping ApexTrade Background Service...")
        streamlit_proc.terminate()
        scheduler_proc.terminate()
        log("=== ApexTrade Background Service Stopped Cleanly ===")

if __name__ == "__main__":
    main()
