"""
System manager for the Agentic AI RAG application.

Provides a single ``newssearch`` CLI command to start, stop and restart the
entire stack:

    newssearch --start   # launch FastAPI (8000) + Streamlit (8501)
    newssearch --stop    # kill everything
    newssearch --restart # stop then start

PIDs are stored under ``.run/pids.json`` and logs under ``.run/``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]  # repo root
RUN_DIR = ROOT / ".run"
PID_FILE = RUN_DIR / "pids.json"
FASTAPI_LOG = RUN_DIR / "fastapi.log"
STREAMLIT_LOG = RUN_DIR / "streamlit.log"

FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501


def _ensure_run_dir() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def _load_pids() -> dict[str, int]:
    if PID_FILE.exists():
        try:
            return json.loads(PID_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_pids(pids: dict[str, int]) -> None:
    _ensure_run_dir()
    PID_FILE.write_text(json.dumps(pids, indent=2))


def _is_running(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _port_in_use(port: int) -> bool:
    try:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _kill_port(port: int) -> None:
    try:
        import subprocess

        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                try:
                    os.kill(int(line), signal.SIGKILL)
                except OSError:
                    pass
    except Exception:  # pragma: no cover - defensive
        pass

def start() -> int:
    """Start FastAPI + Streamlit in the background."""
    _ensure_run_dir()

    existing = _load_pids()
    fastapi_pid = existing.get("fastapi")
    streamlit_pid = existing.get("streamlit")

    if _is_running(fastapi_pid) and _is_running(streamlit_pid):
        print("✅ System already running (FastAPI PID {}, Streamlit PID {})".format(
            fastapi_pid, streamlit_pid))
        return 0

    # Stop stale processes first
    stop(quiet=True)

    # Give ports a moment to release
    time.sleep(1)

    # ---- FastAPI ----
    fastapi_cmd = [
        sys.executable, "-m", "uvicorn",
        "newpapersearch.api.main:app",
        "--app-dir", str(ROOT),
        "--host", "127.0.0.1",
        "--port", str(FASTAPI_PORT),
    ]
    with FASTAPI_LOG.open("w") as fapi_log:
        fastapi_proc = subprocess.Popen(
            fastapi_cmd,
            cwd=str(ROOT),
            stdout=fapi_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # ---- Streamlit ----
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "frontend" / "streamlit_app.py"),
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
    ]
    with STREAMLIT_LOG.open("w") as st_log:
        streamlit_proc = subprocess.Popen(
            streamlit_cmd,
            cwd=str(ROOT),
            stdout=st_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pids = {"fastapi": fastapi_proc.pid, "streamlit": streamlit_proc.pid}
    _save_pids(pids)

    # Brief wait so the user sees confirmation that processes launched
    time.sleep(2)

    print("🚀 Agentic AI RAG system starting...")
    print(f"   FastAPI  : http://127.0.0.1:{FASTAPI_PORT}  (PID {fastapi_proc.pid})")
    print(f"   Streamlit: http://127.0.0.1:{STREAMLIT_PORT}  (PID {streamlit_proc.pid})")
    print(f"   Logs     : {FASTAPI_LOG}, {STREAMLIT_LOG}")
    print(f"   API docs : http://127.0.0.1:{FASTAPI_PORT}/docs")
    print()
    print("   Run: newssearch --stop   to shut everything down")
    print("   Run: newssearch --restart to restart")
    return 0


def stop(quiet: bool = False) -> int:
    """Stop FastAPI + Streamlit and release ports 8000 / 8501."""
    pids = _load_pids()

    for name in ("fastapi", "streamlit"):
        pid = pids.get(name)
        if _is_running(pid):
            if not quiet:
                print(f"🛑 Stopping {name} (PID {pid})...")
            _kill_process_group(pid)

    # Safety net: kill anything still holding the ports
    _kill_port(FASTAPI_PORT)
    _kill_port(STREAMLIT_PORT)

    time.sleep(1)

    # Clear pid file
    _save_pids({})

    # Verify ports are free
    remaining = []
    for name, port in (("FastAPI", FASTAPI_PORT), ("Streamlit", STREAMLIT_PORT)):
        if _port_in_use(port):
            remaining.append(f"{name}:{port}")

    if not quiet:
        if remaining:
            print(f"⚠️  Ports still occupied: {', '.join(remaining)}")
            print("   You may need to kill them manually.")
        else:
            print("✅ System stopped. Ports 8000 and 8501 released.")
    return 0


def restart() -> int:
    """Stop then start the full stack."""
    stop(quiet=True)
    time.sleep(1)
    return start()


def main() -> None:
    """CLI entry point for the ``newssearch`` command."""
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print("Usage: newssearch --start | --stop | --restart")
        print()
        print("  --start    Start FastAPI (8000) + Streamlit (8501)")
        print("  --stop     Stop all running services")
        print("  --restart  Restart all services")
        sys.exit(0)

    if "--start" in args:
        sys.exit(start())
    if "--stop" in args:
        sys.exit(stop())
    if "--restart" in args:
        sys.exit(restart())

    print("❌ Unknown argument. Use: newssearch --start | --stop | --restart")
    sys.exit(1)


if __name__ == "__main__":
    main()

