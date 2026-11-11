"""Run the FastAPI service and Streamlit UI together for local development."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        [sys.executable, "-m", "streamlit", "run", "app/ui.py", "--server.address", "127.0.0.1"],
    ]
    processes = []
    try:
        print("Starting API; waiting for it to become ready...", flush=True)
        api_process = subprocess.Popen(commands[0], cwd=project_root)
        processes.append(api_process)

        ready_deadline = time.monotonic() + 90
        while time.monotonic() < ready_deadline:
            if api_process.poll() is not None:
                print(
                    f"API stopped during startup (exit code {api_process.returncode}). "
                    "See the API error above.",
                    flush=True,
                )
                return api_process.returncode or 1
            try:
                with urlopen("http://127.0.0.1:8000/", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, URLError):
                time.sleep(0.5)
        else:
            print(
                "API did not become ready within 90 seconds. "
                "Check the API startup output above.",
                flush=True,
            )
            return 1

        print("API is ready. Starting Streamlit UI...", flush=True)
        processes.append(subprocess.Popen(commands[1], cwd=project_root))
        print("API: http://127.0.0.1:8000  |  UI: http://127.0.0.1:8501", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        return next(
            (process.returncode or 0 for process in processes if process.poll() is not None),
            0,
        )
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
