"""Persistent inference daemon for agent workflows.

Keeps the model loaded in memory and serves requests via a local
Unix domain socket. Eliminates the 39GB model reload cost on every
CLI invocation.

Usage:
    # Start daemon
    python -m realrestore_cli.daemon start

    # Query daemon (from Rust CLI or another process)
    python -m realrestore_cli.daemon restore --input /path/to/image.png --output /path/to/output.png

    # Stop daemon
    python -m realrestore_cli.daemon stop
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SOCKET_PATH = os.environ.get(
    "REALRESTORE_SOCKET",
    str(Path(tempfile.gettempdir()) / "realrestore.sock"),
)
PID_FILE = str(Path(tempfile.gettempdir()) / "realrestore.pid")
MAX_MSG_SIZE = 1024 * 1024  # 1MB for request/response


def _send_request(request: dict) -> dict:
    """Send a request to the daemon and receive response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(SOCKET_PATH)
        sock.sendall(json.dumps(request).encode() + b"\n")

        # Read response
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        return json.loads(data.decode().strip())
    except ConnectionRefusedError:
        return {"error": "Daemon not running. Start with: realrestore daemon start"}
    except FileNotFoundError:
        return {"error": "Daemon not running. Start with: realrestore daemon start"}
    finally:
        sock.close()


def is_daemon_running() -> bool:
    """Check if daemon is running."""
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def start_daemon(backend: str = "auto", quantize: str = "none") -> None:
    """Start the inference daemon."""
    if is_daemon_running():
        print(json.dumps({"status": "already_running"}))
        return

    # Fork to background
    pid = os.fork()
    if pid > 0:
        # Parent
        print(json.dumps({"status": "started", "pid": pid, "socket": SOCKET_PATH}))
        return

    # Child — become daemon
    os.setsid()

    # Write PID file
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Load pipeline once
    from realrestore_cli.engine import (
        get_device, get_dtype, get_model_path, load_pipeline,
    )

    device = backend if backend != "auto" else get_device()
    dtype = get_dtype(device)
    model_path = get_model_path()

    try:
        pipe = load_pipeline(model_path, device, dtype, quantize)
    except Exception as e:
        with open(PID_FILE + ".error", "w") as f:
            f.write(str(e))
        sys.exit(1)

    # Start socket server
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)

    def handle_client(conn: socket.socket) -> None:
        try:
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            request = json.loads(data.decode().strip())

            if request.get("action") == "restore":
                from realrestore_cli.engine import (
                    TASK_PROMPTS, DEFAULT_NEGATIVE_PROMPT, get_peak_memory_mb,
                )
                from PIL import Image

                start = time.time()
                image = Image.open(request["input"]).convert("RGB")
                task = request.get("task", "auto")
                prompt = request.get("prompt", TASK_PROMPTS.get(task, TASK_PROMPTS["auto"]))

                result = pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    num_inference_steps=request.get("steps", 28),
                    guidance_scale=request.get("guidance_scale", 3.0),
                    seed=request.get("seed", 42),
                    size_level=1024,
                )

                output_path = Path(request["output"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(output_path)

                response = {
                    "status": "success",
                    "output": str(output_path),
                    "elapsed_seconds": round(time.time() - start, 2),
                    "peak_memory_mb": round(get_peak_memory_mb(device), 1),
                }

            elif request.get("action") == "ping":
                response = {"status": "ok", "pid": os.getpid(), "device": device}

            elif request.get("action") == "stop":
                response = {"status": "stopping"}
                conn.sendall(json.dumps(response).encode() + b"\n")
                conn.close()
                _cleanup()
                sys.exit(0)

            else:
                response = {"error": f"Unknown action: {request.get('action')}"}

            conn.sendall(json.dumps(response).encode() + b"\n")
        except Exception as e:
            error_response = {"error": str(e)}
            try:
                conn.sendall(json.dumps(error_response).encode() + b"\n")
            except Exception:
                pass
        finally:
            conn.close()

    def _cleanup() -> None:
        try:
            server.close()
            if os.path.exists(SOCKET_PATH):
                os.unlink(SOCKET_PATH)
            if os.path.exists(PID_FILE):
                os.unlink(PID_FILE)
        except Exception:
            pass

    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(0)))

    while True:
        try:
            conn, _ = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn,))
            thread.daemon = True
            thread.start()
        except Exception:
            break


def stop_daemon() -> None:
    """Stop the running daemon."""
    if not is_daemon_running():
        print(json.dumps({"status": "not_running"}))
        return

    response = _send_request({"action": "stop"})
    print(json.dumps(response))

    # Clean up
    time.sleep(0.5)
    for f in [PID_FILE, SOCKET_PATH]:
        if os.path.exists(f):
            try:
                os.unlink(f)
            except Exception:
                pass


def daemon_restore(
    input_path: str,
    output_path: str,
    task: str = "auto",
    steps: int = 28,
    seed: int = 42,
) -> dict:
    """Send a restore request to the daemon."""
    return _send_request({
        "action": "restore",
        "input": input_path,
        "output": output_path,
        "task": task,
        "steps": steps,
        "seed": seed,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--backend", default="auto")
    start_parser.add_argument("--quantize", default="none")

    subparsers.add_parser("stop")
    subparsers.add_parser("status")

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--input", required=True)
    restore_parser.add_argument("--output", required=True)
    restore_parser.add_argument("--task", default="auto")
    restore_parser.add_argument("--steps", type=int, default=28)
    restore_parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.action == "start":
        start_daemon(args.backend, args.quantize)
    elif args.action == "stop":
        stop_daemon()
    elif args.action == "status":
        running = is_daemon_running()
        print(json.dumps({"running": running, "socket": SOCKET_PATH}))
    elif args.action == "restore":
        result = daemon_restore(args.input, args.output, args.task, args.steps, args.seed)
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "Usage: daemon start|stop|status|restore"}))


if __name__ == "__main__":
    main()
