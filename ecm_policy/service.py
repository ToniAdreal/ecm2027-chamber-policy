"""HTTP policy service exposing the planned ECM submission interface.

The official README for the ECM 2027 sim track states that official
evaluation will use containerized policies with a standardized service
interface -- ``health()``, ``reset(task_info)``, ``predict(observation)``,
``close()`` -- but also states that the transport and message schemas are
still to be published. This service therefore implements those four named
operations as JSON-over-HTTP endpoints with a deliberately small contract,
so the wire format can be re-skinned when the official specification is
released without touching the policy core:

* ``GET  /health``  -> readiness and policy state as JSON.
* ``POST /reset``   -> body ``{"task_info": {...}}`` (optional); starts a
  new episode and resets the policy state. The body may also carry
  ``observation`` (same shape as /predict); when the policy is a
  multi-demo core, the observed initial EE positions select the demo for
  the episode.
* ``POST /predict`` -> body ``{"observation": {...}}`` (optional); returns
  ``{"action": [...], "phase": int, "steps_taken": int}`` (plus
  ``"selected"`` when a multi-demo core is in use). The observation
  may carry ``eef_pos`` (current dual-arm EE positions, shape (2, 3) or
  (1, 2, 3)) to enable closed-loop phase tracking; without it the policy
  replays open-loop.
* ``POST /close``   -> releases the policy and, unless started with
  ``--no-shutdown-on-close``, shuts the server process down after replying.

Only the Python standard library and NumPy are used, and the service needs
no network access at runtime beyond its own listening socket.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .core import ReplayFollowCore
from .multidemo import MultiDemoCore
from .trajectory import load_ee_poses_npz

POLICY_NAME = "twin-teleop-replay-follow"
POLICY_VERSION = "0.3.0"
MAX_BODY_BYTES = 1 << 20  # 1 MiB request cap

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJECTORY_DIR = _PACKAGE_ROOT / "data"


class PolicyService:
    """Thin HTTP adapter over a policy core (one episode at a time).

    Accepts either ReplayFollowCore or MultiDemoCore: both expose
    health()/reset()/predict_with_info()/close(), and the multi-demo core
    additionally takes an observation at reset for demo selection.
    """

    def __init__(self, core):
        self.core = core
        self._lock = threading.Lock()

    def health(self) -> dict:
        state = self.core.health()
        return {
            "policy": POLICY_NAME,
            "version": POLICY_VERSION,
            **state,
        }

    def reset(self, task_info: dict | None, observation: dict | None = None) -> dict:
        with self._lock:
            if isinstance(self.core, MultiDemoCore):
                state = self.core.reset(task_info, observation=observation)
            else:
                state = self.core.reset(task_info)
        return {"status": "ok", **state}

    def predict(self, observation: dict | None) -> dict:
        with self._lock:
            action, info = self.core.predict_with_info(observation)
        response = {
            "action": [float(v) for v in action],
            "phase": info["phase"],
            "steps_taken": info["steps_taken"],
            "phase_tracked": info["phase_tracked"],
        }
        if "selected" in info:
            response["selected"] = info["selected"]
        return response

    def close(self) -> dict:
        with self._lock:
            self.core.close()
        return {"status": "ok", "closed": True}


def make_handler(service: PolicyService, *, shutdown_on_close: bool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # -- helpers ------------------------------------------------------
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError(f"request body too large: {length} bytes")
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("request body must be a JSON object")
            return parsed

        # -- routes ---------------------------------------------------------
        def do_GET(self):  # noqa: N802 (stdlib handler API)
            if self.path.split("?")[0] in ("/health", "/"):
                self._send_json(service.health())
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self):  # noqa: N802 (stdlib handler API)
            path = self.path.split("?")[0]
            try:
                body = self._read_json_body()
                if path == "/reset":
                    task_info = body.get("task_info")
                    if task_info is not None and not isinstance(task_info, dict):
                        raise ValueError("task_info must be a JSON object")
                    observation = body.get("observation")
                    if observation is not None and not isinstance(observation, dict):
                        raise ValueError("observation must be a JSON object")
                    self._send_json(service.reset(task_info, observation))
                elif path == "/predict":
                    observation = body.get("observation")
                    if observation is not None and not isinstance(observation, dict):
                        raise ValueError("observation must be a JSON object")
                    self._send_json(service.predict(observation))
                elif path == "/close":
                    self._send_json(service.close())
                    if shutdown_on_close:
                        threading.Thread(
                            target=self.server.shutdown, daemon=True
                        ).start()
                else:
                    self._send_json(
                        {"error": f"unknown path: {self.path}"}, status=404
                    )
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": f"internal: {exc}"}, status=500)

        def log_message(self, format, *args):  # noqa: A002 - keep stdout clean
            pass

    return Handler


def build_server(host: str, port: int, core, *,
                 shutdown_on_close: bool = True) -> ThreadingHTTPServer:
    service = PolicyService(core)
    handler = make_handler(service, shutdown_on_close=shutdown_on_close)
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ECM 2027 policy service (v3)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--trajectory", default=None,
                        help="single demo .npz bundle (v1 mode)")
    parser.add_argument("--trajectory-dir", default=None,
                        help="directory of demo .npz bundles for multi-demo "
                             "mode (default: package data/ when --trajectory "
                             "is not given)")
    parser.add_argument("--actions-dim", type=int, default=20)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--no-shutdown-on-close", action="store_true",
                        help="keep serving after /close (useful for local tests)")
    args = parser.parse_args(argv)

    if args.trajectory:
        ee_poses = load_ee_poses_npz(args.trajectory)
        core = ReplayFollowCore(
            ee_poses, actions_dim=args.actions_dim, lookahead=args.lookahead
        )
        core_desc = f"trajectory steps={core.num_steps}"
    else:
        demo_dir = args.trajectory_dir or str(DEFAULT_TRAJECTORY_DIR)
        core = MultiDemoCore.from_npz_dir(
            demo_dir, actions_dim=args.actions_dim, lookahead=args.lookahead
        )
        core_desc = f"demos={core.demo_names}"
    server = build_server(
        args.host, args.port, core,
        shutdown_on_close=not args.no_shutdown_on_close,
    )
    host, port = server.server_address[:2]
    print(
        f"{POLICY_NAME} v{POLICY_VERSION} listening on {host}:{port} "
        f"({core_desc}, actions_dim={core.actions_dim if hasattr(core, 'actions_dim') else core.selected_core.actions_dim})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
