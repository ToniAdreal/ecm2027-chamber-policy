#!/usr/bin/env python3
"""Standalone fake evaluator server for the ECM 2027 gRPC policy client.

Serves the official ``ecm27.policy.v1.Policy`` service over a real gRPC
socket, replaying real demo bundles as protocol observations, so the
policy image (``Dockerfile.grpc``) can be verified end to end without
the organizers' evaluator: the container connects to this server, steps
a full episode, and the verdict (steps, episodes, action violations,
gripper agreement) is printed as JSON on shutdown.

The replay logic is reused from the test double
(``tests/test_grpc_client.py::FakePolicyServer``) rather than
duplicated; this tool only adds the socket layer (generic handler, no
generated ``*_pb2_grpc`` module needed) and the verdict reporting.

Usage (from the repo root, with grpcio installed)::

    python3 tools/fake_eval_server.py --port 50051 --max-steps 4107

``--max-steps`` stops the server shortly after that many successful
steps and prints the verdict; without it the server runs until
interrupted (Ctrl-C), then prints the verdict for the steps seen so
far. Exit code is 0 only when at least one episode completed and no
action violation was recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

try:
    import grpc
except ImportError:  # pragma: no cover - exercised on grpc-less hosts
    grpc = None

from ecm_policy import policy_v1_pb2 as pb  # noqa: E402
from test_grpc_client import (  # noqa: E402
    FakePolicyServer,
    _gripper_agreement,
)

SERVICE_NAME = "ecm27.policy.v1.Policy"
DEFAULT_DEMO = REPO_ROOT / "data" / "demo_b2_0_ee_poses.npz"


def create_server(demo_paths, port=0, unix_socket=None):
    """Build a gRPC server replaying ``demo_paths``.

    Listens on TCP ``127.0.0.1:port`` (``port`` 0 picks an ephemeral
    port) or, when ``unix_socket`` is given, on that Unix-domain
    socket path instead. Returns ``(server, fake, stats, bound)``
    where ``bound`` is the TCP port or the socket path. ``port`` 0
    picks an ephemeral port; read the bound port back from ``server``
    via ``server.add_insecure_port`` return value (use ``create_server``
    with an explicit port in tests that need to know it up front).
    """
    if grpc is None:
        raise SystemExit(
            "grpcio is required to run the fake evaluator server "
            "(pip install grpcio==1.85.3 protobuf==7.36.2)"
        )
    fake = FakePolicyServer([str(p) for p in demo_paths])
    stats = {"episodes": 0}

    def get_observation(request, context):
        return fake.get_observation(request)

    def step(request, context):
        resp = fake.step(request)
        stats["episodes"] += sum(resp.terminated)
        return resp

    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SERVICE_NAME,
                {
                    "GetObservation": grpc.unary_unary_rpc_method_handler(
                        get_observation,
                        request_deserializer=pb.Empty.FromString,
                        response_serializer=pb.Transition.SerializeToString,
                    ),
                    "Step": grpc.unary_unary_rpc_method_handler(
                        step,
                        request_deserializer=pb.ActionBatch.FromString,
                        response_serializer=pb.Transition.SerializeToString,
                    ),
                },
            ),
        )
    )
    if unix_socket is not None:
        server.add_insecure_port(f"unix:{unix_socket}")
        bound = unix_socket
    else:
        bound = server.add_insecure_port(f"127.0.0.1:{port}")
    return server, fake, stats, bound


def verdict(fake, stats, demo_paths):
    """Verdict dict from the fake server's recorded run."""
    agreements = {}
    for path in demo_paths:
        agreement = _gripper_agreement(fake, str(path))
        if agreement is not None:
            agreements[Path(path).stem] = round(agreement, 4)
    episodes = stats["episodes"]
    passed = not fake.violations and episodes >= 1
    return {
        "steps": fake.step_successes,
        "step_attempts": fake.step_attempts,
        "episodes": episodes,
        "violations": list(fake.violations),
        "gripper_agreement": agreements,
        "passed": passed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=50051, help="TCP port to listen on (default 50051)"
    )
    parser.add_argument(
        "--unix-socket",
        default=None,
        help="listen on this Unix-domain socket path instead of TCP "
        "(the sandbox transparent proxy breaks grpc TCP loopback: the "
        "client socket is redirected to the egress proxy and never "
        "receives the server SETTINGS frame; raw sockets prove the "
        "server itself is fine)",
    )
    parser.add_argument(
        "--demo",
        action="append",
        default=None,
        help="demo bundle to replay (repeatable; default data/demo_b2_0_ee_poses.npz)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many successful steps, then print the verdict",
    )
    args = parser.parse_args(argv)

    demo_paths = [Path(p) for p in args.demo] if args.demo else [DEFAULT_DEMO]
    for path in demo_paths:
        if not path.exists():
            raise SystemExit(f"demo bundle not found: {path}")

    server, fake, stats, bound = create_server(
        demo_paths, args.port, unix_socket=args.unix_socket
    )
    server.start()
    if args.unix_socket:
        print(f"fake evaluator listening on unix:{args.unix_socket}", flush=True)
    else:
        print(f"fake evaluator listening on 127.0.0.1:{bound}", flush=True)
    print(
        f"replaying {[p.name for p in demo_paths]}"
        + (f", stopping after {args.max_steps} steps" if args.max_steps else ""),
        flush=True,
    )

    stop = threading.Event()

    def _watcher():
        while not stop.is_set():
            if args.max_steps is not None and fake.step_successes >= args.max_steps:
                # Let the final Step reply flush and the client exit
                # before stopping the server underneath it.
                time.sleep(1.0)
                stop.set()
                return
            time.sleep(0.1)

    threading.Thread(target=_watcher, daemon=True).start()
    try:
        while not stop.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    server.stop(grace=2)
    result = verdict(fake, stats, demo_paths)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
