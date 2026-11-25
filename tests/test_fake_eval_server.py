"""Real-socket tests for the fake evaluator server (ECM 2027 milestone 19).

The in-process tests in ``test_grpc_client.py`` exercise the policy
logic through real protobuf serialization but skip the gRPC socket
itself. These tests close that gap: they start
``tools/fake_eval_server.py`` on a real socket and drive it with a
real ``grpc`` channel, including the module entrypoint
(``python -m ecm_policy.grpc_client`` with ``ECM_MAX_STEPS``).

The socket is a Unix-domain socket, not TCP loopback: on this dev VM
the sandbox transparent proxy redirects a grpc client's TCP loopback
connection to the egress proxy (the client socket's peer shows up as
the proxy's ``[fd8b:4f84:7d32:99::1]:3128`` and the handshake dies
waiting for the server SETTINGS frame), while raw sockets prove the
server itself serves HTTP/2 fine. The transport exercised here
(HTTP/2 framing, protobuf serialization) is identical; only the
endpoint differs. The production evaluator runs on its own host, so
this is a dev-host limitation, not a policy defect.

Skipped where ``grpcio`` is not installed, mirroring the existing
protobuf gating in ``test_grpc_client.py``.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PACKAGE_ROOT / "tools"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(TOOLS_DIR))

try:
    import grpc  # noqa: F401

    from ecm_policy import policy_v1_pb2 as pb  # noqa: E402
    from ecm_policy.grpc_client import (  # noqa: E402
        ChamberGrpcPolicy,
        PolicyClient,
        run_policy_loop,
    )
    from ecm_policy.multidemo import MultiDemoCore  # noqa: E402
    from fake_eval_server import create_server, verdict  # noqa: E402

    _GRPC_AVAILABLE = True
except ImportError:  # pragma: no cover - hosts without grpcio
    _GRPC_AVAILABLE = False
    pb = None

DATA_DIR = PACKAGE_ROOT / "data"
DEMO_B2_0 = DATA_DIR / "demo_b2_0_ee_poses.npz"


def _policy_factory():
    return MultiDemoCore.from_npz_dir(
        DATA_DIR,
        actions_dim=20,
        action_quat_order="xyzw",
        lookahead=2,
        phase_advance_cap=4,
    )


@unittest.skipUnless(_GRPC_AVAILABLE, "grpcio not installed")
class FakeEvalServerSocketTests(unittest.TestCase):
    def test_real_socket_short_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            sock = str(Path(tmp) / "fake.sock")
            server, fake, stats, _bound = create_server(
                [DEMO_B2_0], unix_socket=sock
            )
            server.start()
            try:
                channel = grpc.insecure_channel(f"unix://{sock}")
                grpc.channel_ready_future(channel).result(timeout=30)
                client = PolicyClient(channel)
                policy = ChamberGrpcPolicy(_policy_factory)
                result = run_policy_loop(client, policy, max_steps=50)
                self.assertEqual(result["steps"], 50)
                self.assertEqual(result["episodes"], 0)  # 50 steps < 4106
                self.assertEqual(fake.violations, [])
                self.assertEqual(fake.step_successes, 50)
            finally:
                server.stop(grace=0)

    def test_main_entrypoint_with_max_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            sock = str(Path(tmp) / "fake.sock")
            server, fake, stats, _bound = create_server(
                [DEMO_B2_0], unix_socket=sock
            )
            server.start()
            try:
                env = dict(os.environ)
                env["ECM27_EVAL_ENDPOINT"] = f"unix://{sock}"
                env["ECM_MAX_STEPS"] = "50"
                proc = subprocess.run(
                    [sys.executable, "-m", "ecm_policy.grpc_client"],
                    cwd=str(PACKAGE_ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                self.assertEqual(fake.step_successes, 50)
                self.assertEqual(fake.violations, [])
            finally:
                server.stop(grace=0)

    def test_verdict_fails_without_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            sock = str(Path(tmp) / "fake.sock")
            server, fake, stats, _bound = create_server(
                [DEMO_B2_0], unix_socket=sock
            )
            server.start()
            try:
                channel = grpc.insecure_channel(f"unix://{sock}")
                grpc.channel_ready_future(channel).result(timeout=30)
                client = PolicyClient(channel)
                policy = ChamberGrpcPolicy(_policy_factory)
                run_policy_loop(client, policy, max_steps=10)
                result = verdict(fake, stats, [DEMO_B2_0])
                self.assertFalse(result["passed"])  # no episode completed yet
                self.assertEqual(result["episodes"], 0)
                self.assertEqual(result["violations"], [])
            finally:
                server.stop(grace=0)


if __name__ == "__main__":
    unittest.main()
