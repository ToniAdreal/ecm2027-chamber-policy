"""End-to-end tests for the HTTP policy service using a real local server."""

import http.client
import json
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm_policy.core import ReplayFollowCore  # noqa: E402
from ecm_policy.service import build_server  # noqa: E402
from ecm_policy.trajectory import load_ee_poses_npz  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEMO_NPZ = PACKAGE_ROOT / "data" / "demo_7_ee_poses.npz"


class PolicyServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = load_ee_poses_npz(DEMO_NPZ)
        core = ReplayFollowCore(cls.demo)
        cls.server = build_server(
            "127.0.0.1", 0, core, shutdown_on_close=False
        )
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method: str, path: str, payload=None, raw_body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = raw_body
        headers = {}
        if payload is not None:
            body = json.dumps(payload)
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, json.loads(data.decode("utf-8")) if data else {}

    def test_health(self):
        status, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["ready"])
        self.assertEqual(body["trajectory_steps"], 7566)
        self.assertEqual(body["actions_dim"], 20)

    def test_reset_predict_close_cycle(self):
        status, body = self.request("POST", "/reset", {"task_info": {"task": "ChamberServiceTask"}})
        self.assertEqual(status, 200)
        self.assertEqual(body["phase"], 0)

        status, body = self.request("POST", "/predict", {"observation": {}})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["action"]), 20)
        self.assertTrue(all(np.isfinite(body["action"])))

        status, body = self.request(
            "POST", "/predict",
            {"observation": {"eef_pos": self.demo[200, :, 0:3].tolist()}},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["phase_tracked"])
        self.assertLessEqual(body["phase"], 200)

        status, body = self.request("POST", "/close", {})
        self.assertEqual(status, 200)
        self.assertTrue(body["closed"])

        status, _body = self.request("POST", "/predict", {"observation": {}})
        self.assertEqual(status, 409)

        # reset() re-arms the policy after close.
        status, _body = self.request("POST", "/reset", {})
        self.assertEqual(status, 200)
        status, body = self.request("POST", "/predict", {"observation": {}})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["action"]), 20)

    def test_malformed_and_unknown_requests(self):
        status, _ = self.request("POST", "/predict", {"observation": [1, 2, 3]})
        self.assertEqual(status, 400)
        status, _ = self.request("POST", "/predict", raw_body="{not json")
        self.assertEqual(status, 400)
        status, _ = self.request("GET", "/nope")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
