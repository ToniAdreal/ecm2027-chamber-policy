"""Tests for the official gRPC policy client (ECM 2027 milestone 15).

The evaluator server is faked in-process: a ``FakePolicyServer`` replays
real demo bundles as protocol observations, and a ``FakeChannel`` passes
requests and responses through the real protobuf serialization, so the
tests exercise the actual wire format. The only part not exercised here
is the gRPC socket itself (``grpcio`` is not installable on this host:
PyPI is unreachable behind the proxy), which belongs to upstream's
``PolicyClient`` transport, not to this module's policy logic.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import google.protobuf  # noqa: F401  (system python has it; venv does not)
    from ecm_policy import policy_v1_pb2 as pb  # noqa: E402
    from ecm_policy.grpc_client import (  # noqa: E402
        ACTION_DIM,
        FIELD_SHAPES,
        ChamberGrpcPolicy,
        PolicyClient,
        core_action_to_19,
        decode_observation,
        run_policy_loop,
        validate_action_19,
    )

    _PROTOBUF_AVAILABLE = True
except ImportError:  # pragma: no cover - venv python without protobuf
    _PROTOBUF_AVAILABLE = False
    pb = None

from ecm_policy.multidemo import MultiDemoCore  # noqa: E402
from ecm_policy.trajectory import load_ee_poses_npz, load_gripper_pos_npz  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
DEMO_B2_0 = DATA_DIR / "demo_b2_0_ee_poses.npz"
DEMO_B2_3 = DATA_DIR / "demo_b2_3_ee_poses.npz"

GRIPPER_THRESHOLD = 0.8515  # midpoint of the Galbot gripper range


def _zeros_bytes(dtype, shape):
    return np.zeros(shape, dtype=dtype).tobytes()


class FakeRpcError(Exception):
    """Transport error double carrying a status-code name."""

    def __init__(self, code_name, details=""):
        super().__init__(f"{code_name}: {details}")
        self._code_name = code_name
        self._details = details

    def code(self):
        class _Code:
            name = self._code_name

        return _Code()

    def details(self):
        return self._details


class FakePolicyServer:
    """Replays demo bundles as protocol observations, one env per demo."""

    def __init__(self, demo_paths):
        self._demos = []
        for path in demo_paths:
            ee = load_ee_poses_npz(path)
            grip = load_gripper_pos_npz(path)
            self._demos.append((ee, grip))
        self._index = [0] * len(self._demos)
        self._terminated = [False] * len(self._demos)
        self.step_attempts = 0
        self.step_successes = 0
        self.violations = []
        self.recorded = []  # (env, frame_index, gripper_left, gripper_right)
        self.fail_next_get = None
        self.fail_next_step = None
        self._zero_images = {
            name: _zeros_bytes(dtype, shape)
            for name, (dtype, shape) in FIELD_SHAPES.items()
            if "camera" in name
        }

    def _observation(self, env):
        ee, grip = self._demos[env]
        frame = self._index[env]
        poses = ee[frame]
        eef_pos = np.ascontiguousarray(poses[:, 0:3], dtype="<f4")
        # Stored quats are (w, x, y, z); the protocol wants (x, y, z, w).
        eef_quat = np.ascontiguousarray(poses[:, [4, 5, 6, 3]], dtype="<f4")
        gripper = (
            np.ascontiguousarray(grip[frame], dtype="<f4")
            if grip is not None
            else np.zeros(2, dtype="<f4")
        )
        return pb.Observation(
            first_person_camera_rgb=self._zero_images["first_person_camera_rgb"],
            left_hand_camera_rgb=self._zero_images["left_hand_camera_rgb"],
            right_hand_camera_rgb=self._zero_images["right_hand_camera_rgb"],
            actions=_zeros_bytes(np.dtype("<f4"), (19,)),
            joint_pos=_zeros_bytes(np.dtype("<f4"), (77,)),
            joint_vel=_zeros_bytes(np.dtype("<f4"), (77,)),
            eef_pos=eef_pos.tobytes(),
            eef_quat=eef_quat.tobytes(),
            gripper_pos=gripper.tobytes(),
        )

    def get_observation(self, request):
        if self.fail_next_get is not None:
            error, self.fail_next_get = self.fail_next_get, None
            raise error
        return pb.Transition(
            observations=[self._observation(env) for env in range(len(self._demos))],
            terminated=list(self._terminated),
            truncated=[False] * len(self._demos),
            final_observations=[],
        )

    def step(self, request):
        self.step_attempts += 1
        if self.fail_next_step is not None:
            error, self.fail_next_step = self.fail_next_step, None
            raise error
        for action in request.actions:
            values = np.asarray(action.values, dtype=np.float32)
            self.recorded.append(
                (len(self.recorded), values[14], values[15])
            )
            if values.shape != (ACTION_DIM,):
                self.violations.append(f"dim {values.shape}")
            if not np.all(np.isfinite(values)):
                self.violations.append("non-finite")
            for arm in range(2):
                quat = values[arm * 7 + 3 : arm * 7 + 7]
                if float(np.linalg.norm(quat)) < 1e-6:
                    self.violations.append(f"arm {arm} zero quat")
                if np.any(np.abs(values[arm * 7 : arm * 7 + 3]) > 10.0):
                    self.violations.append(f"arm {arm} pos out of range")
        observations = []
        terminated = []
        final_observations = []
        any_end = False
        for env, (ee, _grip) in enumerate(self._demos):
            # The last observation before reset (upstream: final_observations
            # is parallel to observations, meaningful only for ended envs).
            last = self._observation(env)
            self._index[env] += 1
            if self._index[env] >= len(ee):
                terminated.append(True)
                any_end = True
                self._index[env] = 0
                observations.append(self._observation(env))
                final_observations.append(last)
            else:
                terminated.append(False)
                obs = self._observation(env)
                observations.append(obs)
                final_observations.append(obs)
        if not any_end:
            final_observations = []
        self.step_successes += 1
        return pb.Transition(
            observations=observations,
            terminated=terminated,
            truncated=[False] * len(self._demos),
            final_observations=final_observations,
        )


class FakeChannel:
    """Passes calls through real protobuf serialization to the fake server."""

    def __init__(self, server):
        self._server = server

    def unary_unary(self, method, request_serializer=None, response_deserializer=None):
        def call(request, timeout=None):
            raw = request_serializer(request)
            if method.endswith("GetObservation"):
                req = pb.Empty.FromString(raw)
                resp = self._server.get_observation(req)
            elif method.endswith("Step"):
                req = pb.ActionBatch.FromString(raw)
                resp = self._server.step(req)
            else:  # pragma: no cover - unknown method
                raise ValueError(method)
            return response_deserializer(resp.SerializeToString())

        return call


def _policy_factory():
    return MultiDemoCore.from_npz_dir(
        DATA_DIR,
        actions_dim=20,
        action_quat_order="xyzw",
        lookahead=2,
        phase_advance_cap=4,
    )


def _gripper_agreement(server, demo_path):
    """Fraction of steps whose emitted gripper sign matches the recording."""
    grip = load_gripper_pos_npz(demo_path)
    if grip is None:
        return None
    ee = load_ee_poses_npz(demo_path)
    total = len(ee)
    agree = 0
    # recorded[i] is (sequence, left, right); frame index == sequence here
    # because the server advances one frame per successful step.
    for seq, left, right in server.recorded:
        if seq >= total:
            break
        for emitted, recorded in ((left, grip[seq, 0]), (right, grip[seq, 1])):
            expected_closed = recorded >= GRIPPER_THRESHOLD
            actual_closed = emitted < 0
            agree += 1 if expected_closed == actual_closed else 0
    return agree / (2 * min(len(server.recorded), total))


@unittest.skipUnless(
    _PROTOBUF_AVAILABLE, "protobuf not installed (grpc client tests need it)"
)
class GrpcClientTests(unittest.TestCase):
    def test_decode_observation_shapes(self):
        ee_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
        obs = pb.Observation(
            first_person_camera_rgb=_zeros_bytes(np.uint8, (480, 640, 3)),
            left_hand_camera_rgb=_zeros_bytes(np.uint8, (224, 224, 3)),
            right_hand_camera_rgb=_zeros_bytes(np.uint8, (224, 224, 3)),
            actions=_zeros_bytes(np.dtype("<f4"), (19,)),
            joint_pos=_zeros_bytes(np.dtype("<f4"), (77,)),
            joint_vel=_zeros_bytes(np.dtype("<f4"), (77,)),
            eef_pos=ee_pos.tobytes(),
            eef_quat=np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype="<f4"), (2, 1)
            ).tobytes(),
            gripper_pos=np.array([0.0, 1.703], dtype="<f4").tobytes(),
        )
        decoded = decode_observation(obs)
        for name, (dtype, shape) in FIELD_SHAPES.items():
            self.assertEqual(decoded[name].shape, shape, name)
            self.assertEqual(decoded[name].dtype, np.dtype(dtype), name)
        np.testing.assert_array_equal(decoded["eef_pos"], ee_pos)
        np.testing.assert_array_equal(
            decoded["gripper_pos"], np.array([0.0, 1.703], dtype="<f4")
        )

    def test_core_action_to_19_layout(self):
        action20 = np.zeros(20, dtype=np.float32)
        action20[0:7] = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        action20[7:14] = [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]
        action20[14:16] = [-1.0, 1.0]
        action20[16:20] = [9.0, 9.0, 9.0, 9.0]  # legacy wheels: dropped
        action19 = core_action_to_19(action20)
        self.assertEqual(action19.shape, (19,))
        np.testing.assert_array_equal(action19[:16], action20[:16])
        np.testing.assert_array_equal(action19[16:19], [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            core_action_to_19(np.zeros(19, dtype=np.float32))

    def test_validate_action_19_rejects(self):
        good = np.zeros(19, dtype=np.float32)
        good[3:7] = [0.0, 0.0, 0.0, 1.0]
        good[10:14] = [0.0, 0.0, 0.0, 1.0]
        validate_action_19(good)
        bad_quat = good.copy()
        bad_quat[3:7] = 0.0
        with self.assertRaises(ValueError):
            validate_action_19(bad_quat)
        bad_nan = good.copy()
        bad_nan[0] = np.nan
        with self.assertRaises(ValueError):
            validate_action_19(bad_nan)
        bad_pos = good.copy()
        bad_pos[0] = 10.5
        with self.assertRaises(ValueError):
            validate_action_19(bad_pos)
        with self.assertRaises(ValueError):
            validate_action_19(np.zeros(20, dtype=np.float32))

    def test_reset_selects_nearest_demo_tie_rule(self):
        # All Batch 2/3 first frames are bit-identical to the Batch 2 rest
        # pose, so nearest-start selection ties and resolves to the
        # earliest demo (demo_b2_0) per the documented tie rule.
        policy = ChamberGrpcPolicy(_policy_factory)
        ee = load_ee_poses_npz(DEMO_B2_3)
        info = policy.reset_env(0, {"eef_pos": ee[0, :, 0:3]})
        self.assertEqual(info["selected"], "demo_b2_0")

    def test_full_episode_demo_b2_0(self):
        server = FakePolicyServer([DEMO_B2_0])
        client = PolicyClient(FakeChannel(server))
        policy = ChamberGrpcPolicy(_policy_factory)
        total = len(load_ee_poses_npz(DEMO_B2_0))
        # One extra step past the episode length: the termination flag
        # arrives with the last step's transition and is only processed
        # on the next loop iteration.
        stats = run_policy_loop(client, policy, max_steps=total + 1)
        self.assertEqual(stats["steps"], total + 1)
        self.assertEqual(stats["episodes"], 1)
        self.assertEqual(stats["resets"], 1)
        self.assertEqual(server.violations, [])
        self.assertEqual(server.step_successes, total + 1)
        agreement = _gripper_agreement(server, DEMO_B2_0)
        self.assertGreaterEqual(agreement, 0.95)
        # After the termination the env reset; one further step leaves
        # the fresh core near the start of its trajectory.
        self.assertLessEqual(policy._cores[0].phase, 5)

    def test_determinism_short_run(self):
        def _run():
            server = FakePolicyServer([DEMO_B2_0])
            client = PolicyClient(FakeChannel(server))
            policy = ChamberGrpcPolicy(_policy_factory)
            run_policy_loop(client, policy, max_steps=300)
            return [tuple(map(float, (left, right))) for _, left, right in server.recorded]

        first = _run()
        second = _run()
        self.assertEqual(first, second)

    def test_two_envs_independent(self):
        server = FakePolicyServer([DEMO_B2_0, DEMO_B2_3])
        client = PolicyClient(FakeChannel(server))
        policy = ChamberGrpcPolicy(_policy_factory)
        total_b2_3 = len(load_ee_poses_npz(DEMO_B2_3))
        # One extra step so the termination flag is processed (see the
        # single-env test above for why).
        stats = run_policy_loop(client, policy, max_steps=total_b2_3 + 1)
        self.assertEqual(stats["steps"], total_b2_3 + 1)
        self.assertEqual(server.violations, [])
        self.assertEqual(stats["episodes"], 1)  # only env 1 finished
        self.assertEqual(policy.env_count, 2)
        # One recorded action per env per step.
        self.assertEqual(len(server.recorded), 2 * (total_b2_3 + 1))

    def test_failed_precondition_recovery(self):
        server = FakePolicyServer([DEMO_B2_0])
        server.fail_next_step = FakeRpcError("FAILED_PRECONDITION")
        client = PolicyClient(FakeChannel(server))
        policy = ChamberGrpcPolicy(_policy_factory)
        stats = run_policy_loop(client, policy, max_steps=50)
        self.assertEqual(stats["steps"], 50)
        self.assertEqual(server.step_successes, 50)
        self.assertEqual(server.step_attempts, 51)
        self.assertEqual(server.violations, [])

    def test_policy_client_retries_transient_getobservation(self):
        server = FakePolicyServer([DEMO_B2_0])
        server.fail_next_get = FakeRpcError("UNAVAILABLE")
        channel = FakeChannel(server)
        client = PolicyClient(channel)
        # First call fails once with UNAVAILABLE, the retry succeeds.
        transition = client.get_observation(pb.Empty(), timeout=5)
        self.assertEqual(len(transition.observations), 1)
        # A second failure is injected by failing the next call twice via
        # a fresh server whose get fails once more; retry still succeeds.
        server.fail_next_get = FakeRpcError("UNAVAILABLE")
        transition = client.get_observation(pb.Empty(), timeout=5)
        self.assertEqual(len(transition.observations), 1)


if __name__ == "__main__":
    unittest.main()
