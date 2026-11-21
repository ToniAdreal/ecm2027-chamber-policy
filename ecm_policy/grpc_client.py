"""gRPC policy client for the official ECM 2027 evaluator protocol.

Implements the participant-facing interface published by the upstream
starter repo on 2026-11-20 (``robofinals/distributed/policy_v1.proto``,
README section "Participant policy interface", commit ``a1dff75``):
the policy is a gRPC *client* connecting to the evaluator's server at
``ECM27_EVAL_ENDPOINT`` and exchanging ``GetObservation`` / ``Step``
calls. Actions are the preferred 19-dimensional format: left/right arm
absolute end-effector target poses ``[x, y, z, qx, qy, qz, qw]`` in the
robot root frame, gripper commands (negative closes, zero or positive
opens), and body-frame base velocity ``[vx, vy, wz]``.

``ChamberGrpcPolicy`` wraps :class:`MultiDemoCore` (one stateful core
per environment) and converts its 20-dimensional legacy output
(``[left pose 7, right pose 7, gripper 2, wheels 4]``) to the 19-dim
format by dropping the wheel commands and appending a zero base
velocity. Quaternions are emitted in ``(x, y, z, w)`` order, as the
protocol requires.

Frame assumption, stated plainly: demo end-effector poses are passed
through exactly as recorded (the observation frame: world-axis
coordinates relative to the environment origin), with base velocity
held at zero, so the robot base never moves. Whether the recorded
poses coincide with the evaluator's robot-root frame depends on the
spawn pose, which is an open upstream question (starter issue #5,
"robot spawn pose differs from the demos", still unanswered). If the
organizers confirm a spawn offset, the conversion belongs in
:func:`core_action_to_19` (a fixed rigid transform on the pose
targets), not in the phase tracker.

Transport notes:

* ``grpc`` is imported lazily: this module (decode / act / validate /
  the run loop) works without ``grpcio`` installed, and
  :class:`PolicyClient` only needs a channel object exposing
  ``unary_unary``. The real channel comes from ``grpc.insecure_channel``
  in :func:`main`. The retry policy mirrors the upstream
  ``PolicyClient``: ``GetObservation`` retries on transient errors,
  ``Step`` retries only on the explicit server-busy rejection that
  occurs before execution.
* A ``FAILED_PRECONDITION`` on ``Step`` means a wall-clock-timed-out
  episode is waiting for its reset observation to be consumed; the run
  loop re-reads with ``GetObservation`` and continues, exactly as the
  upstream example does.
* The 13-point chamber scoring (open chamber 2, extract old part 1,
  place in box 1, sponge + wipe five faces 6, replacement part 2,
  close chamber 1) is evaluator-side and not part of this interface.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import policy_v1_pb2 as pb

try:  # grpcio exists only where the real evaluator channel is built
    import grpc as _grpc
except ImportError:  # pragma: no cover - exercised on grpc-less hosts
    _grpc = None

# -- protocol constants (upstream README, "Participant policy interface") --

ACTION_DIM = 19
LEGACY_ACTION_DIM = 20
POSITION_LIMIT = 10.0  # metres, per arm target component
BASE_VEL_LIMIT = 10.0  # m/s or rad/s, per base-velocity component
MIN_QUAT_NORM = 1e-6  # the service rejects smaller norms as invalid
DEFAULT_TIMEOUT_SECONDS = 300 * 60  # upstream example timeout
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_RETRIES = 3

FIELD_SHAPES = {
    "first_person_camera_rgb": (np.uint8, (480, 640, 3)),
    "left_hand_camera_rgb": (np.uint8, (224, 224, 3)),
    "right_hand_camera_rgb": (np.uint8, (224, 224, 3)),
    "actions": (np.dtype("<f4"), (19,)),
    "joint_pos": (np.dtype("<f4"), (77,)),
    "joint_vel": (np.dtype("<f4"), (77,)),
    "eef_pos": (np.dtype("<f4"), (2, 3)),
    "eef_quat": (np.dtype("<f4"), (2, 4)),
    "gripper_pos": (np.dtype("<f4"), (2,)),
}

_SERVICE_PATH = "/ecm27.policy.v1.Policy/"


# -- observation decoding ---------------------------------------------------

def decode_observation(observation) -> Dict[str, np.ndarray]:
    """Decode one protocol ``Observation`` into a dict of NumPy arrays.

    Shapes and dtypes follow ``FIELD_SHAPES`` (the upstream example's
    table). Returned arrays are copies; the wire bytes are not retained.
    """
    return {
        name: np.frombuffer(getattr(observation, name), dtype=dtype)
        .reshape(shape)
        .copy()
        for name, (dtype, shape) in FIELD_SHAPES.items()
    }


# -- action conversion + validation -----------------------------------------

def core_action_to_19(action20: np.ndarray) -> np.ndarray:
    """Convert a 20-dim legacy core action to the 19-dim protocol format.

    The core emits ``[left pose 7, right pose 7, gripper 2, wheels 4]``
    with quaternions in ``(x, y, z, w)`` order when built with
    ``action_quat_order="xyzw"``. The protocol keeps the first 16 values
    and replaces the four legacy wheel commands with a zero body-frame
    base velocity ``[vx, vy, wz]``: this policy never drives the base.
    """
    action = np.asarray(action20, dtype=np.float32).reshape(-1)
    if action.shape != (LEGACY_ACTION_DIM,):
        raise ValueError(
            f"expected a {LEGACY_ACTION_DIM}-dim legacy action, "
            f"got shape {action.shape}"
        )
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[:16] = action[:16]
    out[16:19] = 0.0
    return out


def validate_action_19(action: np.ndarray) -> np.ndarray:
    """Enforce the evaluator's action-validity checks locally.

    Raises :class:`ValueError` on anything the service would reject
    with ``INVALID_ARGUMENT`` before advancing the simulation: wrong
    dimension, non-finite values, a zero-norm quaternion, or a position
    / base-velocity component outside the numerical protection limits.
    Returns the action as float32.
    """
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape != (ACTION_DIM,):
        raise ValueError(
            f"expected a {ACTION_DIM}-dim action, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("action contains non-finite values")
    for arm in range(2):
        quat = arr[arm * 7 + 3 : arm * 7 + 7]
        if float(np.linalg.norm(quat)) < MIN_QUAT_NORM:
            raise ValueError(
                f"arm {arm} quaternion norm below {MIN_QUAT_NORM:g}"
            )
        pos = arr[arm * 7 : arm * 7 + 3]
        if np.any(np.abs(pos) > POSITION_LIMIT):
            raise ValueError(
                f"arm {arm} target position outside ±{POSITION_LIMIT:g} m: {pos}"
            )
    base_vel = arr[16:19]
    if np.any(np.abs(base_vel) > BASE_VEL_LIMIT):
        raise ValueError(f"base velocity outside ±{BASE_VEL_LIMIT:g}: {base_vel}")
    return arr


# -- transport client (mirrors upstream PolicyClient) -----------------------

def _code_name(exc: Exception):
    """Best-effort status-code name for a transport error.

    Works for real ``grpc.RpcError`` values (whose ``code()`` returns a
    ``StatusCode`` enum with a ``.name``) and for test doubles that
    return the plain string name.
    """
    code = exc.code() if hasattr(exc, "code") else None
    return getattr(code, "name", code)


class PolicyClient:
    """Minimal stub for the ``ecm27.policy.v1.Policy`` service.

    Retry only requests known not to have advanced the simulation:
    ``GetObservation`` retries on transient errors, ``Step`` retries
    only on the explicit server-busy rejection. A lost or timed-out
    ``Step`` reply may follow an already executed action, so it is
    never blindly retried. Only needs a channel exposing
    ``unary_unary``; the real channel is built with
    ``grpc.insecure_channel`` in :func:`main`.
    """

    def __init__(self, channel):
        self._get_observation = channel.unary_unary(
            _SERVICE_PATH + "GetObservation",
            request_serializer=pb.Empty.SerializeToString,
            response_deserializer=pb.Transition.FromString,
        )
        self._step = channel.unary_unary(
            _SERVICE_PATH + "Step",
            request_serializer=pb.ActionBatch.SerializeToString,
            response_deserializer=pb.Transition.FromString,
        )

    @staticmethod
    def _invoke(call, request, *, timeout, step):
        for attempt in range(MAX_RETRIES + 1):
            try:
                return call(request, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - retry policy below
                if step:
                    # This exact server rejection occurs before any
                    # action is enqueued.
                    safe = (
                        _code_name(exc) == "RESOURCE_EXHAUSTED"
                        and exc.details() == "another request is in progress"
                    )
                else:
                    safe = _code_name(exc) in (
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                        "RESOURCE_EXHAUSTED",
                    )
                if not safe or attempt >= MAX_RETRIES:
                    raise
                time.sleep(0.1 * (2**attempt))

    def get_observation(self, request, *, timeout=None):
        return self._invoke(
            self._get_observation, request, timeout=timeout, step=False
        )

    def step(self, request, *, timeout=None):
        return self._invoke(self._step, request, timeout=timeout, step=True)


# -- policy wrapper ----------------------------------------------------------

class ChamberGrpcPolicy:
    """Per-environment replay-follow policy behind the gRPC interface.

    ``core_factory`` builds one fresh core (a ``MultiDemoCore``) per
    environment index; cores are stateful (phase + selected demo), so
    environments never share one. The first observation seen for an
    environment selects its demo by nearest start pose; a later
    termination resets that environment's core from the new episode's
    first observation.
    """

    def __init__(self, core_factory: Callable[[], object]):
        self._factory = core_factory
        self._cores: Dict[int, object] = {}
        self._initialized: set = set()

    def _core(self, index: int):
        if index not in self._cores:
            self._cores[index] = self._factory()
        return self._cores[index]

    def reset_env(self, index: int, observation: dict) -> dict:
        """Reset one environment's core, selecting its demo from ``observation``."""
        core = self._core(index)
        info = core.reset(observation=observation)
        self._initialized.add(index)
        return info

    def act(self, observations: List[dict]) -> List[np.ndarray]:
        """One validated 19-dim action per observation (one per env)."""
        actions = []
        for index, observation in enumerate(observations):
            if index not in self._initialized:
                self.reset_env(index, observation)
            core = self._cores[index]
            action20 = core.predict(observation)
            action19 = validate_action_19(core_action_to_19(action20))
            actions.append(action19)
        return actions

    @property
    def env_count(self) -> int:
        return len(self._cores)


# -- run loop -----------------------------------------------------------------

def run_policy_loop(client, policy: ChamberGrpcPolicy, max_steps: Optional[int] = None) -> dict:
    """Drive the evaluator protocol until ``max_steps`` (or forever).

    ``client`` needs ``get_observation(Empty)`` and ``step(ActionBatch)``
    methods (the real :class:`PolicyClient`, or a test double). Returns
    ``{"steps", "resets", "episodes"}``. ``max_steps=None`` runs until
    the evaluator stops the process or an RPC error propagates.
    """
    stats = {"steps": 0, "resets": 0, "episodes": 0}
    transition = client.get_observation(
        pb.Empty(), timeout=DEFAULT_TIMEOUT_SECONDS
    )
    while True:
        if max_steps is not None and stats["steps"] >= max_steps:
            return stats
        observations = [decode_observation(obs) for obs in transition.observations]
        for index, (terminated, truncated) in enumerate(
            zip(transition.terminated, transition.truncated)
        ):
            if terminated or truncated:
                # observations[index] is the NEW episode's first
                # observation (per the protocol); reset from it.
                policy.reset_env(index, observations[index])
                stats["resets"] += 1
                stats["episodes"] += 1
        actions = policy.act(observations)
        request = pb.ActionBatch(
            actions=[pb.Action(values=row) for row in actions]
        )
        try:
            transition = client.step(request, timeout=DEFAULT_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 - FAILED_PRECONDITION path
            if _code_name(exc) == "FAILED_PRECONDITION":
                # A wall-clock-timed-out episode waits for its reset
                # observation to be consumed; re-read and continue.
                transition = client.get_observation(
                    pb.Empty(), timeout=DEFAULT_TIMEOUT_SECONDS
                )
                continue
            raise
        stats["steps"] += 1


# -- entrypoint -----------------------------------------------------------------

def main() -> int:
    """Run the policy against the evaluator endpoint from the environment.

    ``ECM27_EVAL_ENDPOINT`` is set by the evaluator to the server's
    ``host:port``. ``ECM_TRAJECTORY_DIR`` points at the bundled demo
    ``*_ee_poses.npz`` files (default: the repo's ``data/`` directory).
    Requires ``grpcio``; without it, exit with a clear message instead
    of an import traceback.
    """
    if _grpc is None:  # pragma: no cover - production path only
        raise SystemExit(
            "grpcio is required to connect to the evaluator "
            "(pip install grpcio==1.85.3 protobuf==7.36.2); "
            "see Dockerfile.grpc"
        )
    from .multidemo import MultiDemoCore

    endpoint = os.environ["ECM27_EVAL_ENDPOINT"]
    demo_dir = Path(
        os.environ.get(
            "ECM_TRAJECTORY_DIR",
            str(Path(__file__).resolve().parent.parent / "data"),
        )
    )
    channel = _grpc.insecure_channel(
        endpoint,
        options=[("grpc.max_receive_message_length", MAX_MESSAGE_BYTES)],
    )
    _grpc.channel_ready_future(channel).result(timeout=DEFAULT_TIMEOUT_SECONDS)
    client = PolicyClient(channel)

    def _factory():
        return MultiDemoCore.from_npz_dir(
            demo_dir,
            actions_dim=LEGACY_ACTION_DIM,
            action_quat_order="xyzw",
            lookahead=2,
            phase_advance_cap=4,
        )

    policy = ChamberGrpcPolicy(_factory)
    stats = run_policy_loop(client, policy)
    return 0


if __name__ == "__main__":  # pragma: no cover - production entrypoint
    raise SystemExit(main())
