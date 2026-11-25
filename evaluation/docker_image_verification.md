# Docker image verification (ECM 2027 milestone 19)

## What the image is

`Dockerfile.grpc` builds the policy as a Docker image for the
evaluator's official participant interface: a gRPC client that
connects to the evaluator's server at `ECM27_EVAL_ENDPOINT`
(`python -m ecm_policy.grpc_client`, entrypoint). The delivery
channel is published: the organizer mail of 2026-11-20 (Liu Haichao,
NTU) says the Docker image is submitted via
https://form.gov.sg/6ae35050aafe2da79edea132 (up to 5 submissions).

## Verification route

The image is verified by GitHub Actions
(`.github/workflows/docker-verify.yml`, runs on every push touching
the Dockerfile, the policy code, or the fake server). The workflow:

1. Runs the full pytest suite on the runner (Python 3.12).
2. Builds the image (`docker build -f Dockerfile.grpc`).
3. Starts `tools/fake_eval_server.py` on the runner host
   (`--unix-socket /tmp/ecm-fake.sock --max-steps 4107`, replaying
   the real `demo_b2_0` bundle).
4. Runs the container with `--network host`, mounting `/tmp` so the
   container sees the socket, with
   `ECM27_EVAL_ENDPOINT=unix:///tmp/ecm-fake.sock` and
   `ECM_MAX_STEPS=4107`.
5. Passes only when the container exits 0 AND the evaluator verdict
   is: 4107 steps, 1 episode, no action violations, gripper
   agreement >= 0.95.

The fake server is the same tool used by the local socket tests
(`tests/test_fake_eval_server.py`, 3 tests, green in the isolated
verification below), so the container run exercises the real
entrypoint, real protobuf serialization, and real gRPC transport —
not a stub.

## Why not on the dev host

The dev VM cannot run containers at all, and this was proven, not
assumed, on 2026-11-25:

- Docker 29.1.3 installed (twice; the VM rebooted twice this round
  and `/usr` is ephemeral, so each reboot wiped the install).
  Daemon started with `--iptables=false --bridge=none`.
- First build failed at layer extraction: overlayfs mount rejected
  (`failed to mount ... overlay ... invalid argument`).
- Daemon restarted with `--storage-driver=vfs` (no mounts needed
  for layers). The build then got past the base image, but the first
  `RUN` step failed: `failed to create a netlink handle ... operation
  not permitted`.
- A trivial `RUN echo` image failed the same way at container
  start: `runc create failed: ... join container mntns: setns:
  operation not permitted`.

The sandbox blocks the mount-namespace operations runc needs, so
neither `docker build` (RUN steps) nor `docker run` can execute on
this host. This is a dev-host limitation, not a defect in the image
or the policy. The production evaluator runs the image on its own
host.

## Isolated test verification (commit c34fe19, 2026-11-25)

Fresh clone at `c34fe19` under
`hidden_files/isolated-verify/ecm-m19/`, with the upstream starter
sibling at `17159e5`:

- venv Python (grpcio 1.85.3, protobuf 7.36.2): **117 passed** in
  119.85 s, 0 failed, 0 skipped.
- System Python 3.12 (`PYTHONPATH=$HOME/.local/lib/python3.12/site-packages`):
  **100 passed, 17 skipped** in 129.13 s (14 h5py/protobuf-gated +
  3 new grpcio-gated socket tests skipped by design).

The in-place venv suite at the same commit also passed 117/117.
The 3 new socket tests drive the real fake server and the real
`python -m ecm_policy.grpc_client` entrypoint over a Unix-domain
socket (50 steps, exit 0, no violations).

## CI run record

- 2026-11-25, run 45573546978 (commit 1175928): FAILED in 17 s at
  test collection — `tests/test_adapter.py` reads the real base
  policy from the sibling `ecm2027-sim-track` checkout, which the
  runner did not have.
- 2026-11-25, run 45573822033 (commit f62de8c): the starter clone
  was added (pinned to 17159e5). The suite passed on the runner
  (103 passed, 14 skipped in 49.26 s), the image built, and the
  container booted cleanly (CONTAINER_EXIT=0; verdict 4107 steps,
  1 episode, no violations, gripper agreement 0.9961, passed true).
  The run still failed only because the verdict-check step assumed
  `gripper_agreement` is a float — it is a per-demo dict
  (`{'demo_b2_0_ee_poses': 0.9961}`).
- 2026-11-25, run 45574640392 (commit 67786c1): **SUCCESS**.
  Suite 103 passed, 14 skipped in 28.09 s; image built;
  CONTAINER_EXIT=0; verdict 4107 steps, 1 episode, no violations,
  gripper agreement 0.9961, passed true; "DOCKER IMAGE VERIFIED:
  full episode, no violations". The Docker image is verified.
