# ECM 2027 policy service image (v1).
#
# Mirrors the submission shape described in the official starter README:
# a Docker image containing the policy, its dependencies and its data,
# exposing health / reset / predict / close, running fully offline.
# The official transport and message schemas were still unpublished as of
# 2026-10-31; this image serves the JSON-over-HTTP contract implemented in
# ecm_policy/service.py and can be re-skinned when the spec lands.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ecm_policy/ ./ecm_policy/
COPY data/ ./data/
COPY README.md ./

ENV PYTHONPATH=/app \
    PORT=8080
# NOTE: no ECM_TRAJECTORY_PATH here on purpose. That variable is only
# honoured by the harness adapter (ecm_policy/adapter.py); the service
# entrypoint below ignores it and defaults to multi-demo mode over
# /app/data (all four bundled episodes). Setting it here used to imply
# a single-demo (demo_7) image, which was not what the image ran.

EXPOSE 8080

# The demo trajectory is baked into the image; no network is needed at runtime.
CMD ["python", "-m", "ecm_policy.service", "--host", "0.0.0.0", "--port", "8080"]
