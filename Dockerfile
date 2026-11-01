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
    PORT=8080 \
    ECM_TRAJECTORY_PATH=/app/data/demo_7_ee_poses.npz

EXPOSE 8080

# The demo trajectory is baked into the image; no network is needed at runtime.
CMD ["python", "-m", "ecm_policy.service", "--host", "0.0.0.0", "--port", "8080"]
