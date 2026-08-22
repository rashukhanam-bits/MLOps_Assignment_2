#!/usr/bin/env bash
# Post-deploy smoke test: hits /health and /predict, exits non-zero on failure
# so the CD pipeline job fails per the M4 spec.
#
# Usage: bash scripts/smoke_test.sh http://localhost:8000 [path/to/sample.jpg]

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SAMPLE_IMAGE="${2:-scripts/sample.jpg}"

echo "== Smoke test target: $BASE_URL =="

echo "-- Checking /health"
HEALTH_STATUS=$(curl -s -o /tmp/health.json -w "%{http_code}" "$BASE_URL/health")
cat /tmp/health.json
if [ "$HEALTH_STATUS" != "200" ]; then
  echo "FAIL: /health returned $HEALTH_STATUS"
  exit 1
fi
echo "OK: /health returned 200"

if [ ! -f "$SAMPLE_IMAGE" ]; then
  echo "-- No sample image found at $SAMPLE_IMAGE, generating a blank one for a smoke check"
  python3 - "$SAMPLE_IMAGE" <<'PY'
import sys
from PIL import Image
Image.new("RGB", (224, 224), (120, 120, 120)).save(sys.argv[1])
PY
fi

echo "-- Checking /predict"
PREDICT_STATUS=$(curl -s -o /tmp/predict.json -w "%{http_code}" -X POST \
  -F "file=@${SAMPLE_IMAGE}" "$BASE_URL/predict")
cat /tmp/predict.json
if [ "$PREDICT_STATUS" != "200" ]; then
  echo "FAIL: /predict returned $PREDICT_STATUS"
  exit 1
fi
echo "OK: /predict returned 200"

echo "== Smoke test passed =="
