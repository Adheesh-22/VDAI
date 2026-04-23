#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8000}"
BASE_URL="http://${HOST}:${PORT}"

PASS=0
FAIL=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "=============================="
echo " VoiceGuard-Proto Health Check"
echo " Target: $BASE_URL"
echo "=============================="
echo ""

# /health
HTTP_CODE=$(curl -s -o /tmp/vg_health.json -w "%{http_code}" "$BASE_URL/health")
check "/health returns 200" $([ "$HTTP_CODE" = "200" ] && echo 0 || echo 1)

STATUS=$(python3 -c "import json,sys; d=json.load(open('/tmp/vg_health.json')); sys.exit(0 if d.get('status')=='healthy' else 1)" 2>/dev/null; echo $?)
check "/health status is 'healthy'" "$STATUS"

MODEL_LOADED=$(python3 -c "import json,sys; d=json.load(open('/tmp/vg_health.json')); sys.exit(0 if d.get('model_loaded') else 1)" 2>/dev/null; echo $?)
check "/health model_loaded is true" "$MODEL_LOADED"

# /stats
HTTP_CODE=$(curl -s -o /tmp/vg_stats.json -w "%{http_code}" "$BASE_URL/stats")
check "/stats returns 200" $([ "$HTTP_CODE" = "200" ] && echo 0 || echo 1)

HAS_KEYS=$(python3 -c "import json,sys; d=json.load(open('/tmp/vg_stats.json')); sys.exit(0 if 'total_inferences' in d and 'avg_inference_ms' in d else 1)" 2>/dev/null; echo $?)
check "/stats has required keys" "$HAS_KEYS"

# /metrics
HTTP_CODE=$(curl -s -o /tmp/vg_metrics.txt -w "%{http_code}" "$BASE_URL/metrics")
check "/metrics returns 200" $([ "$HTTP_CODE" = "200" ] && echo 0 || echo 1)

HAS_METRIC=$(grep -q "active_connections" /tmp/vg_metrics.txt && echo 0 || echo 1)
check "/metrics contains 'active_connections'" "$HAS_METRIC"

echo ""
echo "=============================="
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "=============================="
echo ""

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
