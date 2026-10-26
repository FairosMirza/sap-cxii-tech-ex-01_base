#!/usr/bin/env bash
# Smoke + zero-downtime test of the deployed service (used by `make kind-smoke` and CI).
# Starts scripts/k8s_smoke.py in a pod, restarts the Deployment while it sends requests,
# and fails if any request failed.
set -euo pipefail

KUBECTL="${KUBECTL:-kubectl}"
IMAGE="${IMAGE:-product-similarity:1.0.0}"
DURATION="${DURATION:-60}"
cd "$(dirname "$0")/.."

$KUBECTL rollout status deployment/product-similarity --timeout=180s
$KUBECTL delete pod smoke --ignore-not-found --wait
$KUBECTL run smoke --image="$IMAGE" --image-pull-policy=IfNotPresent --restart=Never \
  --env="DURATION=$DURATION" --command -- python -c "$(cat scripts/k8s_smoke.py)"

for _ in $(seq 90); do
  $KUBECTL logs pod/smoke 2>/dev/null | grep -q "functional checks" && break
  sleep 1
done
$KUBECTL rollout restart deployment/product-similarity
$KUBECTL rollout status deployment/product-similarity --timeout=180s
$KUBECTL wait --for=jsonpath='{.status.phase}'=Succeeded pod/smoke --timeout="$((DURATION + 60))s" || true

$KUBECTL logs pod/smoke
phase="$($KUBECTL get pod smoke -o jsonpath='{.status.phase}')"
$KUBECTL delete pod smoke --wait=false >/dev/null
echo "smoke test: $phase"
[ "$phase" = "Succeeded" ]
