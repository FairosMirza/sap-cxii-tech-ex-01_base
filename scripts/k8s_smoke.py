"""In-cluster smoke and availability test, run in a pod by scripts/k8s_smoke_test.sh.

Checks the API contract through the Service, then calls it continuously for DURATION
seconds while the caller restarts the Deployment. Any failed request fails the test,
so this verifies zero-downtime rollouts.
"""

import collections
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

PRODUCT = "80e47075f39a4bdf6b83f2d7690c95fe"
SIMILAR = f"/find_similar_products?product_id={PRODUCT}&num_similar=5"
DURATION = float(os.getenv("DURATION", "60"))
BASE = f"http://{socket.gethostbyname(os.getenv('SERVICE', 'product-similarity'))}"  # test the Service, not DNS


def get(path: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, None


checks = {
    "similar products": (SIMILAR, 200),
    "unknown product -> 404": ("/find_similar_products?product_id=nope&num_similar=5", 404),
    "num_similar=0 -> 422": (f"/find_similar_products?product_id={PRODUCT}&num_similar=0", 422),
    "readiness": ("/ready", 200),
}
failed = [name for name, (path, expected) in checks.items() if get(path)[0] != expected]
body = get(SIMILAR)[1]
if not (isinstance(body, list) and len(body) == 5 and all(isinstance(x, str) for x in body)):
    failed.append("response is a list of 5 ids")
print("functional checks:", "OK" if not failed else f"FAILED {failed}", flush=True)

codes, latencies, start = collections.Counter(), [], time.time()
while time.time() - start < DURATION:
    t = time.time()
    try:
        codes[str(get(SIMILAR)[0])] += 1
    except Exception as exc:  # connection refused/reset: the failure mode a bad rollout causes
        codes[type(exc).__name__] += 1
    latencies.append(1000 * (time.time() - t))
    time.sleep(0.02)

latencies.sort()
print(json.dumps({
    "requests": len(latencies),
    "status": dict(codes),
    "p50_ms": round(latencies[len(latencies) // 2], 2),
    "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
}))
sys.exit(1 if failed or set(codes) != {"200"} else 0)
