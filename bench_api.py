"""Starts the API on your own machine, sends 300 real HTTP requests, and prints the response times.
Run:  python bench_api.py        (needs: pip install fastapi uvicorn requests)"""
import subprocess, sys, time
import numpy as np
import requests
import orion3_core as core

URL = "http://127.0.0.1:8000"
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--port", "8000", "--log-level", "warning"])
try:
    for _ in range(120):                       # wait until the API is up (it trains models on first start)
        try:
            if requests.get(URL + "/health", timeout=1).ok: break
        except requests.RequestException:
            time.sleep(1)
    else:
        raise SystemExit("API did not start in 2 minutes")
    payloads = list(core.PRESETS.values())
    s = requests.Session()
    for p in payloads: s.post(URL + "/score", json=p)          # warm-up
    ms, actions = [], {}
    for i in range(300):
        p = payloads[i % len(payloads)]
        t0 = time.perf_counter(); r = s.post(URL + "/score", json=p); ms.append((time.perf_counter() - t0) * 1000)
        a = r.json()["action"]; actions[a] = actions.get(a, 0) + 1
    print("requests: 300 | actions:", actions)
    print(f"round-trip ms (local HTTP, incl. scoring): p50={np.percentile(ms, 50):.1f}  p95={np.percentile(ms, 95):.1f}  p99={np.percentile(ms, 99):.1f}  max={max(ms):.1f}")
    print("under 200 ms for", f"{(np.array(ms) < 200).mean():.1%}", "of requests")
finally:
    server.terminate()
