#!/usr/bin/env python3
"""G3: out-of-order arrival harness for ninfer-serve.

The lockstep benchmarks start every request at the same instant, which hides what a service
actually sees: requests arriving at random times, queueing behind each other, finishing out of
order. This harness sends R requests with jittered arrivals spread over N workers, and reports the
latency distribution (min/p50/p90/p99/max) plus failures.

Usage:
  python g3_jitter.py --workers 4 --requests 40 --jitter-ms 200 --label "C=4 jitter"
  python g3_jitter.py --workers 1 --requests 20 --jitter-ms 0   --label "C=1 lockstep"

Exit code is 0 when every request succeeded, 1 otherwise.
"""

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

PROMPT = (
    "压测提示：这段文本用于测量请求在乱序到达时的服务时延，它与缓存内容无关，"
    "每次都会走完整的前填充与解码路径。请只回一句话。"
)

# A per-request nonce goes FIRST, so two requests never share a prefix and the prefix cache cannot
# turn the load test into a cache-hit test.
NONCE = "n{}"


def percentile(values, fraction):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def one_request(url, model, max_tokens, timeout, text):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode("utf-8")
    started = time.perf_counter()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json; charset=utf-8"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return time.perf_counter() - started, payload.get("usage", {}), None
    except urllib.error.HTTPError as error:
        return time.perf_counter() - started, {}, f"HTTP {error.code}"
    except Exception as error:  # noqa: BLE001 - the harness reports whatever it got
        return time.perf_counter() - started, {}, f"{type(error).__name__}: {error}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18099/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unique", action="store_true", help="prepend a nonce so no prefix is shared")
    args = parser.parse_args()

    random.seed(args.seed)
    # One arrival time per request, in nondecreasing order, so the load is spread the way a
    # service sees it rather than released as a single burst.
    arrivals = sorted(random.uniform(0.0, args.jitter_ms / 1000.0) for _ in range(args.requests))

    pending = list(arrivals)
    pending_lock = threading.Lock()
    latencies, errors = [], []
    result_lock = threading.Lock()

    def worker():
        while True:
            with pending_lock:
                if not pending:
                    return
                offset = pending.pop(0)
            now = time.perf_counter() - start_at
            if offset > now:
                time.sleep(offset - now)
            text = (NONCE.format(random.getrandbits(48)) + " " + PROMPT) if args.unique else PROMPT
            elapsed, usage, error = one_request(
                args.url, args.model, args.max_tokens, args.timeout, text
            )
            with result_lock:
                latencies.append(elapsed)
                if error is not None:
                    errors.append(f"{error} (after {elapsed:.2f}s)")
                else:
                    tokens = usage.get("prompt_tokens", 0)
                    if tokens == 0:
                        errors.append("no usage in response")

    start_at = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(max(1, args.workers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - start_at

    print(
        f"[{args.label}] workers={args.workers} jitter={args.jitter_ms:.0f}ms "
        f"requests={args.requests} ok={len(latencies) - len(errors)} failed={len(errors)} "
        f"wall={wall:.1f}s"
    )
    if latencies:
        print(
            f"[{args.label}] latency s: min={min(latencies):.2f} "
            f"p50={percentile(latencies, 0.50):.2f} p90={percentile(latencies, 0.90):.2f} "
            f"p99={percentile(latencies, 0.99):.2f} max={max(latencies):.2f}"
        )
    for error in errors[:5]:
        print(f"[{args.label}] error: {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
