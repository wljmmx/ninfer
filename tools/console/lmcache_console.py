"""Serve the console and expose LMCache's Prometheus metrics to it as JSON.

Why this exists: LMCache's HTTP server sends no CORS headers, so a dashboard opened from
file:// cannot read http://127.0.0.1:18080/metrics directly. This serves the page and the
metrics from one origin, which also means the browser never needs a CORS exception.

Parsing happens here rather than in the page so the front end only has to read named fields,
and so the metric-name mapping lives next to the code that can be checked against the source.

Also serves /api/gpu.json: NVML telemetry (power / clocks / temperature / fan / clocks
event reasons, plus low-frequency PCIe throughput) sampled on a side thread. Everything is
a control-path read: it never touches a CUDA context and cannot affect kernel timing,
PPL or hash gates. Voltage is deliberately absent — nvml.dll exports no voltage symbol
and `nvidia-smi -q -d VOLTAGE` is empty on GeForce; power-limit plus the power-cap event
bits are the honest substitute.

And /api/engine.json: the engine's own /monitor/info|snapshot|events, polled at 1 Hz and
subscribed once for events from this same process (no second collector). Counters are
converted to rates here so the page only reads numbers; a failed poll marks a gap and the
page breaks the line instead of interpolating. When the engine is unreachable the payload
says so (ok=false plus a reason) and the page labels its demo fallback as 演示数据.

Usage:
    python lmcache_console.py [--port 8099] [--upstream http://127.0.0.1:18080/metrics]
                              [--engine http://127.0.0.1:11438]
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import datetime
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Matches "name{labels} value" and "name value". Prometheus text format allows a trailing timestamp
# after the value, which we ignore.
SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[^\s]+)")

# The metrics worth putting on a dashboard, grouped the way they will be displayed. Anything not
# listed here is still counted in "families" so a newly appearing metric is noticed rather than
# silently ignored.
GROUPS = {
    "l1": {
        "usage_ratio": "lmcache_mp_l1_usage_ratio",
        "memory_bytes": "lmcache_mp_l1_memory_usage_bytes",
        "staging_bytes": "lmcache_mp_l1_staging_bytes",
        "evict_ticks": "lmcache_mp_l1_eviction_loop_ticks_total",
    },
    "l2": {
        "store_adapters": "lmcache_mp_l2_store_adapters",
        "prefetch_adapters": "lmcache_mp_l2_prefetch_adapters",
    },
    "jobs": {
        "active_prefetch": "lmcache_mp_active_prefetch_jobs",
        "active_p2p_lookup": "lmcache_mp_active_p2p_lookup_jobs",
    },
    "events": {
        "queue_depth": "lmcache_mp_event_bus_queue_depth",
        "dropped": "lmcache_mp_event_bus_dropped_events_total",
        "log_depth": "lmcache_mp_kv_events_log_depth",
        "drain_lag_s": "lmcache_mp_event_bus_drain_lag_seconds",
    },
    "process": {
        "rss_bytes": "process_resident_memory_bytes",
        "cpu_seconds": "process_cpu_seconds_total",
        "open_fds": "process_open_fds",
    },
}

LOOKING_FOR = [
    "lmcache_mp_lookup_hit",
    "lmcache_mp_lookup_total",
    "lmcache_mp_retrieve_hit",
]


def parse_prometheus(text: str) -> tuple[dict[str, float], set[str]]:
    """Return {metric_name: summed value} and the set of every family present.

    Summing across label sets is deliberate: a dashboard row answers "how much in total", and the
    per-label split is not worth a second panel yet.
    """
    values: dict[str, float] = {}
    families: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        if not match:
            continue
        name = match.group("name")
        families.add(name)
        try:
            value = float(match.group("value"))
        except ValueError:
            continue  # NaN / Inf as text; not useful on a gauge row
        values[name] = values.get(name, 0.0) + value
    return values, families


def build_snapshot(upstream: str, timeout: float = 3.0) -> dict:
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        with urllib.request.urlopen(upstream, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "at": stamp, "upstream": upstream, "error": str(exc)}

    values, families = parse_prometheus(text)
    snapshot: dict = {"ok": True, "at": stamp, "upstream": upstream,
                      "samples": len(values), "families": sorted(families)}
    for group, mapping in GROUPS.items():
        snapshot[group] = {
            field: values.get(metric) for field, metric in mapping.items()
        }
    # Reported so the dashboard can say "尚未出现" instead of drawing a zero that means nothing.
    snapshot["hit_metrics_present"] = sorted(n for n in families if n.startswith("lmcache_mp_lookup"))
    snapshot["hit_metrics_missing"] = [n for n in LOOKING_FOR if n not in families]
    return snapshot


# ------------------------------------------------------------------ GPU (NVML)

# Measured on the RTX 4090 (driver 621.01): the scalar getters below cost microseconds per
# call; nvmlDeviceGetPcieThroughput blocks ~31 ms each because it waits for the driver's
# counter window, so it runs only every PCIE_EVERY ticks on this side thread and never
# inline with anything timing-sensitive.
NVML_EVENT_BITS = (
    (0x01, "gpu_idle"),
    (0x02, "applications_clocks"),
    (0x04, "sw_power_cap"),
    (0x08, "hw_power_cap"),
    (0x10, "sw_thermal_slowdown"),
    (0x20, "hw_thermal_slowdown"),
    (0x40, "hw_slowdown"),
    (0x80, "sync_boost"),
)


class GpuCollector:
    """Samples NVML telemetry on a side thread and keeps a short history.

    Without nvmlInit_v2() every call returns NVML_ERROR_UNINITIALIZED instantly — the one
    trap here. When NVML is unusable the snapshot reports ok=false with a reason instead
    of fake zeros, and per-field failures leave that field absent for the same reason.
    """

    HISTORY = 60        # ~1 minute at 1 Hz
    PCIE_EVERY = 5      # every ~5 s; each call blocks ~31 ms on this thread

    ARGTYPES = {
        "nvmlDeviceGetHandleByIndex": [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
        "nvmlDeviceGetPowerUsage": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
        "nvmlDeviceGetClockInfo": [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        "nvmlDeviceGetTemperature": [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        "nvmlDeviceGetFanSpeed": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
        "nvmlDeviceGetCurrentClocksEventReasons": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)],
        "nvmlDeviceGetPowerManagementLimit": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
        "nvmlDeviceGetPcieThroughput": [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
    }
    REQUIRED = (
        "nvmlDeviceGetHandleByIndex",
        "nvmlDeviceGetPowerUsage",
        "nvmlDeviceGetClockInfo",
        "nvmlDeviceGetTemperature",
        "nvmlDeviceGetCurrentClocksEventReasons",
    )

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.lock = threading.Lock()
        self.hist: dict[str, list[float]] = {k: [] for k in ("power", "sm", "mem", "temp", "fan")}
        self.current: dict = {}
        self.pcie: dict = {"at": None, "rx_kb": None, "tx_kb": None}
        self.power_limit_w: float | None = None
        self.reason: str | None = None
        self._ticks = 0
        self.nvml, self.handle = self._open()
        if self.nvml is not None:
            threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        try:
            nvml = ctypes.CDLL("nvml.dll")
        except OSError as exc:
            self.reason = f"nvml.dll 不可用（{exc}）"
            return None, None
        init = getattr(nvml, "nvmlInit_v2", None)
        if init is None:
            self.reason = "nvmlInit_v2 未导出"
            return None, None
        init.restype = ctypes.c_uint
        if init() != 0:                                  # 0 == NVML_SUCCESS
            self.reason = "nvmlInit_v2 失败（无 NVIDIA 驱动？）"
            return None, None
        for name, argtypes in self.ARGTYPES.items():
            fn = getattr(nvml, name, None)
            if fn is None:
                if name in self.REQUIRED:
                    self.reason = f"{name} 未导出"
                    return None, None
                continue                                  # optional getter: skip quietly
            fn.restype = ctypes.c_uint
            fn.argtypes = argtypes
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex(0, ctypes.byref(handle)) != 0:
            self.reason = "GetHandleByIndex 失败（无 NVIDIA GPU？）"
            return None, None
        return nvml, handle

    def _loop(self) -> None:
        nvml, h = self.nvml, self.handle
        u, ull = ctypes.c_uint(), ctypes.c_ulonglong()
        sm_type, mem_type, temp_gpu, pcie_rx, pcie_tx = 1, 2, 0, 0, 1
        while True:
            sample: dict = {}
            if nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(u)) == 0:
                sample["power"] = u.value / 1000.0
                if self.power_limit_w is None and hasattr(nvml, "nvmlDeviceGetPowerManagementLimit"):
                    if nvml.nvmlDeviceGetPowerManagementLimit(h, ctypes.byref(u)) == 0:
                        self.power_limit_w = u.value / 1000.0
            if nvml.nvmlDeviceGetClockInfo(h, sm_type, ctypes.byref(u)) == 0:
                sample["sm"] = u.value
            if nvml.nvmlDeviceGetClockInfo(h, mem_type, ctypes.byref(u)) == 0:
                sample["mem"] = u.value
            if nvml.nvmlDeviceGetTemperature(h, temp_gpu, ctypes.byref(u)) == 0:
                sample["temp"] = u.value
            if hasattr(nvml, "nvmlDeviceGetFanSpeed") and nvml.nvmlDeviceGetFanSpeed(h, ctypes.byref(u)) == 0:
                sample["fan"] = u.value
            if nvml.nvmlDeviceGetCurrentClocksEventReasons(h, ctypes.byref(ull)) == 0:
                sample["events"] = ull.value
            with self.lock:
                self.current = sample
                for key in ("power", "sm", "mem", "temp", "fan"):
                    if key in sample:
                        self.hist[key].append(sample[key])
                        if len(self.hist[key]) > self.HISTORY:
                            del self.hist[key][0]
            self._ticks += 1
            if self._ticks % self.PCIE_EVERY == 0 and hasattr(nvml, "nvmlDeviceGetPcieThroughput"):
                rx, tx = ctypes.c_uint(), ctypes.c_uint()
                ok_rx = nvml.nvmlDeviceGetPcieThroughput(h, pcie_rx, ctypes.byref(rx)) == 0
                ok_tx = nvml.nvmlDeviceGetPcieThroughput(h, pcie_tx, ctypes.byref(tx)) == 0
                if ok_rx or ok_tx:
                    with self.lock:
                        self.pcie = {
                            "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                            "rx_kb": rx.value if ok_rx else None,
                            "tx_kb": tx.value if ok_tx else None,
                        }
            time.sleep(self.interval)

    def snapshot(self) -> dict:
        with self.lock:
            current = dict(self.current)
            hist = {k: list(v) for k, v in self.hist.items()}
            pcie = dict(self.pcie)
            limit = self.power_limit_w
        mask = current.get("events")
        active = [key for bit, key in NVML_EVENT_BITS if mask is not None and (mask & bit)]
        return {
            "ok": self.nvml is not None,
            "reason": self.reason,
            "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "power_limit_w": limit,
            "current": current,
            "event_reasons": {"mask": mask, "active": active},
            "history": hist,
            "pcie": pcie,
            "samples": max((len(v) for v in hist.values()), default=0),
        }


class EngineCollector:
    """Polls the engine's /monitor endpoints on this process's own threads.

    One process serves three sources (LMCache, NVML, engine). The engine poll is a
    control-path read like the others: /monitor/snapshot copies already-published counters,
    so a 1 Hz poll cannot perturb kernel timing, PPL or the hash gates.

    Rates come from differencing cumulative engine counters and the wall time between two
    polls of this side. Nothing is interpolated: a failed poll is recorded as a gap and the
    rate fields go None, which the page draws as a broken line. Fields the engine does not
    publish stay absent rather than zero.
    """

    HISTORY = 12         # 1 Hz points kept for the ~5 s window rate and its min/max band
    EVENT_TAIL = 64      # events handed to the page per poll
    INFO_EVERY = 1.0     # /monitor/info is static; still polled at the same 1 Hz cadence
    WINDOW = 5.0         # engine's own throughput log uses a 5 s interval

    def __init__(self, base: str, interval: float = 1.0, api_key: str | None = None):
        self.base = base.rstrip("/")
        self.interval = interval
        # Same credential the engine's own pre-routing handler checks; empty means the engine was
        # started without one (the local default), and no header is sent.
        self.api_key = api_key or ""
        self.lock = threading.Lock()
        self.data: dict = {}
        self.info: dict | None = None
        self.info_at = 0.0
        self.events: collections.deque = collections.deque(maxlen=256)
        self.history: collections.deque = collections.deque(maxlen=self.HISTORY)
        self.counter_hist: collections.deque = collections.deque(maxlen=self.HISTORY)
        self.prev: dict | None = None
        self.failures = 0
        self.ok = False
        self.reason = "尚未采样"
        self.gaps = 0
        self.gap_seconds = 0.0
        self.last_success = 0.0
        self.down = False
        self.event_ok = False
        self.event_reason = "尚未连接"
        self.event_gaps = 0
        self.event_down = False
        self.samples = 0
        self._stop = threading.Event()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        threading.Thread(target=self._events_loop, daemon=True).start()

    # -- transport -------------------------------------------------------
    def request(self, path: str, *, data: bytes | None = None, method: str | None = None,
                timeout: float = 2.5) -> urllib.request.Request:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if data is not None:
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(self.base + path, data=data, headers=headers,
                                      method=method)

    def _get_json(self, path: str, timeout: float = 2.5) -> dict:
        with urllib.request.urlopen(self.request(path, timeout=timeout), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))

    @staticmethod
    def _counter(value) -> float:
        return float(value) if isinstance(value, (int, float)) else 0.0

    # -- snapshot polling ------------------------------------------------
    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            at = time.monotonic()
            try:
                snapshot = self._get_json("/monitor/snapshot")
                if not isinstance(snapshot, dict):
                    raise ValueError("snapshot is not an object")
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                self._failed(f"/monitor/snapshot: {exc}", at)
                time.sleep(self.interval)
                continue

            runtime = (snapshot.get("engine") or {}).get("runtime") or {}
            counters = {
                "prompt_tokens": self._counter(runtime.get("prompt_tokens")),
                "prefill_tokens": self._counter(runtime.get("computed_prefill_tokens")),
                "decode_tokens": self._counter(runtime.get("committed_decode_tokens")),
                "generated_tokens": self._counter(runtime.get("generated_tokens")),
                "decode_rounds": self._counter(runtime.get("decode_rounds")),
                "running_requests": self._counter(runtime.get("running_requests")),
                "prefilling_requests": self._counter(runtime.get("prefilling_requests")),
                "waiting_requests": self._counter(runtime.get("waiting_requests")),
            }
            service = snapshot.get("service") or {}
            ttft = service.get("ttft") or {}
            ttft_bins = ttft.get("bins") or []
            ttft_count = self._counter(sum(ttft_bins)) if ttft_bins else 0.0
            ttft_sum = self._counter(ttft.get("sum_seconds"))

            point = {"at": at, "decode": counters["decode_tokens"],
                     "prefill": counters["prefill_tokens"]}
            rate_1s = None
            raw = None
            if self.prev is not None:
                dt = at - self.prev["at"]
                if dt <= 0:
                    dt = 1e-3
                d_decode = counters["decode_tokens"] - self.prev["decode"]
                d_prefill = counters["prefill_tokens"] - self.prev["prefill"]
                if d_decode < 0 or d_prefill < 0:
                    # Counter went backwards: the engine restarted. Not a rate.
                    with self.lock:
                        self.gaps += 1
                    self.prev = None
                    self.history.clear()
                    self.counter_hist.clear()
                    point["decode_rate"] = None
                    point["prefill_rate"] = None
                else:
                    rate_1s = {
                        "decode_tps": d_decode / dt,
                        "prefill_tps": d_prefill / dt,
                        "ttft_ms": (1000.0 * (ttft_sum - self.prev["ttft_sum"]) /
                                    (ttft_count - self.prev["ttft_count"])
                                    if ttft_count > self.prev["ttft_count"] else None),
                    }
                    point["decode_rate"] = rate_1s["decode_tps"]
                    point["prefill_rate"] = rate_1s["prefill_tps"]
                    raw = {"window_seconds": dt, "decode_tokens": d_decode,
                           "prefill_tokens": d_prefill,
                           "ttft_samples": ttft_count - self.prev["ttft_count"],
                           "ttft_sum_seconds": ttft_sum - self.prev["ttft_sum"]}
            else:
                point["decode_rate"] = None
                point["prefill_rate"] = None

            self.history.append(point)
            self.counter_hist.append(point)

            window = self._window_rate()
            self.prev = {"at": at, "decode": counters["decode_tokens"],
                         "prefill": counters["prefill_tokens"],
                         "ttft_sum": ttft_sum, "ttft_count": ttft_count}
            if self.down:
                self.down = False
            self.ok = True
            self.reason = ""
            self.failures = 0
            self.last_success = at
            self.samples += 1
            self._refresh_info(at)
            self._publish(snapshot, counters, service, window, rate_1s, raw)
            time.sleep(self.interval)

    def _window_rate(self) -> dict:
        """Rate over the newest ~5 s of 1 Hz points, plus the band of the 1 Hz points."""
        if len(self.history) < 2:
            return {"window_seconds": None, "decode_tps": None, "prefill_tps": None,
                    "ttft_ms": None, "band": {"decode": None, "prefill": None, "ttft": None}}
        newest = self.history[-1]
        oldest = self.history[0]
        span = newest["at"] - oldest["at"]
        decode_band = [p["decode_rate"] for p in self.history if p["decode_rate"] is not None]
        prefill_band = [p["prefill_rate"] for p in self.history if p["prefill_rate"] is not None]
        band = {
            "decode": [min(decode_band), max(decode_band)] if decode_band else None,
            "prefill": [min(prefill_band), max(prefill_band)] if prefill_band else None,
            "ttft": None,
        }
        if span < self.WINDOW * 0.8:
            # Not enough of a window yet: report nothing rather than a 1-point "5 s" rate.
            return {"window_seconds": span if span else None, "decode_tps": None,
                    "prefill_tps": None, "ttft_ms": None, "band": band}
        return {
            "window_seconds": span,
            "decode_tps": (newest["decode"] - oldest["decode"]) / span,
            "prefill_tps": (newest["prefill"] - oldest["prefill"]) / span,
            "ttft_ms": None,
            "band": band,
        }

    def _refresh_info(self, at: float) -> None:
        if self.info is not None and at - self.info_at < self.INFO_EVERY:
            return
        try:
            info = self._get_json("/monitor/info")
            if isinstance(info, dict):
                self.info = info
                self.info_at = at
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass  # identity is optional; the snapshot is the source of truth for numbers

    def _publish(self, snapshot: dict, counters: dict, service: dict, window: dict,
                 rate_1s: dict | None, raw: dict | None) -> None:
        with self.lock:
            self.data = {
                "ok": True,
                "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "base": self.base,
                "error": None,
                "info": self.info,
                "info_age_seconds": (round(time.monotonic() - self.info_at, 1)
                                     if self.info else None),
                "snapshot": snapshot,
                "counters": counters,
                "rate": {
                    "decode_tps_1s": (rate_1s or {}).get("decode_tps"),
                    "prefill_tps_1s": (rate_1s or {}).get("prefill_tps"),
                    "ttft_ms_1s": (rate_1s or {}).get("ttft_ms"),
                    "decode_tps": window["decode_tps"],
                    "prefill_tps": window["prefill_tps"],
                    "window_seconds": window["window_seconds"],
                    "band": window["band"],
                },
                "last_interval": raw,
                "service": service,
                "events": self._recent_events(),
                "recent_ttft_ms": self._recent_ttft_ms(),
                "health": {"samples": self.samples, "failures": self.failures,
                           "gaps": self.gaps, "gap_seconds": round(self.gap_seconds, 1),
                           "last_success_seconds_ago":
                               round(time.monotonic() - self.last_success, 1)
                               if self.last_success else None,
                           "event_ok": self.event_ok, "event_reason": self.event_reason,
                           "event_gaps": self.event_gaps},
            }

    def _recent_events(self) -> list:
        return list(self.events)[-self.EVENT_TAIL:]

    def _recent_ttft_ms(self) -> list:
        """First-token observations from the last 15 s, in milliseconds.

        The histogram only carries buckets; the SSE events carry the exact per-request
        value, which is what the page plots and what the acceptance compares against the
        engine's own `ttft=` log line.
        """
        cutoff = (time.time() - 15.0) * 1000.0
        values = []
        for event in list(self.events):
            if event.get("kind") != "first_token":
                continue
            details = event.get("details") or {}
            seconds = details.get("ttft_seconds")
            if seconds is None or event.get("timestamp_unix_ms", 0) < cutoff:
                continue
            values.append(round(float(seconds) * 1000.0, 1))
        return values

    def _failed(self, reason: str, at: float) -> None:
        self.failures += 1
        self.ok = False
        self.reason = reason
        self.history.clear()
        self.counter_hist.clear()
        self.prev = None
        if not self.down:
            self.down = True
            self.gaps += 1
        if self.last_success:
            self.gap_seconds = at - self.last_success
        with self.lock:
            self.data = {
                "ok": False,
                "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "base": self.base,
                "error": reason,
                "info": self.info,
                "snapshot": None,
                "counters": None,
                "rate": {"decode_tps": None, "prefill_tps": None, "window_seconds": None,
                         "band": {"decode": None, "prefill": None, "ttft": None}},
                "service": None,
                "events": self._recent_events(),
                "recent_ttft_ms": [],
                "health": {"samples": self.samples, "failures": self.failures,
                           "gaps": self.gaps, "gap_seconds": round(self.gap_seconds, 1),
                           "event_ok": self.event_ok, "event_reason": self.event_reason,
                           "event_gaps": self.event_gaps},
            }

    # -- event subscription ----------------------------------------------
    def _events_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                request = self.request("/monitor/events", timeout=30)
                request.add_header("Accept", "text/event-stream")
                with urllib.request.urlopen(request, timeout=30) as response:
                    backoff = 1.0
                    self.event_ok, self.event_reason, self.event_down = True, "", False
                    kind = None
                    for raw_line in response:
                        if self._stop.is_set():
                            return
                        line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                        if line.startswith("event:"):
                            kind = line[6:].strip()
                        elif line.startswith("data:"):
                            try:
                                payload = json.loads(line[5:].strip())
                            except ValueError:
                                continue
                            if not isinstance(payload, dict):
                                continue
                            payload.setdefault("kind", kind or "event")
                            with self.lock:
                                self.events.append(payload)
                        # blank line / ":" keep-alive need no handling
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                self.event_ok = False
                self.event_reason = f"/monitor/events: {exc}"
                if not self.event_down:
                    self.event_down = True
                    self.event_gaps += 1
                time.sleep(backoff)
                backoff = min(5.0, backoff * 1.5)

    # -- payload ---------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            if self.data:
                return dict(self.data)
        return {"ok": False, "base": self.base, "error": self.reason,
                "rate": {"decode_tps": None, "prefill_tps": None, "window_seconds": None},
                "events": [], "recent_ttft_ms": [], "service": None, "info": None,
                "snapshot": None}


class Handler(BaseHTTPRequestHandler):
    page: Path = Path("35b-moe-console.html")
    upstream: str = "http://127.0.0.1:18080/metrics"
    gpu: GpuCollector | None = None
    engine: EngineCollector | None = None

    def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/35b-moe-console.html"):
            try:
                body = self.page.read_bytes()
            except OSError as exc:
                self._send(500, f"cannot read {self.page}: {exc}".encode(), "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/lmcache.json":
            snapshot = build_snapshot(self.upstream)
            self._send(200, json.dumps(snapshot, ensure_ascii=False).encode(), "application/json")
            return
        if path == "/api/gpu.json":
            body = self.gpu.snapshot() if self.gpu is not None else {"ok": False, "reason": "collector not started"}
            self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")
            return
        if path == "/api/engine.json":
            body = self.engine.snapshot() if self.engine is not None else {
                "ok": False, "error": "engine collector not started",
                "rate": {"decode_tps": None, "prefill_tps": None}}
            self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")
            return
        if path == "/api/engine/snapshot.json":
            body = self.engine.snapshot() if self.engine is not None else {"ok": False}
            snapshot = body.get("snapshot")
            if not snapshot:
                self._send(503, json.dumps({"ok": False, "error": body.get("error")},
                                           ensure_ascii=False).encode(), "application/json")
                return
            self._send(200, json.dumps(snapshot, ensure_ascii=False).encode(), "application/json")
            return
        if path == "/api/engine/events.json":
            body = self.engine.snapshot() if self.engine is not None else {"events": []}
            self._send(200, json.dumps({"ok": body.get("ok", False),
                                        "events": body.get("events", [])},
                                       ensure_ascii=False).encode(), "application/json")
            return
        if path == "/api/metrics.txt":
            try:
                with urllib.request.urlopen(self.upstream, timeout=3.0) as response:
                    body = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self._send(502, str(exc).encode(), "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    # -- T131: the control plane's write path ---------------------------------
    # The console is served from this origin, so its POSTs land here and are forwarded to the
    # engine's /control/* endpoints with the same credential the polls use. The engine's answer
    # (status and body) is returned unchanged: this proxy never re-writes a refusal into an OK.
    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        path = self.path.split("?", 1)[0]
        routes = {
            "/api/control/mode": "/control/mode",
            "/api/control/freeze": "/control/freeze",
        }
        target = routes.get(path)
        if target is None:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        engine = self.engine
        if engine is None:
            self._send(503, json.dumps(
                {"ok": False, "error": "engine collector not started"},
                ensure_ascii=False).encode(), "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            with urllib.request.urlopen(
                    engine.request(target, data=body, method="POST", timeout=10.0),
                    timeout=10.0) as response:
                payload = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            # A refusal (400 invalid mode/value, 500 engine error) belongs to the caller.
            payload = exc.read()
            status = exc.code
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._send(502, json.dumps(
                {"ok": False, "error": "engine unreachable: " + str(exc)},
                ensure_ascii=False).encode(), "application/json")
            return
        self._send(status, payload, "application/json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Console + LMCache metrics proxy")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--upstream", default="http://127.0.0.1:18080/metrics")
    parser.add_argument("--page", default=None, help="path to the console html")
    parser.add_argument("--gpu-interval", type=float, default=1.0, help="NVML sampling interval in seconds")
    parser.add_argument("--engine", default="http://127.0.0.1:11438",
                        help="ninfer serve base URL; 'none' disables the engine source")
    parser.add_argument("--engine-key", default="",
                        help="API key for an engine started with one (sent as a Bearer token)")
    args = parser.parse_args()

    page = Path(args.page) if args.page else Path(__file__).resolve().parent / "35b-moe-console.html"
    if not page.is_file():
        print(f"console page not found: {page}", file=sys.stderr)
        return 2

    Handler.page = page
    Handler.upstream = args.upstream
    Handler.gpu = GpuCollector(args.gpu_interval)
    Handler.engine = (EngineCollector(args.engine, api_key=args.engine_key)
                      if args.engine and args.engine != "none" else None)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"console  : http://{args.host}:{args.port}/")
    print(f"metrics  : http://{args.host}:{args.port}/api/lmcache.json  (upstream {args.upstream})")
    gpu_state = "NVML OK" if Handler.gpu.nvml is not None else f"NVML 不可用（{Handler.gpu.reason}）"
    print(f"gpu      : http://{args.host}:{args.port}/api/gpu.json  ({gpu_state})")
    print(f"engine   : http://{args.host}:{args.port}/api/engine.json  "
          f"({Handler.engine.base if Handler.engine else 'disabled'})")
    print("ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
