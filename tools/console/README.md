# Local console

The single-machine operations front end for `ninfer-serve`: controller modes and expert freeze
policy (writes), plus the readings the engine, LMCache and the GPU publish (reads).

```bat
console.cmd                          :: engine defaults to http://127.0.0.1:11438
console.cmd http://127.0.0.1:9908    :: another serve instance
```

Then open **http://127.0.0.1:8099/**. Keep the window open; Ctrl+C stops it.

## Why it is a server and not a page

`lmcache_console.py` exists because LMCache's metrics endpoint sends no CORS headers, so a page
opened from `file://` cannot read it. Serving the page and the metrics from one origin removes
the need for a CORS exception, and lets the same origin carry three more sources:

| Route | Source |
|---|---|
| `/api/lmcache.json` | LMCache Prometheus metrics, parsed into named fields |
| `/api/gpu.json` | NVML telemetry (power/clocks/temperature/fan/PCIe), sampled on a side thread |
| `/api/engine.json` | the engine's `/monitor/info|snapshot|events`, 1 Hz, one event subscription |
| `POST /api/control/mode`, `POST /api/control/freeze` | forwarded verbatim to the engine's `/control/*` |

All of it is control-path only: it never touches a CUDA context, so it cannot move kernel timing,
PPL or the hash gates.

## Honesty rules the page follows

* When the engine is unreachable the payload says `ok=false` with a reason and the page labels
  its fallback as 演示数据 — it never presents a simulation as a measurement.
* A failed poll marks a gap; the line breaks instead of interpolating.
* Quantities the engine does not measure are shown as "引擎未发布", not as zero.
* Control writes are shown with the engine's own answer (`applied` / `queued`) and its sequence
  number; a refusal (400) is displayed with the engine's reason.

## Requirements

Python 3.10+ on `PATH`. Nothing else: the NVML read degrades to "absent" when the DLL is not
found, and the LMCache source is optional (start it only if LMCache runs on 18080).
