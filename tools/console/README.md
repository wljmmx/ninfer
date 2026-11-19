# Local console

The single-machine operations front end for `ninfer-serve`: pick a model, start and stop the
engine, and read what the engine, LMCache and the GPU publish.

```bat
console.cmd                          :: engine defaults to http://127.0.0.1:11438
console.cmd http://127.0.0.1:9908    :: another serve instance
python lmcache_console.py --models-root D:\models
```

Then open **http://127.0.0.1:8099/**. Keep the window open; Ctrl+C stops it.

## Launching from the page

The page has no hard-coded model. It lists `.ninfer` files under the configured models roots
(`--models-root`, default `R:\model` and the release's own `..\model`), and asks this server to
start one. **选择目录…** opens the host's own folder picker, so a model that lives anywhere on
the machine can be reached: the chosen directory is added to the roots for this session, and a
directory holding exactly one `.ninfer` selects that file straight away.

| Route | Does |
|---|---|
| `GET /api/models.json?dir=<abs path>` | one level of directories and `.ninfer` files under a root |
| `POST /api/dialog/folder` | opens the host folder picker (`initial`), adds the result as a root |
| `POST /api/launch` | `{model, port, max_context, max_concurrency, prefill_chunk, kv_dtype, extra}` |
| `POST /api/launch/stop` | terminates the engine this server started |
| `GET /api/launch/status.json` | pid, uptime, the exact command line, and the log tail |

The picker is a native Windows dialog owned by the server process, which is why the console has to
run in an interactive desktop session. A browser cannot supply this: its directory chooser hands
back a sandboxed name, never the host path the engine needs.

The server parses every field, range-checks it, refuses a model outside the roots, and spawns
`ninfer-serve.exe` without a shell, so a value from the page can never become a command. One
child at a time: a second engine on the same GPU would corrupt every number the page reports.
A POST from another origin is refused; only this page may start a process.

## Why it is a server and not a page

`lmcache_console.py` exists because LMCache's metrics endpoint sends no CORS headers, so a page
opened from `file://` cannot read it. Serving the page and the metrics from one origin removes
the need for a CORS exception, and lets the same origin carry the rest:

| Route | Source |
|---|---|
| `/api/lmcache.json` | LMCache Prometheus metrics, parsed into named fields |
| `/api/gpu.json` | NVML telemetry (power/clocks/temperature/fan/PCIe), sampled on a side thread |
| `/api/engine.json` | the engine's `/monitor/info|snapshot|events`, 1 Hz, one event subscription |
| `POST /api/control/mode`, `POST /api/control/freeze` | forwarded verbatim to the engine's `/control/*` |

All of it is control-path only: it never touches a CUDA context, so it cannot move kernel timing,
PPL or the hash gates.

## Honesty rules the page follows

* Every number on the page is a measurement. There is no simulated series, no synthetic request
  stream and no stand-in curve: when no engine is running the readings are `—` and the page says
  so. The earlier demo generator was deleted rather than relabelled.
* A failed poll marks a gap; the rate window restarts instead of interpolating.
* Quantities the engine does not publish stay `—`, not zero.
* Control writes are shown with the engine's own answer.

## Requirements

Python 3.10+ on `PATH`. Nothing else: the NVML read degrades to "absent" when the DLL is not
found, and the LMCache source is optional (start it only if LMCache runs on 18080).

## LMCache

The panel's **启动 LMCache** button runs `start_lmcache.cmd` next to this file: the script owns the
command (`lmcache_server` on PATH, else `python -m lmcache.server`, on port 18080, the port the
panel reads), the console only decides when to run it and reports the child's pid, exit code and
log tail. If LMCache is not installed the button is honest about it - the log says so and the exit
code is 3.