# Grammar-constrained requests return 400 — root cause and the one remaining patch

Date: 2026-11-21. Engine: RTX 4090 / sm_89 / CUDA 13.5, Qwen3.8-27B, `--spec mtp --draft-tokens 7`.
Status: **engine usable**; constrained requests (`response_format` / tools) can still answer 400.

## 1. What was fixed (committed)

`4f120123` — a grammar violation fails the REQUEST, not the engine.

Before: `GrammarSession::accept` threw `std::logic_error` out of the output policy; the worker treats any
escape as fatal (`fail_all_locked`), set `failed_`, and every later request answered 503.

After: `text::GrammarViolation` (dedicated type) is caught per row in `EngineCore::commit` →
`pending.mark_constraint_failed(row)` + `failed[row] = true` + `decision{terminal, failed}`.

Measured: 12 constrained requests → 3-6 answered 200, the rest 400, **engine stayed `ready` throughout**.
Before the fix the second request took the whole engine down.

Two contract checks forced the shape of that fix (both hit in order while iterating):

| error | what it required |
|---|---|
| `pending transaction decision is invalid` | the program must mark the row constraint-failed before `commit()` |
| `Runtime commit row disposition is invalid` | the engine must set its own `failed[row]` |

## 2. The remaining defect (proven)

```
[grammar] rejected set: 1144 248045 74455 0 | id0=!
```

`token id 0` decodes to the **ordinary character `!`** — not a pad, not EOS, not an uncomputed column.
The committed run ends with `!` at a position where the grammar expects a structural character
(`}` / `"` / `:` …), so the host matcher rejects the run.

`decode.cpp` proves the bonus slot is mandatory: `count_i == accepted + 1` is validated
(`MTP batch returned invalid row metadata` when a run is emitted without it).

So the defect is: **the bonus token is the raw argmax of the verify head and never passes the grammar
mask.** Every draft token is masked (the mask bitmap has `draft_window + 1` positions); the extra
position is only consumed by the *next* round's sampler, never by the argmax that fills
`target_tokens[accepted]`.

Hypotheses already disproved (do not repeat them):

| attempt | outcome |
|---|---|
| clamp the licensed prefix before commit (`grammar_dead_positions`) | bitmap is always 0 → never fires |
| truncate in the output policy | `output policy returned an invalid licensed prefix` (non-finished rounds must accept the full extent) |
| unify `target_tokens` row stride (`k + 1` → `tensor.ne[0]`) | compiles, no behaviour change |
| drop the bonus when it is 0 | `MTP batch returned invalid row metadata`, engine unusable (12/12 503) — reverted |

## 3. The patch to apply

**SUPERSEDED — see section 7. The plan below is kept only as the record of what was tried; it names the
wrong header (`apply_sampling_mask` lives in `include/ninfer/ops/speculative_round.h`) and, more
importantly, it masks the bonus column only, which the sampled accept path never needed: that path had
its own defect.**

Existing API to reuse: `ops::apply_sampling_mask(logits, masked, mask, ...)`
(`include/infer/ops/sampling.h`) and the `(row, position)` mask addressing already used by
`TextContext::mtp_propose_topk_strided` (`mask_column` + `mask_window_words`).

1. `src/models/qwen3_5/execution/text.h` — both `target_verify_batch` overloads gain
   `const ops::SamplingConfig* configs = nullptr` (default keeps today's behaviour).
2. `src/models/qwen3_5/execution/text.cpp` — after `ops::argmax(flat_logits, flat_tokens, vocab, stream)`:

   * for each row with `configs[row].mask.words != nullptr` and `valid[row] > 0`:
     * `col = valid[row] - 1` (the bonus column), `words = (vocab + 31) / 32`,
     * the row's mask window for that position is `configs[row].mask.words + col * words`,
     * the logits of one column are contiguous (`element(v, c) = c * vocab + v`), so copy `vocab`
       elements at offset `col * vocab` into a scratch `[vocab, 1]` tensor,
     * `apply_sampling_mask(scratch, masked, mask_at_col)`, `argmax(masked, token, vocab)`,
     * write the result back to `target_tokens[row * stride + col]`.
3. `src/models/qwen3_5/program/speculative/target_verification.cpp` — pass `frame.sampling`
   (already a member of `TargetVerifyFrameView`, `execution_context.h:155`) at both call sites.

Keep the whole block behind the existing `configs != nullptr` guard so an unconstrained request is
byte-identical to today.

## 4. Acceptance

* the 12-shot JSON-schema loop answers 12/12 (today: 3-6), `[grammar] rejected set` never prints;
* engine `ready` throughout;
* `py -3 H:\TeleAgentWorks\.temp\kv\regress.py --tier fast` → PPL **7.359779** bit-exact,
  dump **e95cacb864d8b1ab**, regressions 0.

## 5. Build discipline learned the hard way (2026-11-21)

1. **Two executables in the release dir.** A stray `infer-serve.exe` (a `Copy-Item` with the `n` missing)
   held port 11438, so every `ninfer-serve.exe` start exited with `cannot bind` and requests were served by
   an OLD binary — several "the fix did not work" conclusions were measured against the wrong process.
   After every start: `Get-NetTCPConnection -LocalPort 11438` must show the pid you just started.
2. **ninja does not always track header dependencies.** `engine.cpp.obj` stayed at 11:28 while
   `engine_core.h` changed at 12:10 and ninja answered `no work to do`. Delete the affected `*.obj`
   after touching `engine_core.h` / `observations.h`.
3. **Never rewrite a source file with `Set-Content`.** A null `$t` truncated a file to 0 bytes
   (it happened to be an untracked empty stray). Use the edit tool, or verify the file size afterwards.

## 6. Current state

* Engine running with `--spec mtp --draft-tokens 7`, checkpoint cache off, 256k fully resident.
* Committed: `4f120123` (stability), `7853decc` (diagnostics: `[grammar] provider saw/matcher got`,
  `[mtp] target_tokens | columns | valid`, `[ckpt] capture stage/publish`, `phases`).
* Not pushed: `cd35824f`, `c7cd47c7`, `f70b5e5e`, `4f120123`, `7853decc`.
* Uncommitted: the `rejected set / id0` probe in `output_session.cpp` (keep it — it is what proved the
  token is `!`).

## 7. Resolved 2026-11-21 (the two defects, and what actually closed them)

Two separate defects produced the 400s, and only fixing both makes 12/12 pass.

**7.1 The verify rectangle was argmaxed without the grammar bitmap** (`e1f4fcc9`). The rectangle's
columns are already the bitmap's own `(row, position)` order, so the projection's output is masked with
the request's bitmap before the argmax. An unconstrained row keeps an all-ones block, so the pass has no
per-request branch (one captured-graph path) and is bit-identical for unconstrained requests — the PPL
gate confirms it. The masked copy and the bitmap are priced in `target_body` (`planning/startup.cpp`),
which mirrors this pass's allocations; an allocation here that `target_body` does not reserve throws at
run time.

Graph-unsafe detail worth remembering: any *host* value read while building the pass (for example
`row_sampling[row].mask.words`) is baked into a captured graph. That is why the row masks are addressed
by a fixed device base plus a per-column stride and their contents are refreshed on the device every
round — the pointer must be stable, only the bits may change.

**7.2 The sampled accept kept drafts the target cannot emit** (`366383b3`). This was the actual cause of
"all seven drafts accepted" on a constrained request. In `speculative_sparse_warp_accept`,
`reject = !(pd >= qd || u * qd < pd)` treats `pd == 0` as acceptable whenever `qd == 0` too: an illegal
draft has no mass in the masked target distribution (`pd == 0`) and the proposal head is deliberately
unmasked (`qd == 0`), so zero versus zero kept it. The rule is now `pd > 0` to accept, i.e. exact
rejection sampling (`min(1, pd/qd)` is zero when `pd` is zero).

The temperature arm test that separated them (`probe_grammar_arms.py`): greedy 4/4 passed while
temperature 0.7 failed 4/4 — the greedy accept compares against the target argmax, so it never needed
the accept fix, and the sampled path never needed the bonus-column argmax fix.

Verified after both: 12/12 constrained requests answer 200, engine `ready` throughout, no
`[grammar] rejected set`; `regress.py --tier fast` → PPL **7.359779** bit-exact, dump
**e95cacb864d8b1ab**, regressions 0; throughput on the constrained request 92.5 tok/s
(142 of 394 drafts accepted).

Note on an empty `content` in those tests: the JSON-schema requests spent the whole `max_tokens` on
`reasoning_content` (`finish_reason: length`), so `content` was empty while the request succeeded. Check
`constraint.complete` / `finish_reason` before reading an empty body as a failure.

## 8. LMCache path, measured 2026-11-21 (blocked on the import, not on the publish)

State: the engine can store and can fetch, but a fetched record cannot be brought into a live
request, so attaching the cache today turns a hit into a hard 500 - worse than a miss.

What was measured:

| step | result |
|---|---|
| store, 1916-token prefix | `[lmcache] store 28.1 MiB in 1.660 s (17 MiB/s) | ok=1`, `publish covers 1909 of 1916` |
| restore after a restart, same prefix | `[lmcache] restore: match 0.002 s, transfer 2.876 s`, `fetch 27525120 KV byte(s) + 155920388 state byte(s)` |
| the request that used it | `HTTP 500`, worker: `KV coverage exceeds reserved unit growth: tokens=1921 required_pages=31 mapped_pages=0 reserved_pages=1` |
| same test with the synchronous store (no `NINFER_LMCACHE_ASYNC_WRITE`) | identical 500 |

So `NINFER_LMCACHE_ASYNC_WRITE` is not what produced the failures attributed to it; the import is.
`[ckpt] adopt: written epoch=3 read-back epoch=3` shows the restored pages reach a checkpoint handle
(`ResourceManager::...` -> `import_checkpoint_current_layout`), but the live KV address that then runs
`ensure_mapped_to_tokens` has `page_count = 0` and a one-page reservation, while the restored frontier
needs 31. The address has to take the checkpoint's membership before that call (or the request plan has
to reserve from the imported page count instead of from the restore frontier).

Two side findings worth keeping:

* `--max-context` is not a capacity promise. `--max-context 200000 --kv-capacity 200000` started with
  693 pages = 44k tokens and `free 0 B`, so a 163k prompt was refused with `context length exceeded`.
  Read the `capacity |` line at startup, not the flag.
* 256k full residency on this card only holds while ~7 GiB stays free. When the desktop took that
  memory, decode fell from 92 to 1.4-46 tok/s with `devwait` in the minutes and the GPU at 15 % /
  92 W: WDDM was re-reading the weights every step. 128k KV gives 98-109 tok/s with 2.7 GiB spare.
  This, not LMCache, is what made a 163k request look like it took 12 minutes.

## 9. The LMCache 500, root-caused and fixed (2026-11-21)

Section 8 recorded the symptom. The cause was in binding, not in the cache.

`bind` (`transactions/binding.cpp`) set `backend_frontier = reuse_frontier - 1`
whenever the request ran MTP and reused a prefix. An in-process capture carries a
draft KV in its export, so that is true there. A record that arrives as bytes does
not: `CheckpointExport` (`program.h`) has `frontier / page_count / page_bytes /
epochs / kv_bytes` and nothing for the backend. So `record.kv->backend` was empty,
the backend address took the cold branch with zero pages, and the first decode
step asked `ensure_mapped_to_tokens` for 1921 tokens against `reserved_pages=1`:

    KV coverage exceeds reserved unit growth: tokens=1921 required_pages=31
    mapped_pages=0 reserved_pages=1

`NINFER_BIND_PROBE=1` shows both binds of that round, which is what named it:

    [bind] lane=0 prefix=1909 target=1916 growth=0 branch=prefix-fork dest_pages=0 source=1
    [bind] lane=0 prefix=1908 target=1921 growth=1 branch=cold dest_pages=0 source=0

Fixes (`6f51dee0`):

* the backend claim is gated on the source actually having a backend history;
* `Program::accepts_byte_records()` is asked before the transfer, so a
  speculative-backend engine declines the record for free instead of paying
  2.85 s (16k) / minutes (163k) before refusing;
* `import_checkpoint_current_layout` refuses the same case with a reason, which
  is the rule `tier_open` already states for the disk tier;
* a record covering the whole prompt is declined (its zero-suffix path needs a
  tail hidden a byte record does not carry).

Measured after the fix:

| case | before | after |
|---|---|---|
| `--spec mtp`, hit | 4.27 s then 500 | restore declined before transfer, cold prefill 1.36 s, 200 |
| `--spec none`, hit | 500 | `cache_n 1909`, `branch=prefix-fork`, 200 |
| `--spec none`, after restart | - | `restore: match 0.002 s, transfer 2.8 s`, `cache_n 1909`, 200 |

Still true, and the next thing to fix if the cache is to pay for itself: this
LMCache server ingests and serves at ~16 MiB/s on this box (28.1 MiB in 1.66-1.79 s
on store; 2.7-2.9 s to transfer the same block back), while the model prefills at
~2.4 GB/s. At 1916 tokens the restore is therefore slower than the prefill it
replaces, and at 163k it would cost minutes. The engine side is correct now; the
throughput is the cache's.

## 10. Where the L3 cache actually lives (verified 2026-11-21)

Question: is the checkpoint cache on an SSD? Answer: it is not on a disk at all, and the disk WSL
does have is spinning rust.

Evidence:

* `tools/console/l3_lmcache.sh` starts the server with `--l1-size-gb 4` and no `--l2-adapter`. The
  server log starts an `L2EvictionController` but no L2 connector is ever constructed.
* After stores, there is no `~/.lmcache` and no `/tmp/lmcache*`; the server's RSS is 4.4 GiB, i.e.
  the L1 arena. Restarting the server (or WSL) loses every record - only an engine restart keeps
  them, because the server process survives it.
* The WSL distro's disk is `H:\Vm_hosts\Vms\WSL\Ubuntu-26.04\ext4.vhdx`, and `H:` is the Storage
  Space `Raid0` built from three `ST8000NM017B` SATA **HDD**s. So anything written under `$HOME`
  inside WSL lands on a 3-disk HDD array.
* The only NVMe visible here is `R:` (`ZHITAI TiPlus9100 2TB`). WSL reaches it as `9p` at
  `/mnt/r`, measured `dd` 331-360 MB/s - the 9p bridge, not the device.
* `dd` inside WSL on `$HOME` reports 3.7 GB/s buffered / "6.3 GB/s direct": the VHDX and the host
  page cache absorb the write, so that number describes neither the HDD array nor what a cache
  would see.

Throughput measured end to end (engine `[lmcache] store`, 28.1 MiB block) with worker counts set by
`L3_WORKERS`: 4 -> 12 MiB/s, 8 -> 20 MiB/s, 16 -> 11 MiB/s. No disk is involved in any of those runs
(L1 only), so the ceiling is the engine<->server path (ZMQ framing plus the Python object handling),
and raising workers does not move it reliably.

Consequence for the next step: an SSD tier is a configuration change (`--l2-adapter` with
`{"type": "fs_native", "base_path": ...}`), but it has to be a path that is actually on NVMe - a
dedicated WSL disk on `R:` rather than `$HOME` (HDD) or `/mnt/r` (9p). Adding it without fixing the
transport will make stores slower, not faster, because L1 is already the fast tier.

## 11. Why the in-memory cache path is slow (measured 2026-11-21)

The memory tier is not the limit. Inside the server, during a store the traced `StorageManager`
calls took `reserve_write` 0.14 ms and `finish_write` 2.9-8.2 ms; during a restore every call was
0.04-16 ms. The whole cost sits outside them.

Split of one turn from the client (`NINFER_LMCACHE_IO_PROBE=1`, `[lmcache io]` lines):

    restore: op=18 bytes=7375 send=0.000 s wait=2.745 s     <- one 28 MB reply
    store:   reserve_write is entered 2.57 s after the previous call returns

So the client hands the request over instantly and then waits while the server assembles and sends
one large message; the same happens in the store direction before storage is entered.

Raw transport, for reference (plain TCP, Windows client to a socket inside WSL): 143 MiB/s send,
78 MiB/s receive. So the bridge is 4-8x faster than what the cache achieves, and it is itself a
ceiling: the engine runs on Windows, the server inside WSL, and that relay is the path.

Throughput against payload size (engine-side `[lmcache] store`):

| payload | rate |
|---|---|
| 28.1 MiB (1916 tok) | 12-20 MiB/s |
| 69.4 MiB (4723 tok) | 32 MiB/s |

Fitting those two points gives roughly **1.1 s fixed + 65 MiB/s marginal**: the fixed part dominates
short prefixes, the marginal part roughly matches the bridge. Worker count does not move it
(4 -> 12, 8 -> 20, 16 -> 11 MiB/s), so it is not a server concurrency knob.

And the shape that would benefit most is never stored at all: a 163,689-token request runs its
prompt-frontier capture (the done line shows `boundary 3.4 s`) but **no `[lmcache] store` line and no
growth in the server's RSS appear** - `store()` is never called, because the engine publishes only a
checkpoint that `finish()` hands back and this request does not hand one back. At 4.7k tokens one of
four concurrent requests did publish, so the gate is the capture policy, not the size of the payload.

Ordered next steps, by what they unlock:

1. make a 163k turn produce a finished checkpoint, so the cache sees the shape it is for;
2. send a turn's chunks across the pool's connections in parallel (attack the ~1.1 s fixed part; the
   restore direction cannot be split - `prepare_retrieve` returns the whole run in one reply);
3. accept ~78 MiB/s as the ceiling of this topology: 2.4 GiB restores in ~31 s at best, against a
   111-114 s cold prefill. Beating that needs the client inside WSL or a shared-memory transfer mode.

## 12. Correction: section 11's cost model was wrong, and the real term is the link

Section 11 fitted "1.1 s fixed + 65 MiB/s marginal" from two store timings. That model is wrong, and
the reason is a measurement bug worth remembering: the store/restore payload is **KV chunks plus the
whole state image**, and the image is `153,954,304 B = 147 MiB` regardless of prefix length. The rate
line counted only `checkpoint.kv_bytes`, so a 28.1 MiB store (1916 tokens) actually put ~175 MiB on
the wire while the log said 28.1 MiB.

Sampling the server's socket counter (`ss -tin` on :6555, every 50 ms) during that store shows the
bytes arriving steadily:

    1795265221.73 ... delta=3245580
    1795265221.85 ... delta=5569900
    ...
    1795265222.99 ... delta=9038860
    final total=183476609        (~175 MiB, ~150 MB/s sustained)

So the transfer is a plain byte stream at the link's rate. Direction matters:

| direction | rate | evidence |
|---|---|---|
| engine -> server (store) | **~150 MB/s** | socket counter above: ~9 MB per 60 ms |
| server -> engine (restore) | **~64 MB/s** | `fetch 27.5 MB KV + 155.9 MB state` in 2.876 s |

The storage calls inside the server remain milliseconds (`reserve_write` 0.14 ms, `finish_write`
3-8 ms, restore-path calls 0.04-16 ms), so the server's own work is not the limit - the link is, and
the restore direction is the slow half, which is also the half TTFT waits on.

Consequence for the 1-2 s target: a 163k restore moves 2.4 GiB of KV **plus** 147 MiB of state
(2.55 GiB). At 64 MB/s that is ~40 s; at 1.3 GB/s it is 2 s. No amount of client-side parallelism
changes that, because the limit is bytes over a VM boundary - which is why the options are, in order:

1. put the cache on the same side of the boundary as the engine (a Windows-native L3 service, or a
   Windows build of the server), so the transfer becomes a memcpy at multi-GB/s;
2. shrink the payload: the state image is a constant 147 MiB per turn and compresses well (BF16/FP32
   recurrent state), and a prefix whose state did not change needs no re-send at all;
3. keep the current topology and accept ~40 s at 163k, which still beats the 111-114 s cold prefill.

## 13. The 127.0.0.1 path in mirrored mode is the slow one (measured 2026-11-21)

The comparison that made this visible: a 2.5 GbE link to a NAS carries ~300 MB/s, while this machine's
WSL loopback was delivering 150 MB/s one way and 64 MB/s the other. That is not a bandwidth ceiling,
it is a chosen path.

Evidence - the same 28 MB TCP transfer, one Python client, three destinations:

| destination | send | receive |
|---|---|---|
| `127.0.0.1` | 112.7 MB/s | 68.2 MB/s |
| `192.168.50.54` (mirrored NIC) | 146.1 MB/s | 195.9 MB/s |
| `10.222.222.1` (WSL virtual NIC) | **331.1 MB/s** | 150.9 MB/s |

The route table says why: `127.0.0.1 via 169.254.73.249 dev loopback0`. With
`networkingMode=Mirrored`, Windows reaching 127.0.0.1 lands on a user-space loopback proxy instead of
a kernel path, and that proxy is the narrow end.

End to end through the real cache (a 1916-token checkpoint = 175 MiB on the wire: 28.1 MiB KV plus the
146.8 MiB state image):

| engine endpoint | store | restore (183.5 MB) |
|---|---|---|
| `tcp://127.0.0.1:6555` | ~97 MB/s | 64 MB/s (2.876 s) |
| `tcp://10.222.222.1:6555` | 132 MB/s | **129 MB/s (1.424 s)** |
| `tcp://192.168.50.54:6555` | - | **130 MB/s (1.410 s)** |

So the address alone buys ~2x on the restore TTFT waits on, and `tools/console/l3_lmcache.sh` now binds
`--host 0.0.0.0` for that reason (`L3_HOST` overrides).

What this does not fix: at 130 MB/s a 163k restore (2.55 GiB including the image) is still ~20 s, so the
1-2 s target still needs the cache on the engine's side of the boundary. What it does mean is that the
remaining gap is no longer a misconfiguration - it is the relay plus the Python server, and the raw
numbers above show the path itself can do 331 MB/s in the store direction.

Next measurements worth taking, in order: the same three destinations with a C++ client (these are
Python numbers, so they are a floor), then the same cache with the state image compressed, since it is
146.8 MiB of every turn regardless of prefix length.

## 14. The single-connection ceiling is the relay, not the socket window (measured 2026-11-21)

Four measurements on `10.222.222.1` (the WSL virtual NIC), same Python client and server, 16 MB each
way per connection:

| connections | aggregate |
|---|---|
| 1 | 81.9 MB/s |
| 4 | 287.1 MB/s |
| 8 | **358.5 MB/s** |

and, on one connection, socket buffer size makes no difference at all:

| SO_SNDBUF/SO_RCVBUF | rate |
|---|---|
| 1 MB | 98.1 MB/s |
| 4 MB | 89.5 MB/s |
| 16 MB | 80.8 MB/s |
| 64 MB | 88.5 MB/s |

So this is not a TCP window problem (`tcp_rmem` 4 KB-32 MB, `wmem` 4 MB, window scaling on, BBR) and
not memory bandwidth. It is one connection being served serially at ~90 MB/s somewhere in the relay,
with parallelism absorbing it: 4 connections reach 287 MB/s, 8 reach 358 MB/s.

Where that leaves each direction of a checkpoint:

* **store can be split** - `kCommitStore` carries `groups[].chunk_indices`, so disjoint chunk ranges can
  go over separate pooled connections, and each connection is its own ZMQ identity and therefore its own
  server worker thread. Measured headroom: 4-8x.
* **restore cannot** - `kPrepareRetrieve` takes only `(cache_key, instance)` and answers with the whole
  run in one reply. A restore stays at whatever one connection does (~90-130 MB/s), which is the
  direction TTFT waits on.

That is the real boundary of this topology, and it is why the two structural options remain: change the
WSL networking mode so the path stops being a user-space relay (needs `wsl --shutdown`, so it must be a
deliberate step), or put the cache on the engine's side of the boundary.

## 15. Optimization plan for the LMCache link, and the first experiment's result

Targets implied by the measured link (8 connections reach 960 MiB/s on the WSL virtual NIC,
190-254 MiB/s on one):

| scenario | payload | floor at 1 GB/s | 500 ms reachable? |
|---|---|---|---|
| short prefix (1916 tok) | 175 MiB | 0.18 s | yes |
| growth turn (+31 tok, new chunks only) | ~2 MiB + state | ms | yes |
| 163k full restore | 2.48 GiB (2.34 GiB KV + 147 MiB state) | 2.3-2.6 s | no - needs 5 GB/s |

Ordered plan, and where each one stands:

1. **Publish only new chunks** - not started. The key is chunk-addressed (one immutable object per 128
   tokens), so a growth turn only needs the new chunk(s). The client now takes `first_chunk` and
   `with_state` on `store()` for exactly this, with defaults that keep today's behaviour. What is
   still missing is the engine side: remembering, per continuing conversation, the frontier that was
   already published.
2. **Capture only new pages** - not started; matters for the first capture (2.4 GiB at 705 MB/s =
   3.4 s, of which only 64 MiB was pinned).
3. **Store across several connections** - tried and does not work as a client-side fan-out (below).
4. **Many frames instead of one big msgpack bin** - not started.
5. **Parallel restore** - not started; needs a chunk range on `kPrepareRetrieve`, which the protocol
   does not have. This is the only way a 163k restore gets near 1-2 s.
6. **State image shrink** - not started; 146.8 MiB of every turn regardless of prefix length.
7. **Endpoint / ZMQ tuning** - endpoint done (bind 0.0.0.0, connect to the WSL address, 5-6x versus
   127.0.0.1); ZMQ socket buffers/HWM not yet tuned.

### Experiment: sharding one store across the pool (NINFER_LMCACHE_SPLIT_STORE)

Implemented as four contiguous chunk ranges with absolute indices, the last range carrying the state
image. Result on a 1916-token publish:

    [lmcache] publish failed | 14 chunk(s) | 27525120 byte(s) |
      sharded store failed on lane 0: receive failed: Resource temporarily unavailable
    [lmcache] store 174.9 MiB in 5.108 s (34 MiB/s) | kv 28.1 + state 146.8 MiB | ok=0

Lane 0 never received its `prepare_store` reply, so the server serializes a key's store across
identities. Client-side fan-out cannot work; the server needs an answer for concurrent partial stores
of one key. The code is reverted; the `first_chunk` / `with_state` parameters stay because item 1 needs
them and they default to today's behaviour.

Consequence for the plan: item 3 is blocked on a protocol change, and item 1 (client side) is ready
but useless until the engine tracks the published frontier.

## 16. Verifying option C (`--enable-segmented-prefix`): it does not apply to this path

C was the hope that a prefix could be stored as independent segments, making both directions
parallel without touching the protocol. Two facts close it.

1. **Chunk identity is chained to the prefix.** `token_hasher.py`:

       def blake3_packed_hash(chunk, prefix_hash):
           h.update(prefix_hash)
           h.update(chunk)

   so chunk i's hash depends on every chunk before it. A key that starts at 32k tokens hashes from a
   different seed and can never match a lookup for the full 163k prefix. Storing a prefix as N
   independent keys is therefore not equivalent to storing it as one.

2. **The flag is not on our path.** `enable_segmented_prefix` lives in
   `lmcache/v1/multiprocess/modules/blend/` and is consumed only there, where its documented meaning
   is "retain the gapped prefix on a mid-prefix L2 retrieve failure" - not hash decoupling. Our server
   composes five modules (`engine_driven_transfer, lookup, p2p_controller, kv_events, management`) and
   does not load blend, so the flag is inert for the client we use. The content-addressed variant
   (`--enable-dedup-content`, `BlendTokenRangeMatcher(dedup_content=...)`) is blend-only for the same
   reason.

Consequence: parallelism cannot come from key granularity. It has to come from the transfer protocol:

* **restore**: a chunk range on `kPrepareRetrieve` (server + client), 8 connections x ~190 MiB/s
  = ~1.5 GB/s, which puts a 163k restore at ~1.7 s;
* **store**: concurrent partial stores for one key (currently serialized by the server, measured in
  section 15).

Both are server changes. Nothing client-side can work around them.

## 17. Parallel restore: ranges, the silent drop, and where the cache actually pays

Section 16 predicted the next step would be a chunk range on `kPrepareRetrieve`. That landed, and
two measurements changed the plan around it.

### 17.1 The payload never reached the cache above 256 MiB

`publish_to_cache` handed the writer thread its token list by move. When `enqueue()` refused the job
(the window is 256 MiB), the fallback called `backend->store(kept, ...)` on the vector the move had
already emptied, and `store()` returns immediately on an empty list. The effect was a silent split
by payload size:

| prefix | payload | result before the fix |
|---|---|---|
| 1736 tok | 176 MB | stored |
| 16k tok | 401,680,384 B | dropped, no log, no server record |

The probe that found it printed `store entered: tokens=0` while `sink` had just reported 252 pages
exported. Fix: the job captures a copy of the token ids, and a refused enqueue now prints. Measured
after: `store 382.1 MiB in 3.137 s (122 MiB/s) | ok=1` and `Stored 16000 tokens in 2.912 seconds`
server-side.

### 17.2 The state image does not compress

Exported 153,954,304 B, then measured on those exact bytes: zstd-1/3/9/19 and zlib-6 all 1.07x, best
de-interleave stride 4 gives 1.16x, entropy 7.53 bits/byte, 0.4% zeros. Layout split of the image:
recurrent 150,994,944 (98.1%), conv 2,949,120, hidden 10,240, dflash k/v 0 with spec off. It is real
high-entropy model state, so "compress the state image" is closed.

### 17.3 Ranges on prepare_retrieve

Server patch (local to `lmcache-src`, commit `b52bdfe`): `prepare_retrieve` takes
`chunk_begin/chunk_end/state_begin/state_end`, all defaulting to -1 (whole payload). The handler
slices the KV group and shifts `group_first_chunk`; the inline strategy slices the state tensor.
KV and state are both `uint8`, so byte slicing is exact.

Client (`ninfer` commit `3cf706b7`): `NINFER_RESTORE_SPLIT=N` runs N contiguous, exhaustive parts,
each with its own `request_id`. That detail is required, not cosmetic: the server's prefetch job is
consumed once per request_id, and with a shared one only the first part finds the prefetched objects
(the other three each timed out at 5.008 s).

Measured, cross-process, same 1736-token key:

| parts | restore |
|---|---|
| 1 | 1.42 s |
| 4 | 0.54-0.80 s |
| 8 | 0.46-0.66 s |

Sharing one prefetch across parts was tried and does not work: with `NINFER_RESTORE_SHARE` all eight
parts missed (`cache_n 0`), because the prefetched result is released by the first reader. Every part
therefore re-prefetches the whole entry, which is why 8 parts is not 4x faster than 1.

### 17.4 The cache does not pay on short prefixes

End to end, 1736-token prefix, engine restarted each round:

| path | total |
|---|---|
| cold prefill | 0.8 s (engine-wall 816 ms, prefill 2.4k tok/s) |
| cache hit, 1 connection | 2.03 s |
| cache hit, 4-8 connections | 1.09-1.12 s |

So at this size the cache is still slower than not using it. The 16k prefix is the first size
measured where it wins clearly:

| path | total |
|---|---|
| cold prefill | 8.6 s |
| cache hit, 8 parts | 2.8 s (restore 1.535 s, queue 2.0 s, engine-wall 145 ms) |

Break-even sits where prefill time exceeds restore time, at roughly 3k tok/s prefill and ~250 MiB/s
restore: a few thousand tokens for this model. Below it the cost model should decline the cache;
today it does not, which is the next piece of work.

### 17.5 Still open

* `enqueue` refuses anything over 256 MiB, so every large store runs synchronously on the finish
  path (3.1 s for 16k). The window needs to follow the payload or be configurable.
* Parts re-prefetch the whole entry; a server-side shared read (or a reference count that outlives
  the first reader) would make 8 parts actually scale.
* Restore still ships one message per part; multi-frame streaming was never measured.
* The cost model does not yet refuse the cache on prefixes where prefill is cheaper.
