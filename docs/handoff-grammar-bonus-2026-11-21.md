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
