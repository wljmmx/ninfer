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
