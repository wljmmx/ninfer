# Keeping the sm_89 port in sync with upstream

Upstream targets **sm_120a (RTX 5090)**. This branch adds an **sm_89 (RTX 4090) + Windows**
port, so some upstream code cannot compile or run here as-is. This note records the two
failure modes seen while merging upstream `68c54356..81c8ce09` (28 commits) and the
workflow that catches them cheaply. It also records the current absorption boundary
(`81c8ce09..a754fcf3`, 305 commits) and where the per-shape fallback records live.

## The two failure modes

1. **Blackwell-only call sites without a gate.** Upstream added sixteen bf16 linear shape
   selectors that call `launch_bf16_tma_mma`, which is declared only under
   `#if defined(NINFER_ENABLE_TMA)`. On sm_89 that is a hard compile error
   (`identifier "launch_bf16_tma_mma" is undefined`), discovered minutes into the build.

2. **A build that "succeeded" while objects were stale.** Ninja recompiles only what its
   dependency graph reports as stale. When a build aborts part-way (compile error,
   interrupted merge, cancelled run) the dependency files for the affected targets can be
   left unwritten. Afterwards whole translation units may still be compiled against
   pre-merge headers, so a green build proves nothing. In this merge it hid a real
   signature mismatch (`Q4MmaDecodeAtom::decode_eight` lost its `scale` argument) that had
   been latent in the port since the GEMM tuning commit.

## Workflow

```powershell
$repo = '<path>\ninfer-4090-port\ninfer'      # note: the directory is ninfer, not infer

# 1. Fetch and size up the merge (read-only).
git -C $repo fetch upstream
git -C $repo diff --stat (git -C $repo merge-base HEAD upstream/master)..upstream/master
git -C $repo merge-tree --write-tree --name-only HEAD upstream/master   # exit 0 = no conflicts

# 2. Fail fast on Blackwell-only call sites, before spending minutes on a build.
python $repo\tools\port\check_blackwell_guards.py

# 3. Merge.
git -C $repo merge --no-ff upstream/master

# 4. Do NOT trust the incremental build. Report (then, if needed, delete) objects that
#    predate the point you want the build to reflect, and rebuild.
pwsh $repo\tools\port\reset_stale_objects.ps1 -BuildDir $repo\build-win -Reference <merge-sha>
pwsh $repo\tools\port\reset_stale_objects.ps1 -BuildDir $repo\build-win -Reference <merge-sha> -Delete
& <build script> $repo\build-win

# 5. Judge the test surface from the rebuilt tree.
ctest --test-dir $repo\build-win --timeout 180 --output-on-failure
```

Notes on step 4: the cutoff is a commit time, so if the reference is newer than the build
*everything* looks stale and a full rebuild is the honest answer — which is what a merge
usually warrants anyway. The script therefore **reports by default** and only deletes with
`-Delete`, for the targeted case of a build that aborted at a known point.

## What must be gated, and what must not

| Construct | On sm_89 | Rule |
|---|---|---|
| `launch_bf16_tma_mma` (and its callers) | undefined | wrap in `#if defined(NINFER_ENABLE_TMA)` **with an Ada fallback** |
| `Bf16A16TmaMmaSchedule<...>` aliases | fine (type only) | no gate needed |
| NVFP4 entry points (`nvfp4_*_dispatch`, `nvfp4_*_launch`) | defined as throwing stubs | **do not gate the caller**; the guard lives inside the plan `.cpp` |
| a TMA call inside a *templated* lambda body | fine | the body is instantiated only where the (guarded) caller is |
| `tma_load_*` / `tma_store_*` / `cp_async_bulk*` / `cuTensorMapEncodeTiled` | undefined | wrap in `#if defined(NINFER_ENABLE_TMA)` (checker class `tma_bulk`) |
| `__cluster_dims__`, `cluster.sync()`, `cooperative_groups::this_cluster`, `cudaLaunchAttributeClusterDimension`, `cudaOccupancyMaxActiveClusters` | undefined | wrap in the TMA guard (checker class `cluster`); do not emulate a cluster launch |
| any TP2 / ADP2 symbol under `src/` | sealed dead group | no fallback: a reference is a violation (checker class `tp2`); only tests may mention it, under `--include-tests` |

The checker implements exactly these rules. Its current rule table covers `tma` (launcher
calls plus non-template TMA entry points), `tma_bulk`, `cluster` and `tp2`; definitions are
not calls, comments and string literals are skipped, templated lambda bodies are exempt, and
NVFP4 entry points are exempt from the `tma` rule (the skip count is printed).

## Writing the sm_89 fallback

`Bf16A16TmaMmaSchedule<R,T,K,WR,WT,S,M,…>` **inherits** `Bf16A16MmaSchedule` with the same
tile geometry (`bf16_schedule.cuh`); only the producer/staging and swizzle differ. So a
like-for-like Ada fallback is mechanical: instantiate the cp.async schedule with the same
`R,T,K,WR,WT,S,M` and the cp.async swizzle. See the guarded ladders in
`n14336_k5120.cu`, `n256_k5120.cu`, `n5120_k6144.cu` for the shape of the edit.

## Fallback records for the a754fcf3 absorption

Upstream now carries **no `NINFER_ENABLE_TMA` macro at all** (its targets select whole
translation units). Every adopted shape therefore needs its gate added during the merge; a
shape copied without one compiles on sm_120a and fails differently on sm_89.

[`sm89-fallback-table.md`](sm89-fallback-table.md) is the only per-shape record; this note
does not duplicate its rows. It has three tiers:

| Tier | Content | State |
|---|---|---|
| A | the 16 local shape files with an inline TMA ladder | wired and measured on sm_89 |
| B | 5 upstream-only bf16 ladders (`linear/add/bf16` ×1, `linear/split/bf16/rtx5090` ×3, `linear/split/bf16` shared ×1), 20 rows | T004 draft: mapping generated, **not compiled or measured** |
| C | 10 quantized shapes plus QSA / PLE / GDN / MoE execution | registration only: "to be measured at port time" |

Rules for Phase 5:

1. Tier B rows are hypotheses. Confirm or replace each row with a measured result before a
   shape is called ported; its token intervals express upstream intent only.
2. Tier C gets no mechanical mapping. The only verified rule is "same tile geometry,
   producer swizzle Tma128 → Xor64". fp8 is structurally similar but unmeasured, and NVFP4
   has no sm_89 tensor-core path (a fallback changes the arithmetic profile and must be
   re-qualified).
3. Spark / DGX shapes stay in the T001 dead group; do not draft fallbacks for them.
4. Regenerate Tier B with `tools/port/gen_bf16_fallback_table.py --draft` after an upstream
   resync. Measured deviations live in the `#else` branches and are not regenerated.

## Line endings and encoding

Upstream files carry **no UTF-8 BOM**. A BOM added by tooling here produced the only merge
conflict against upstream/master. Keep the sources BOM-free.
