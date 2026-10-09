# Keeping the sm_89 port in sync with upstream

Upstream targets **sm_120a (RTX 5090)**. This branch adds an **sm_89 (RTX 4090) + Windows**
port, so some upstream code cannot compile or run here as-is. This note records the two
failure modes seen while merging upstream `68c54356..81c8ce09` (28 commits) and the
workflow that catches them cheaply.

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

The checker implements exactly these rules: it flags `launch_*tma*<` calls outside
`#if defined(NINFER_ENABLE_TMA)` **or** `#if defined(NINFER_ENABLE_NVFP4)` (NVFP4 code is
Blackwell-only, so its guard also gates TMA), and it exempts templated lambda bodies.

## Writing the sm_89 fallback

`Bf16A16TmaMmaSchedule<R,T,K,WR,WT,S,M,…>` **inherits** `Bf16A16MmaSchedule` with the same
tile geometry (`bf16_schedule.cuh`); only the producer/staging and swizzle differ. So a
like-for-like Ada fallback is mechanical: instantiate the cp.async schedule with the same
`R,T,K,WR,WT,S,M` and the cp.async swizzle. See the guarded ladders in
`n14336_k5120.cu`, `n256_k5120.cu`, `n5120_k6144.cu` for the shape of the edit.

## Line endings and encoding

Upstream files carry **no UTF-8 BOM**. A BOM added by tooling here produced the only merge
conflict against upstream/master. Keep the sources BOM-free.
