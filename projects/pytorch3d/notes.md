# pytorch3d

3D deep-learning library (facebookresearch/pytorch3d). ext_type: torch-extension (Strategy B -- torch `BuildExtension` auto-hipify).

## gfx90a: COMPLETED upstream (exception to the jeffdaily-fork model)

The gfx90a ROCm port landed UPSTREAM as facebookresearch/pytorch3d PR #2039 "[ROCm] Port pytorch3d" (author jeffdaily), commit `b73d735ecf194c31de812feffef3a55cc3726128` (2026-06-01). Verified by the PR on AMD Instinct MI250X (gfx90a, warpSize=64), HIP 7.2, PyTorch 2.13, including the pulsar subrenderer and the test suite.

This is an exception to the usual MOAT model: the canonical port is upstream, not a jeffdaily moat-port fork. So `fork_url` points at upstream and `validated_sha` is the landed commit. MOAT adopts it as the gfx90a baseline. Optional: re-confirm on this MOAT gfx90a host with a validator run; not required since the PR already verified on MI250X.

### What the port did (from PR #2039)
- `setup.py`: detect ROCm via `torch.version.hip is not None`; treat `ROCM_HOME` as the GPU-toolkit-root analogue of `CUDA_HOME` (without it, `CUDA_HOME is None` silently demoted the build to a CPU-only `CppExtension`); skip `CUB_HOME`, CUDA-13 visibility flags, and `-ccbin=` on ROCm.
- torch `BuildExtension` auto-hipifies the `.cu` (`cuda_runtime.h`->`hip/hip_runtime.h`, `cub::`->`hipcub::`, `cudaStream_t`->`hipStream_t`), so most of the lift is build-system glue.
- `pytorch3d/csrc/pulsar/gpu/commands.h`: the CUDA `_rn`-suffixed FP rounding intrinsics (`__fadd_rn`/`__fdiv_rn`/`__fsqrt_rn`/`__fmaf_rn`/`__frcp_rn`) and `__saturatef` have no HIP equivalent (AMD's ISA has no instruction-level rounding-mode override), so on the `USE_ROCM` arm they expand to plain operators / `sqrtf` / `fmaf` / `1.0f/x` / `fmaxf(0,fminf(1,x))` -- rounding-mode-equivalent (both round-to-nearest-even). If FMA-fusion drift ever surfaces, add `-ffp-contract=off` to the pulsar TU. This is exactly our UPSTREAM_FINDINGS B1 (`__fsqrt_rn` 1-ULP) / B4 (`-ffp-contract`) story, now resolved in-tree upstream.

## gfx1100 + gfx1151: port-ready -- IMMEDIATE verification target

Followers reuse the upstream landed commit (validate-first; delta-port only on failure). This MOAT gfx90a host cannot validate gfx1100/gfx1151 -- a gfx1100 host's `orient.sh` selector picks up the `port-ready` follower.

### gfx1100 verification task
1. Clone upstream facebookresearch/pytorch3d at `b73d735ecf194c31de812feffef3a55cc3726128`.
2. Build the `_C` extension against a ROCm-built PyTorch for gfx1100 (RDNA3, wave32).
3. Run the pytorch3d test suite (`tests/`), including the pulsar renderer tests, on real gfx1100.

The decisive risk: the landed port was verified only on gfx90a (warpSize=64); gfx1100 is **wave32** (warpSize=32). Audit for wave64-baked assumptions the gfx90a verification could not catch:
- explicit-mask `__shfl*`/`__ballot` intrinsics (32- vs 64-bit masks) and hardcoded warpSize / `%32` / `/32` lane math, especially in the pulsar renderer's warp reductions.
- `cub`/`hipcub` `WarpReduce`/`WarpScan` defaulting to the 64-wide logical warp on hipCUB (pin `<...,32>` if any survive).
- the `_rn`->plain-op change is wave-agnostic (fine on wave32).
- the rocPRIM 4.2.0 GFX10/11 DPP bug: any `cub::DeviceSegmentedRadixSort` / `warp_exchange` path can fail to COMPILE on gfx1100 ("wavefront shifts are not supported on GFX10+") -- the raft lesson; may need a target exclusion or an upstream rocPRIM fix.

Outcome handling:
- Passes as-is -> set gfx1100 `completed`, validated_sha = `b73d735ecf...` (same commit; pure verification, no delta).
- wave32 breaks -> this would be the first time pytorch3d needs a jeffdaily fork: delta-port the wave32 fixes onto `jeffdaily/pytorch3d` @ `moat-port` (branched off the landed commit), validate on gfx1100, then the wave32 fix is a candidate to upstream as a follow-up PR (user-gated, like any upstream PR).

## Validation 2026-06-01 (gfx1100)

**Host**: linux-gfx1100, AMD Radeon Pro W7800 48GB x2, gfx1100 (RDNA3, wave32), ROCm 7.2.1, PyTorch 2.13.0a0+gitb5e90ff (HIP 7.2.53211), Python 3.12.

**Commit validated**: `b73d735ecf194c31de812feffef3a55cc3726128` (upstream facebookresearch/pytorch3d, same as gfx90a baseline).

### Build fix applied (follower delta)

`setup.py` hardcodes `-std=c++17` in `extra_compile_args`. PyTorch 2.13 headers use C++20 features (`requires` concept constraint in `TensorImpl.h:2516`), causing compile failures with gcc/clang under c++17. Changed both occurrences of `-std=c++17` to `-std=c++20`. This is a build-system compatibility fix required for the PyTorch 2.13 + RDNA3 environment; it does not affect GPU kernel semantics and is wave-size-agnostic.

**Build command**:
```
ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 FORCE_CUDA=1 HIP_VISIBLE_DEVICES=0 \
  pip install -e /var/lib/jenkins/moat/projects/pytorch3d/src --no-build-isolation -v
```
Build time: 81 seconds (timeit.sh stats.jsonl). torch `BuildExtension` auto-hipified 30 TUs.

**gfx1100 code-object evidence**: `roc-obj-ls _C.cpython-312-x86_64-linux-gnu.so` shows 30 `hipv4-amdgcn-amd-amdhsa--gfx1100` bundles, zero other GPU arches. No gfx90a objects present.

### Wave32 audit findings

**pulsar warp reductions (commands.h / renderer.render.device.h)**:
- `WARP_CUMSUM`/`WARP_MAX`/`WARP_SUM` (explicit-mask `__shfl_down_sync` with 16/8/4/2/1 offsets) are guarded `#if !defined(USE_ROCM)` -- not compiled on ROCm. Not a risk.
- `__ballot(done)` stored as `unsigned long long` in `renderer.render.device.h:290`. HIP defines `__ballot` to always return `unsigned long long` via `__builtin_amdgcn_ballot_w64`; on wave32 only bits 0-31 are set, bits 32-63 are zero. `__popcll` counts correctly. Type is correct for the HIP API; no wave32 bug.
- `BALLOT`/`SHFL_SYNC` macros are defined but only the `#if !defined(USE_ROCM)` template functions use the shfl-with-offset math. Safe.

**warp_reduce.cuh (WarpReduceMin/WarpReduceMax)**:
- Uses shared-memory tree reduction: `min_dists[tid + 32]` etc. are array index accesses, not warp shuffle offsets. Block size is 128, shared mem has 128 slots; `tid` ranges 0..31 in the warp-unroll tail, `tid+32` indexes 32..63. This is wave-size independent (shared memory, not lane-relative). `__syncwarp()` calls are gated `#if !defined(USE_ROCM)` -- correct per AMD lockstep execution. No wave32 bug.

**hipCUB WarpReduce/WarpScan**: Not used. pytorch3d uses only device-wide CUB/hipCUB APIs (`DeviceRadixSort`, `DeviceReduce`, `DeviceSelect`) which map to hipcub device-level algorithms. No warp-width pin needed.

**rocPRIM 4.2.0 GFX10/11 DPP**: `cub::DeviceRadixSort::SortPairsDescending` is used in pulsar. This maps to hipcub device radix sort (not segmented). Built cleanly on gfx1100 -- no "wavefront shifts are not supported on GFX10+" compile error observed. Not affected.

**HSA faults**: None. Zero 0x1016 errors across all test runs.

### Test results

**Pulsar renderer (highest wave32 risk)**:
- 9/9 passed (determinism confirmed: identical results on second run)
- 1 test (test_principal_point) failed: `imageio.imsave` rejects 1-channel array -- imageio API change, not a GPU issue. The GPU render and `np.allclose` correctness check both passed before the imsave call. This failure is platform- and wave-size-agnostic.

**KNN** (device radix sort path): 7/7 passed (determinism confirmed: identical results on second run)

**Ball query**: 5/5 passed

**Point mesh distance** (WarpReduceMin path): 9/9 passed

**Rasterize meshes**: 17/17 passed

**Sample farthest points** (WarpReduceMax path): 4/4 passed

**Chamfer**: 18/18 passed

**Broader GPU suite** (18 test files): 148/148 passed

**Non-GPU suite** (7 test files): 160/160 passed

Test timing: ~94 seconds for the wave32-risk GPU suite (pulsar + knn + ball_query + point_mesh + rasterize + sample_farthest + chamfer).

### Wave32 verdict

**PASS.** No wave32 failures. The pulsar warp reductions are correctly guarded behind `#if !defined(USE_ROCM)` and the HIP `__ballot` type is compatible. WarpReduceMin/Max uses shared-memory indexing (wave-size agnostic). No hipCUB warp primitives used. No rocPRIM DPP compile failure. All GPU results correct and deterministic on gfx1100 wave32. The c++20 build fix is the only delta needed for this environment.

## Adoption note
priority 7.075 is the computed discovery score (stars 9891 -> 3.995, forks 1455 -> 1.582, pushed 2026-06-01 -> recency 1.498; weights stars 1.0 / forks 0.5 / recency 1.5 / half-life 180d). That places pytorch3d near the top of the table (above cuml 6.61 and cugraph 6.11; below only vllm 8.53 in the discovery pool), versus the lowest real table score op43dgs 3.393.

## windows-gfx1151 progress 2026-06-01 (validator, in-progress -- NOT yet a fork)

Build env on this host WORKS: venv-gsplat torch 2.12.0+rocm7.14.0a (hip 7.14), torch.cuda.is_available()=True on AMD Radeon 8060S (gfx1151). Cloned upstream @ b73d735 into projects/pytorch3d/src (shallow). Build script: agent_space/pytorch3d/win_build.sh (sources gsplat_buildenv.sh; PYTORCH_ROCM_ARCH=gfx1151; `python setup.py build_ext --inplace`).

First Windows blocker (the gsplat fault, NOT wave32): the pulsar host C++ TUs `pytorch3d/csrc/pulsar/pytorch/{util,tensor_util,renderer}.cpp` (and their torch-generated `*_hip.cpp` variants) include c10/cuda (-> c10/hip) headers and are compiled by MSVC cl.exe, which cannot parse the HIP types (`c10/hip/HIPFunctions.h: error C2146: syntax error before 'WarningState'/'memcpy_and_sync'/'stream_synchronize'` -- amd_hip_vector_types.h GCC attributes). FIX (gsplat lesson [[gsplat-windows-port]]): route these host TUs through hipcc by giving them a `.cu` extension (or otherwise force the HIP compiler). This is a Windows-only build delta, so pytorch3d will need its FIRST jeffdaily fork (jeffdaily/pytorch3d @ moat-port, branched off the upstream landed b73d735) per the gfx1100 plan's "wave32 breaks -> fork" note -- here triggered by the host-compiler issue rather than wave32.

After that delta, continue: rebuild, then audit wave32 (pulsar warp reductions / shfl masks / hipCUB WarpReduce 64-wide default / rocPRIM GFX10+ DPP compile bug per the plan), then run tests/ incl. pulsar on real gfx1151. Deploy TheRock runtime beside any test exe if needed (torch loads its own amdhip64). Next session resumes here.
