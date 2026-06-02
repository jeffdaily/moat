# evogp notes

## Port summary (linux-gfx90a)
Strategy B (torch CUDAExtension + build-time hipify). Single `evogp.evogp_cuda`
op library over four .cu files. Fork: https://github.com/jeffdaily/evogp,
branch `moat-port`, HEAD ba4fa7e. Validated on gfx90a (MI250X, wave64), ROCm
7.2, torch 2.13a (hip 7.2.53211).

## Build (gfx90a)
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a
cd src
rm -rf build src/evogp.egg-info src/evogp/*.so src/evogp/hip   # clean stale hipify mirror after any source edit
pip install -e . --no-build-isolation
```
Requires a ROCm PyTorch active (confirm `torch.version.hip` is non-None).

## Validation (real GPU gate)
```
export HIP_VISIBLE_DEVICES=0
python -m evogp.sr_test     # 1000-pop SR, 100 gens, manual_seed(0)
```
PASS: completes with no HIP fault; best fitness (negated MSE) converges
monotonically from -0.25 (gen 0) to -0.018 (gen 99), finite throughout. Two
fixed-seed runs gave the identical final best (-0.0179). AMD_LOG_LEVEL=3 shows
native `amdgcn-amd-amdhsa--gfx90a` code objects dispatching.

## Changes made (4 fault classes hit; 2 build-only, 1 correctness)
1. setup.py: gate the device-compiler ("nvcc" key) flag list on
   `torch.version.hip`. The nvcc/ptxas-only flags (--ptxas-options, -Xptxas,
   -lineinfo, -maxrregcount, -lcudart, --expt-relaxed-constexpr, -use_fast_math)
   reach hipcc verbatim (hipify rewrites source, not compile args) and fail/are
   meaningless. Drop them on ROCm, keep -O3.
2. kernel.h: `#include "device_launch_parameters.h"` is a CUDA-toolkit-only
   header. It is quoted, so hipify treats it as a local header and does NOT
   strip/translate it -> `fatal error: file not found` on hipcc. Guarded behind
   `#ifndef USE_ROCM`; hip_runtime.h (from hipified `cuda_runtime.h`) already
   supplies blockIdx/threadIdx/etc.
3. Correctness (the real bug): per-thread `alloca` scratch corruption on
   gfx90a. The tree-interpreter kernels (treeGPEvalKernel,
   treeGPRegressionFitnessKernel x3 variants, treeGPGenerate, _gpTreeReplace)
   allocated ~8 KB per-thread working stacks via `alloca` (dynamic device
   scratch). On gfx90a these dynamic allocations alias across concurrently
   resident wavefronts: results were CORRECT up to ~128 threads in flight but
   wrong at/above ~200 (tree_evaluate returned garbage; tree_SR_fitness tripped
   the interpreter's `assert(top == 1)` and aborted with
   HSA_STATUS_ERROR_EXCEPTION). Fix: replace every `alloca(MAX_STACK * ...)`
   with a fixed-size local array (`float stack[MAX_STACK]` etc.), same size as
   before. clang gives this a stable per-lane scratch frame. Arch-unified: no
   wave-width dependence, identical on CUDA / wave32 / wave64.

## Gotchas / for followers (gfx1100, gfx1151)
- The alloca->local-array fix is the load-bearing change. If a follower
  regresses, it is NOT wave width (there are zero warp intrinsics, zero
  hardcoded 32, and the fitness reduction is fully `__syncthreads()`-guarded
  with the warp-synchronous tail commented out). Suspect scratch sizing instead.
- The `top == 1` device assert in _treeGPEvalByStack* is the canary: if it
  fires, the tree data feeding eval is corrupt (almost certainly scratch
  aliasing again), not a logic bug -- the generate kernel produces structurally
  valid trees (verified: 0 malformed over 1000).
- Diagnostic trick that localized this: eval the SAME forest at increasing pop
  sizes and compare tree[0] against a CPU reference. Correct below the
  occupancy threshold, wrong above it -> scratch corruption, not a data bug.
- Stale hipify mirror: `src/evogp/hip/` is regenerated each build. It is now in
  .gitignore. After ANY source edit, `rm -rf build src/evogp/hip` before
  rebuilding or you compile stale .hip files.
- `thrust::random` (taus88) works as a drop-in under rocThrust; no source edit
  needed. No cub/curand/cuBLAS/textures/CUTLASS anywhere.

## Pre-existing upstream test bug (NOT a port issue)
`test/test_bind_success.py` passes a 24-element `roulette_funcs` tensor, but the
binding (torch_wrapper.cu) correctly requires `Function::END` = 29 entries
(the Function enum grew). The op rejects it with "roulette_funcs must have
shape [29], but got shape [24]" -- identical on CUDA. Did not patch upstream's
test; the authoritative GPU gate is sr_test, which passes. If you want the
per-op probe to run, the test's roulette tensor needs padding to 29.

## Actions disabled on fork
`gh api -X PUT repos/jeffdaily/evogp/actions/permissions -F enabled=false` run
after fork.
