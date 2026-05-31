# arrayfire notes

ArrayFire 3.10.0, pure-CMake multi-backend array library. Port shape: a NEW sibling
backend `src/backend/hip` cloned from `src/backend/cuda` (additive; the CUDA/NVIDIA path
stays byte-for-byte). The HIP backend reports `AF_BACKEND_CUDA` so the unified dispatcher and
the 139 gtest binaries treat it as the "cuda" backend (no public ABI/enum change). Lead
platform linux-gfx90a (MI250X, ROCm 7.2.1).

Fork: https://github.com/jeffdaily/arrayfire (branch `moat-port`; `master` stays a clean
upstream mirror). Actions disabled on the fork. base_sha 492718b.

## Install as a dependency
N/A -- arrayfire is a leaf (no other MOAT project depends on it). It vendors only
`extern/half` and finds host/GPU libraries externally; no inter-project MOAT deps.

## Phase 0 (GO/NO-GO): hipRTC runtime-JIT -- RESULT: GO

The make-or-break risk was arrayfire's bespoke NVRTC + CUDA-Driver-API runtime-JIT engine
(`src/backend/cuda/compile_module.cpp`). On NVIDIA it does
`nvrtcCompileProgram(--gpu-architecture=compute_XX)` -> `nvrtcGetPTX` ->
`cuLinkCreate`/`cuLinkAddData(CU_JIT_INPUT_PTX)`/`cuLinkComplete` -> `cuModuleLoadData`. The
question was whether the hipRTC analogue (direct code object, no PTX, no cuLink, `--offload-arch`)
works for arrayfire's TWO JIT paths.

Standalone repro `agent_space/af_hiprtc_poc/af_poc.cpp` (built with
`/opt/rocm/bin/hipcc -O2 -std=c++17 af_poc.cpp -o af_poc -lhiprtc`, run
`HIP_VISIBLE_DEVICES=0 ./af_poc`) reproduces both arrayfire JIT paths with arrayfire's actual
constructs and runs them end to end on gfx90a. RESULT: PASS.

- JIT path (`sourceIsJIT=true`): an `extern "C" __global__` element-wise kernel carrying the
  `kernel/jit.cuh` preamble -- `typedef float2 cuFloatComplex;` / `double2`, `#include <cuda_fp16.h>`,
  `__device__` complex helpers (`__caddf`/`__cmulf`...), and an integer-round intrinsic
  (`__double2ll_rn`). hipRTC compiles it to a 5848-byte code object; `hipModuleLoadData` +
  `hipModuleGetFunction("af_jit_kernel")` + `hipModuleLaunchKernel` + host verify all pass.
- Templated path (`sourceIsJIT=false`): a templated kernel instantiated by name expression
  via `hiprtcAddNameExpression` / `hiprtcGetLoweredName` (lowered name
  `_Z15af_templ_kernelIfLi3EEviPT_PKS0_S3_`), loaded and launched, host verify passes. (This
  matches the already-proven cudf hipRTC PoC, re-confirmed in arrayfire's shape.)

The flow `hiprtcCompileProgram(--offload-arch=gfx90a)` -> `hiprtcGetCodeSize`/`hiprtcGetCode`
-> `hipModuleLoadData` -> `hipModuleGetFunction` -> `hipModuleLaunchKernel` is fully functional.
The arch flag is derived from `hipGetDeviceProperties(...).gcnArchName` with the
`:sramecc+:xnack-` feature suffix stripped (`gfx90a`).

### Three concrete arrayfire-specific hipRTC deltas (all fixed in the repro; none a blocker)

1. **Empty-string ("") header source is REJECTED by `hiprtcCreateProgram` (INVALID_INPUT)**,
   where NVRTC accepts it. `compile_module.cpp` passes several `string("")` dummy header
   bodies ("DUMMY ENTRY TO SATISFY ..." for math.h/stdbool.h/stdlib.h/vector_types.h/utility).
   On HIP each dummy header source must be NON-EMPTY (a `"/* empty */\n"` placeholder works;
   `min4.cpp` mode 1 vs mode 2 isolates this exactly). A NULL header pointer aborts in the
   std::string ctor.
2. **`--device-as-default-execution-space` is REJECTED by hipRTC** ("unknown argument").
   arrayfire passes it unconditionally (`compile_module.cpp:275`). Drop it on HIP -- HIP makes
   device the default execution space for `__global__`/`__device__` via `__HIP_DEVICE_COMPILE__`
   anyway. Same for `--gpu-architecture=compute_XX` -> `--offload-arch=gfx90a`, and the
   `--device-debug`/`--generate-line-info` NVRTC debug flags (drop or map).
3. **An embedded header whose body does `#include <hip/...>` needs `-I/opt/rocm/include`** as a
   runtime compile option (hipRTC does not implicitly know the ROCm include path). This is the
   cudf lesson; the cleanest model is to inject `cuda_fp16.h` (the name arrayfire's JIT source
   `#include`s) as a body of `#include <hip/hip_fp16.h>` and pass `-I/opt/rocm/include`. NOTE:
   hipRTC also provides `__half`/`__float2half`/`__half2float` AND `float2`/`double2` as
   builtins with NO header at all (`min2.cpp` modes 2,3), so the fp16/vector-type headers may
   even be droppable; the `-I/opt/rocm/include` + injected-header model is the conservative
   choice that keeps the source strings unchanged.

Repro files: `agent_space/af_hiprtc_poc/{af_poc.cpp, RESULT.log}` (full repro) and the
isolation probes `min.cpp`/`min2.cpp`/`min3.cpp`/`min4.cpp`. ROCm 7.2.1, hiprtc version 9.0,
AMD clang (roc-7.2.1), 4x MI250X (GCDs 0-3), pinned to GCD 0.

## JIT engine restructure (compile_module.cpp on HIP)

Given Phase 0 GO, the port of `compile_module.cpp` is:
- `nvrtc.h` -> `hip/hiprtc.h`; `nvrtc*` -> `hiprtc*`; `NVRTC_SUCCESS` -> `HIPRTC_SUCCESS`.
- Drop the entire PTX + cuLink block (`nvrtcGetPTXSize`/`nvrtcGetPTX`, `cuLinkCreate`/
  `cuLinkAddData(CU_JIT_INPUT_PTX)`/`cuLinkComplete`, `cuLinkDestroy`). Replace with
  `hiprtcGetCodeSize`/`hiprtcGetCode` into a `vector<char>` and `hipModuleLoadData(&mod, code)`.
- Arch: replace `--gpu-architecture=compute_%d%d` with `--offload-arch=<gcnArchName-stripped>`.
- Drop `--device-as-default-execution-space` and the NVRTC `--device-debug`/`--generate-line-info`.
- Empty dummy header bodies -> non-empty placeholder; add `-I/opt/rocm/include` to the JIT
  compile options (or to the non-JIT options) so injected `hip/...` includes resolve.
- The disk cache stores the code-object blob instead of cubin (key on gcnArchName instead of
  compute capability). Keep the deterministicHash integrity check.
- `cuModuleGetFunction`/`cuLaunchKernel` -> `hipModuleGetFunction`/`hipModuleLaunchKernel` (these
  are in Kernel.hpp/jit.cpp, the runtime side).

## Build commands (gfx90a, headless)
See plan.md "Build commands". Cap `-j 16`. CPU backend stays ON as the in-process reference for
the gtest CPU-vs-GPU diffs.

## Validatable core (first `ported`)
arith/JIT, reduce/scan/ireduce, sort/sort_by_key/set, blas/dot/matmul, fft/fft_real,
random/rng_quality, transpose/reorder/join/moddims, index/assign/lookup. Deferred (documented):
cudnn-gated convolveNN (MIOpen), graphics/imageio (Forge/FreeImage), nonfree sift/gloh, the CV
kernels (fast/harris/orb/sift).

## Gotchas log
(append as found)
