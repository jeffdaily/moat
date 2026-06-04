# qrack notes

GPU quantum-computer simulator. Strategy-A HIP port of the self-contained CUDA
backend (QEngineCUDA / CUDAEngine). No prior ROCm/HIP backend existed; AMD was
reachable only via the OpenCL backend. Net-new native ROCm/HIP backend.

- Upstream: https://github.com/unitaryfoundation/qrack (base 49d2efc7)
- Fork: https://github.com/jeffdaily/qrack, branch moat-port
- Lead platform: linux-gfx90a (MI250X), ROCm 7.2.1

## What changed (6 files)
- ADD include/common/cuda_to_hip.h -- the only HIP-aware file. On USE_HIP it
  includes <hip/hip_runtime.h> (+ hip_fp16.h on FPPOW<5) and aliases the ~20
  cudaXxx symbols this backend uses to hipXxx, plus `#define CUDART_CB` (empty;
  CUDA-only callback tag). On NVIDIA it is inert and includes <cuda_runtime.h>.
- cmake/CUDA.cmake -- `option(USE_HIP)`. When set: enable_language(HIP), the
  three .cu marked LANGUAGE HIP, CMAKE_HIP_ARCHITECTURES defaults to gfx90a only
  when unset, `find_package(hip)` + link `hip::host` so the HOST C++ TUs (which
  pull HIP vector types float2/float4/make_float2 via qrack_types.hpp) get the
  ROCm headers + __HIP_PLATFORM_AMD__. CUDA path unchanged (elseif branch).
- CMakeLists.txt -- gate the nvcc-only QRACK_CUDA_COMPILE_OPTS (-use_fast_math,
  -Werror all-warnings, --ptxas-options, -Xcompiler, --cudart=shared) behind the
  non-HIP path; add QRACK_HIP_COMPILE_OPTS = -O3 -ffp-contract=on -fno-math-errno
  and a `$<COMPILE_LANGUAGE:HIP>` generator-expression line in each
  target_compile_options block.
- include/common/qrack_types.hpp -- route the `#include <cuda_runtime.h>` (and
  the FPPOW<5 `cuda_fp16.h`) through cuda_to_hip.h. THIS header is included by
  EVERY TU, so on HIP the host compiler must resolve HIP vector types -> hence
  hip::host on the qrack target.
- include/common/cuda_kernels.cuh, include/common/cudaengine.cuh -- include the
  compat header instead of bare cuda_fp16.h / cuda_runtime.h. CUDADeviceContext
  made non-copyable/non-movable (= delete) -- it owns two streams destroyed in
  its dtor; double hipStreamDestroy faults on ROCm (rule-of-five).

## Build recipe (gfx90a, ROCm 7.2.1)
```
cd projects/qrack/src
cmake -S . -B _build -DENABLE_CUDA=ON -DENABLE_OPENCL=OFF \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DENABLE_TESTS=ON
cmake --build _build -j16 --target unittest
```
Config detected QBCAPPOW=7, FPPOW=5 (float -- fp16 path NOT compiled, off the
default validation path). Build clean; the three .cu compile as "Building HIP
object", links "HIP static library libqrack.a" + "HIP executable unittest".

Multi-arch check (build only): add -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100";
both `--offload-arch=gfx90a` and `--offload-arch=gfx1100` appear in the HIP
compile commands and compile cleanly -> wave-size class is source-clean (same
source builds wave64 and wave32, no per-arch guard).

## Validation (real GPU, HIP_VISIBLE_DEVICES=3)
The unittest binary selects the GPU engine at compile time; with ENABLE_OPENCL
OFF + ENABLE_CUDA ON, `--proc-cuda --layer-qengine` binds testEngineType to
QINTERFACE_CUDA (now HIP -> QEngineCUDA) and runs circuits vs analytic/CPU
references. Init prints "CUDA device #0: AMD Instinct MI250X / MI250" on GCD3.

Every assertion that completed on the HIP engine PASSED, no mismatches against
references, across: complex/par_for helpers, exp2x2/log2x2, lossy save/load,
getmaxqpower, setconcurrency, highestproball, global_phase, cnot/anticnot/
anticy, apply_single_bit/invert, u, s/is/t/it/sh, cs/cis/ct/cit, x, xmask/
ymask/zmask, phaserootnmask, phaseparity. (8411 assertions across 32 cases in
one run; 8346 across 15 cases in another -- both interrupted only by my own
timeout SIGTERM on an in-progress slow test, never by an assertion failure.)

GOTCHA -- test wall-clock: the unittest fixture builds 20-qubit registers
(2^20 state vectors) and several tests sweep over all qubit positions
(test_apply_controlled_single_bit and the controlled/swap/fsim family are
O(n) or O(n^2) gate applications; test_approxcompare and the [mirror] tests
likewise). At width 20 on the GPU these individual tests run 5-10+ minutes
EACH, so the full --proc-cuda sweep takes hours. The width is hardcoded (and
several tests reference bit indices 18/19, so it cannot be lowered below 20
without breaking them). For a quick gate, run the fast subset and exclude the
slow sweeps with separate `~name` args (NOT comma-joined -- Catch2 ignores
`~` inside a single comma token; pass each as its own argument):
`_build/unittest --proc-cuda --layer-qengine -d yes ~test_ucmtrx ~test_ccnot ...`
Use `-d yes` for per-test progress and redirect to a FILE (a `| tail` pipe
buffers everything and loses it when a timeout SIGTERMs the run).

## Fault classes encountered
- CUDART_CB: CUDA-only host-callback calling-convention macro on the
  cudaLaunchHostFunc callback; HIP has no equivalent -> `#define CUDART_CB` to
  empty in the compat header.
- Host TUs need HIP headers: qrack_types.hpp pulls HIP/CUDA vector types
  (float2/float4/make_float2) into EVERY TU including plain .cpp built by the
  host compiler. On HIP that needs <hip/hip_runtime.h> + __HIP_PLATFORM_AMD__ in
  scope for the host compiler -> link `hip::host` on the qrack target (it
  propagates the include dirs + platform define). Without it the .cpp TUs fail
  with "cuda_runtime.h: No such file or directory".
- Rule-of-five: CUDADeviceContext (two hipStream_t, dtor destroys them) made
  non-copyable/non-movable.
- -ffp-contract: clang for HIP defaults to fast (cross-statement FMA); pinned
  -ffp-contract=on to match nvcc and avoid float drift on tolerance compares.
- Wave size: NO source change. Reductions are __syncthreads-fenced shared-mem
  trees syncing at every level (no last-warp elision); block dim from runtime
  warpSize (64 on gfx90a). Builds for gfx90a + gfx1100 from one source.
- ZERO library swaps (no cuBLAS/cuFFT/Thrust/CUB/cuRAND in this backend).

## Follower notes (gfx1100 / gfx1151)
Reuse this fork branch; validate first. No wave-size source delta anticipated
(source already compiles wave32). fp16 path (FPPOW<5, __half2) is NOT built on
the default float config and is unvalidated -- a follow-up if needed.
