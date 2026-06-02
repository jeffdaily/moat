# ntransformer notes

## Port summary (linux-gfx90a, lead)
- Fork: https://github.com/jeffdaily/ntransformer, branch `moat-port`.
- Strategy A (pure CMake): `option(USE_HIP)` selects `project(... HIP)` vs `project(... CUDA)`; `set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)` compiles the `.cu` files as HIP without renaming them. NVIDIA path is byte-for-byte unchanged (USE_HIP default OFF).
- Compat header `src/core/cuda_to_hip.h` aliases the exact runtime + fp16 surface (1:1 hip*). A HIP-only shim dir `src/core/hip_compat/` holds `cuda_runtime.h` and `cuda_fp16.h` that just `#include "../cuda_to_hip.h"`, so the project's literal `#include <cuda_runtime.h>`/`<cuda_fp16.h>` resolve to HIP aliases. The shim dir is added with `target_include_directories(... BEFORE PUBLIC ...)` only under USE_HIP -- ROCm ships no cuda_*.h shims, hence this dir.
- Links `hip::host` (vs `CUDA::cudart`). HIP std = C++20.

## Build (gfx90a)
```
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build-hip -j
ctest --test-dir build-hip --output-on-failure
```
Multi-arch correctness build (must emit both code objects):
```
cmake -S . -B build-multi -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build-multi -j
roc-obj-ls build-multi/test_gemm   # -> gfx90a AND gfx1100
```
USE_GPUNVME stays OFF (needs external gpu-nvme-direct lib + VFIO/root; not GPU-validatable here).

## Validation (real gfx90a, GCD 1, ROCm 7.2.1, MI250X)
- Multi-arch build clean; `roc-obj-ls` shows both `gfx90a` and `gfx1100` native code objects in test_gemm and the lib.
- `ctest`: test_tensor PASS, test_gemm PASS (gemv f32/q4_0/q6_k smem + q6_k_large no-smem, silu_mul, rmsnorm -- exact/tolerance matches).
- Deterministic across 3 runs (byte-identical output). AMD_LOG_LEVEL=3 confirms `Using native code object for device: gfx90a`.
- End-to-end GGUF decode intentionally NOT run (no model downloaded; host egress is slow). Unit tests are the agreed gate.

## Gotchas (the two real build-breakers, both the __HIPCC__-vs-__CUDACC__ trap)
1. `src/core/types.h` guarded BOTH the fp16 `half` typedef (line ~12) AND the `NT_CUDA_CHECK` macro (line ~223) on `#ifdef __CUDACC__`. hipcc defines `__HIPCC__`, not `__CUDACC__`, so every device TU fell through (uint16_t typedef / undefined NT_CUDA_CHECK) and failed to compile. Fix: `#if defined(__CUDACC__) || defined(__HIPCC__)` on both. (This is the MPPI-Generic trap; note there were TWO such guards in this file, not one -- grep the whole file for `__CUDACC__`.)
2. ROCm 7.2.1 `__shfl_xor_sync` requires a 64-bit lane mask and statically rejects an implicitly promoted 32-bit literal (`static_assert ... sizeof(MaskT)==8`). The kernels passed `0xFFFFFFFF`. Fix: a backend-portable `NT_WARP_MASK` in types.h (`0xFFFFFFFFFFFFFFFFULL` on `__HIPCC__`, else `0xFFFFFFFFu`), substituted at all 28 `__shfl_xor_sync` sites in gemm.cu/attention.cu/rmsnorm.cu/softmax.cu. Arch-unified: full-wavefront constant is correct on wave64 (all 64 lanes) and wave32 (high bits ignored). On CUDA it is the original 32-bit literal -> NVIDIA path semantically unchanged.

## Warp-width analysis (confirmed during port)
- rmsnorm/softmax/attention reductions use the `warpSize` builtin (offset starts at `warpSize/2`) and a `__shared__ float[32]` cross-warp buffer (32 = wave32 upper bound at 1024 threads; safe on wave64 at 16 warps). Wave-agnostic, no change.
- gemm.cu GEMV is a logical-warp-of-32 butterfly: `block(32, GEMV_WARPS=8)`, `flat_id = warp_id*32+tid`, `b += 32`, reduction `offset` starts at 16. Every shfl offset <= 16 so the XOR butterfly never crosses the 32-lane boundary; on wave64 two 32-rows pack into one wavefront but each reduces independently and correctly. Correct on wave64 AND wave32, no change. test_gemm's exact dot-product checks (e.g. y[0]=256 q6_k, 32768 q6_k_large) confirm this on gfx90a.
- `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 64KB)` -> `hipFuncSetAttribute`; accepted on gfx90a, tests pass (q6_k_large exercises the no-smem fallback anyway).
- nodiscard warnings on `cudaFree`/`cudaEventDestroy` etc. in streamer.cu/gemm.cu are benign (HIP marks these [[nodiscard]]; upstream ignores the return). Not errors.

## Followers (gfx1100 / gfx1151)
- Multi-arch build already proves gfx1100 compiles and emits a code object. Build is `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (no source edit). Validate a wave32 run of test_gemm/test_tensor.
- Watch points on wave32: the `warpSize`-based reductions and the logical-32 GEMV butterfly are wave-agnostic by construction (see analysis above); NT_WARP_MASK high bits are ignored on wave32. Expect a clean pass; delta-port only if a numeric test fails.
