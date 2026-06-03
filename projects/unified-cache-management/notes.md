# unified-cache-management (UCM) notes

Has a device-backend abstraction (cuda/ascend/maca/musa/simu); the ROCm/HIP port
adds a new `rocm` backend alongside rather than rewriting, selected by
`-DRUNTIME_ENVIRONMENT=rocm` (PLATFORM=rocm in setup.py).

## What the port does (Strategy A, additive rocm backend)
- New compat shim `ucm/shared/vendor/hip_compat/{cuda_runtime.h,cuda.h}`: on a
  ROCm build the per-backend CMake puts this dir ahead of the toolchain
  includes so every `#include <cuda_runtime.h>` / `<cuda.h>` resolves here; it
  pulls in `<hip/hip_runtime.h>` and aliases the small cuda* runtime surface the
  KV-transfer backend uses (Malloc/Free/Memcpy[Async]/HostMalloc/HostRegister/
  Stream*/Event*) to hip*. The NVIDIA path never sees it.
- New `rocm/CMakeLists.txt` arms under `ucm/shared/trans`, `ucm/store/nfsstore/
  device`, and `ucm/sparse/gsa_on_device/csrc/rocm/ham_dist`. Each does
  `enable_language(HIP)`, `find_package(hip)`, marks the reused cuda `.cu`
  `LANGUAGE HIP`, and sets `CMAKE_HIP_ARCHITECTURES` from the cache var
  (default gfx90a only when unset -- never a literal). The store/trans arms
  reuse the cuda `.cc`/`.cu` sources verbatim (like the maca backend does).
- PTX rewrite (the one non-mechanical fix): the two grid-stride copy kernels use
  inline PTX (`ld.global.cs.v4.b32` / `st.volatile.global.v4.b32` in
  trans/cuda/cuda_sm_kernel.cu; `ld.global.cs.v2.u64` / `st.global.cg.v2.u64` /
  `st.volatile.global.v2.u64` in store/.../cuda/cuda_device.cu). Guarded with
  `#if defined(__CUDA_ARCH__)`; the HIP `#else` does a plain vectorized uint4
  load/store (32B and 16B units). ROCm 7.2.1 has no `__ldcs`/`__stcg`/`__stcs`
  (only `__ldg`), so unlike the maca template the portable plain uint4 copy is
  used; the PTX cache-streaming hints are perf-only, so the copy is
  semantically identical. CUDA path byte-identical (guards only).
- cp_async.cuh needed no change: `FLASHINFER_CP_ASYNC_ENABLED` is gated on
  `__CUDACC_VER_MAJOR__`, undefined under hipcc, so the portable `*(uint4*)`
  fallback is selected automatically.
- Hamming module (`paged_ham_dist_mla.cu`, links libtorch): no PTX, no warp
  collectives. Built as a HIP pybind extension against a ROCm torch. operator.h
  now includes `<ATen/hip/HIPContext.h>` under `USE_ROCM` (the cuda-spelled
  context header pulls in NVIDIA-only cuda_runtime_api.h/cusparse.h; the hipified
  header provides the same `c10::cuda::getCurrentCUDAStream` backed by HIP).
  Built at C++20 (torch 2.13 headers use `requires` clauses). `-ffast-math` is
  NOT used (the kernel uses INFINITY as a masking sentinel; clang fast-math
  implies -ffinite-math-only and would drop it). `pybind11_add_module(... NO_EXTRAS)`
  disables pybind's default LTO+strip, which under -fvisibility=hidden was
  dropping the PyInit module init symbol and made the .so unloadable.

## Build (gfx90a)
Store/trans C++ surface + unit tests (no torch), via `build_rocm.sh`:
```
cmake -S src -B build_rocm -DRUNTIME_ENVIRONMENT=rocm -DBUILD_UCM_STORE=ON \
  -DBUILD_UNIT_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-Wno-error=unused-result"
cmake --build build_rocm -j16
```
The `-Wno-error=unused-result` is a LOCAL build flag only (NOT a source change):
g++ 13.3 + `_FORTIFY_SOURCE=2` flags the upstream test helper
`ucm/store/test/case/detail/path_base.h` ignoring `system()` return values; this
is pre-existing and affects the NVIDIA build identically.

Sparse hamming module (needs a ROCm torch in the active env):
```
cmake -S src -B build_sparse -DRUNTIME_ENVIRONMENT=rocm -DBUILD_UCM_STORE=OFF \
  -DBUILD_UCM_SPARSE=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=<rocm-torch-python> -DCMAKE_CXX_FLAGS="-Wno-error=unused-result"
cmake --build build_sparse -j16 --target hamming
```

## GPU validation (gfx90a, ROCm 7.2.1, all serial, HIP_VISIBLE_DEVICES=0)
- C++ gtests via ctest: 79/80 PASS. The copy-kernel correctness gates all pass:
  - UCTransUnitTest.{CopyDataWithCE, CopyDataWithSM, CopyDataBatchWithSM} (3/3)
    -- byte-exact host<->device copy round-trip, exercises the rewritten
    cuda_sm_kernel.
  - UCPosixTransManager/Queue TransBlock[LayerWise], UCCacheTransBuffer cases --
    exercise the store cuda_device.cu H2D/D2H batch copy with readback.
  - The ONE failure, `UCMetricsUT.ConcurrentUpdateAndCollect`, is a pure-CPU
    multi-threaded metrics counter test in untouched `ucm/shared/metrics`; it
    fails deterministically here independent of the port (host concurrency
    assertion, no GPU). Not a regression.
- Hamming kernel: `ucm/sparse/test/gsa/test_hamming_rocm_ref.py` (new) computes
  an independent CPU popcount reference for the paged block-mode score and
  asserts the kernel matches. mla (no kv reduction) PASS within fp16 rounding
  (the kernel accumulates large sums in fp16); gqa (reduce_kvhead min over kv)
  PASS exact (max_abs_err=0); two-run determinism PASS bit-identical. The key
  block layout the kernel assumes is (num_blocks, num_kv_head, block_size,
  num_chunk) -- the reference indexes the flat buffer with the kernel's stride
  (kv*block_size+offset)*num_chunk+chunk to match.
  Run with HAMMING_DIR=<dir holding hamming*.so>.

## Fault classes hit
- Inline-PTX-only copy intrinsics -> portable uint4 load/store (no __ldcs/__stcg
  on ROCm 7.2.1; plain copy is semantically equivalent to the streaming hints).
- Warp size: N/A to correctness (no warp collectives, no literal 32; block dims
  256/512 with __syncthreads only). Fat-binary gfx90a;gfx1100 not required for
  correctness but followers validate on their own hardware.
- torch CUDA->HIP header spelling: ATen/cuda/CUDAContext.h pulls NVIDIA-only
  headers on ROCm; use ATen/hip/HIPContext.h under USE_ROCM.
- pybind11 NO_EXTRAS: default LTO+strip drops PyInit under -fvisibility=hidden.

## Gotchas for followers (gfx1100/gfx1151)
- The rocm CMake arms already read CMAKE_HIP_ARCHITECTURES; pass the target arch,
  no source edit needed.
- The hamming module needs a ROCm torch in the active env. If unavailable, the
  store/trans C++ gtests (no torch) are a sufficient GPU gate on their own.
- The UCMetricsUT.ConcurrentUpdateAndCollect failure is pre-existing/CPU-only;
  do not attribute it to the port.
