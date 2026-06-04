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

## Review 2026-06-04 (reviewer, linux-gfx90a)
Verdict: review-passed. Additive Strategy-A rocm backend; NVIDIA path is guard-only
(no change inside any `#if defined(__CUDA_ARCH__)` branch), other backends untouched.
No must-fix defects. Fault-class sweep clean: no hardcoded warpSize/32 (the
`CUDA_TRANS_BLOCK_NUMBER (32)` is the grid block count `<<<32,256>>>`, not a wave
width); no warp collectives (copy kernels are grid-stride memcpy, hamming uses
512-thread blocks + __syncthreads + __popc/__popcll only -- wave-agnostic); no
textures/surfaces/resource handles (rule-of-five N/A); no library swaps; uint4
vector path inherits the same 16B alignment requirement the PTX .v4.b32/.v2.u64
already imposed, so no new OOB/alignment fault. cp_async.cuh PTX is excluded under
hipcc (FLASHINFER_CP_ASYNC_ENABLED gated on __CUDACC_VER_MAJOR__) -> portable uint4
fallback, as claimed. Commit hygiene clean ([ROCm] subject 63 chars, Claude
disclosed, no noreply trailer, no MOAT jargon, Test Plan present).

Verified (not a defect, recorded so it is not re-flagged): the rocm trans target
omits gdr sources in ucm/shared/trans/rocm/CMakeLists.txt, but the parent
ucm/shared/trans/CMakeLists.txt appends gdr_mr_buffer.cc/gdr_config.cc via
target_sources for every non-cuda backend and sets UCM_ENABLE_GDR_STREAM=0, so the
unconditional GdrMrBuffer::* calls in cuda_buffer.cc/cuda_device.cc resolve. A clean
reconfigure from the committed tree reproduces the gdr objects in libtrans.a.

Volatile-store / uint4 semantic-equivalence judgment (the key question): EQUIVALENT.
The PTX st.volatile.global / st.global.cg qualifiers are within-kernel
compiler/cache-policy hints (defeat dead-store elimination + write-combining,
streaming bypass of L1); they do NOT encode a cross-kernel ordering or
host-visibility guarantee. Each thread writes a disjoint 32B/16B unit, so there is
no intra-grid reader and inter-thread store ordering is irrelevant. The only
consumer is the host after hipStreamSynchronize / the stream callback, and the
kernel-completion barrier makes all device writes globally visible regardless of the
volatile qualifier. The HIP path casts away the parameter's `volatile` and does a
plain uint4 store -- safe here. Caveat (informational only, no consumer exists): if a
future change added a mid-kernel peer/host poller of the destination buffer, the
plain store would not carry the per-store visibility the volatile PTX implied; worth
a one-line comment but not blocking. The new test_hamming_rocm_ref.py is a genuine
independent CPU popcount reference (exact integer ref, fp16-tolerance compare,
inf-mask check, two-run determinism) -- a real correctness gate, not a smoketest.

## Validation 2026-06-04 (linux-gfx90a, ROCm 7.2.1, gfx90a / MI250X)

GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0, ROCm 7.2.1 / HIP 7.2.53211.
Fork: jeffdaily/unified-cache-management moat-port @ 7a8eb04b96c8045ff5ea1164c42c30fa81399355.

Build commands:
```
cmake -S projects/unified-cache-management/src -B projects/unified-cache-management/build_rocm \
  -DRUNTIME_ENVIRONMENT=rocm -DBUILD_UCM_STORE=ON -DBUILD_UNIT_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-error=unused-result"
cmake --build projects/unified-cache-management/build_rocm -j16

cmake -S projects/unified-cache-management/src -B projects/unified-cache-management/build_sparse \
  -DRUNTIME_ENVIRONMENT=rocm -DBUILD_UCM_STORE=OFF -DBUILD_UCM_SPARSE=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/opt/conda/envs/py_3.12/bin/python3 \
  -DCMAKE_CXX_FLAGS="-Wno-error=unused-result"
cmake --build projects/unified-cache-management/build_sparse -j16 --target hamming
```

gfx90a code objects confirmed:
- ucm/shared/trans/rocm/CMakeFiles/kernel.dir/__/cuda/cuda_sm_kernel.cu.o: hipv4-amdgcn-amd-amdhsa--gfx90a
- ucm/store/nfsstore/device/rocm/CMakeFiles/storedevice.dir/__/cuda/cuda_device.cu.o: hipv4-amdgcn-amd-amdhsa--gfx90a
- hamming.cpython-312-x86_64-linux-gnu.so: hipv4-amdgcn-amd-amdhsa--gfx90a

C++ gtest (ctest -j1, HIP_VISIBLE_DEVICES=0): 79/80 PASS.

Copy-kernel correctness gates (all PASS):
- UCTransUnitTest.CopyDataWithCE: PASS (378 ms)
- UCTransUnitTest.CopyDataWithSM: PASS (343 ms)
- UCTransUnitTest.CopyDataBatchWithSM: PASS (334 ms)
- UCPosixTransManagerTest.TransBlock: PASS
- UCPosixTransManagerTest.TransBlockLayerWise: PASS
- UCPosixTransQueueTest.TransBlock: PASS
- UCPosixTransQueueTest.TransBlockLayerWise: PASS
- SharedCondition/UCCacheTransBufferTest.* (12 tests): all PASS

One failure (pre-existing, NOT a regression): UCMetricsUT.ConcurrentUpdateAndCollect.
This is a pure CPU multi-threaded counter test in untouched ucm/shared/metrics
(expectedUpdates=16000, totalCounter=15995 -- host-side atomic race in UpdateStats).
No GPU, HIP, or ROCm code path involved. Confirmed by inspecting metrics_test.cc:
it spawns 8 threads calling UpdateStats() concurrently with GetAllStatsAndClear();
the counter is not losslessly atomic under concurrent clear. Pre-existing in upstream.

Hamming kernel (test_hamming_rocm_ref.py, HIP_VISIBLE_DEVICES=0):
```
HAMMING_DIR=build_sparse/ucm/sparse/gsa_on_device/csrc/rocm/ham_dist \
  python3 ucm/sparse/test/gsa/test_hamming_rocm_ref.py
```
- mla: PASS (max_abs_err=42.0, max_rel_err=1.25e-03, within fp16 tolerance)
- gqa: PASS (max_abs_err=0.0, max_rel_err=0.00e+00, exact)
- determinism: PASS (two-run bit-identical)
ALL HAMMING TESTS PASSED

Result: linux-gfx90a COMPLETED at 7a8eb04b96c8045ff5ea1164c42c30fa81399355.
