# raft notes (ROCm/HIP port)

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated
RAPIDS). raft is the second RAPIDS base after rmm; cuvs, cugraph, and cuml all
build on it. The `## Install as a dependency` section below is the contract
those consume.

Pinned upstream: tag `v25.08.00` (commit 2fb92393, in `projects/raft/src`,
gitignored). Chosen to match the rmm `v25.08.00` we built (API/ABI), and so the
same vendored ROCm/libhipcxx CCCL surface (carried by the rmm install) applies.

## Existing AMD support

None. No ROCm/HIP branch or fork upstream. From-scratch CUDA-to-HIP, Strategy A.

## Build classification: Strategy A (header-heavy pure-CMake library)

raft is a header-only C++ library (~755 headers under `cpp/include/raft/`) plus
a small compiled `libraft.so`. The stock libraft compiles 8 explicit-
instantiation `.cu` (rmat generators x4 + lanczos solvers x4); the lanczos 4
are deferred on HIP (see below), so our libraft is the 4 rmat TUs. Not a torch
extension -> Strategy A. The headers are the deliverable RAPIDS links against.

depends_on: **rmm** (consumed via find_package(rmm) + rmm::rmm; the rmm install
also carries the vendored libhipcxx cuda::std/cuda::mr and the rmm compat
header, force-included on every HIP TU).

## CCCL redirect (the recurring RAPIDS obstacle) -- same solution as rmm

raft's stock `cpp/CMakeLists.txt` bootstraps rapids-cmake and FETCHES NVIDIA
CCCL (`rapids_cpm_init` + `get_cccl`, gated on `CUDAToolkit REQUIRED` +
`enable_language(CUDA)`), plus cuco + CUTLASS. Same fix as rmm: a top-of-file
`option(USE_HIP) ... if(USE_HIP) include(cmake/hip/raft_hip.cmake) return()`
guard bypasses the whole bootstrap. CCCL is satisfied by include paths, no fetch:
- Thrust -> rocThrust, CUB -> hipCUB: `/opt/rocm/include` (hipcc default path).
  raft `#include <cub/cub.cuh>` is redirected by a `hip_compat/cub/cub.cuh` shim
  (`#include <hipcub/hipcub.hpp>` + `namespace cub = hipcub;`).
- libcudacxx `cuda::std` / `cuda::mr`: vendored ROCm/libhipcxx, carried by the
  rmm install (`-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE` comes from
  rmm::rmm's exported defs).

cuco and CUTLASS are NOT fetched in the HIP path: the compiled libraft and the
validated header subset do not pull them (CUTLASS is used only in
`distance/detail/fused_distance_nn` + a few distance detail files; cuco in 1
sparse file -- both off the validated path). cuco/CUTLASS get addressed when the
distance/neighbors modules are brought up.

The standalone HIP build lives in `cpp/cmake/hip/raft_hip.cmake` (+
`raft_hip_tests.cmake`). It regenerates the headers rapids-cmake would
(`version_config.hpp`; `logger_macros.hpp` via rapids-logger's
`create_logger_macros.cmake` -- reuse the rmm-vendored rapids_logger clone),
builds libraft from the 4 rmat `.cu` marked LANGUAGE HIP, force-includes the
raft compat header, and exports `raft::raft` / `raft::raft_lib` /
`raft::raft_compiled` + a minimal `raft-config.cmake`.

## Compat header + shims

- `cpp/include/raft/util/hip/cuda_to_hip.h` -- the raft compat header,
  force-included on every HIP TU. Adds the runtime symbols rmm's compat does not
  cover (occupancy, func-cache, events, datatype enums CUDA_R_*/cudaDataType),
  fp16/bf16 type aliases (`__nv_bfloat16` -> `__hip_bfloat16`), `curand` ->
  `hiprand`, `CUSOLVERAPI` (empty), HIP fallbacks for the NVIDIA cache-streaming
  intrinsics (`__ldcv`/`__ldcs`/`__stwt`/`__stcs` -> plain load/store), and the
  math-library alias block. CRITICAL: it defines `__CUDA_ARCH__ 800` ONLY in the
  HIP device pass (`__HIP_DEVICE_COMPILE__`) -- see fault classes.
- `cpp/include/raft/util/hip/raft_mathlib_aliases.inc` -- 250+ cuBLAS/cuSOLVER/
  cuSPARSE/cuRAND symbol aliases, GENERATED from pytorch's hipify
  cuda_to_hip_mappings (agent_space/raft_gen_compat.py), filtered to raft's
  actual symbol surface, plus a hand-curated tail for symbols hipify omits
  (cusolverStatus_t/CUSOLVER_STATUS_*, cusolverSp handle trio, syevjInfo_t,
  cublasLt enum constants, EIG_RANGE, SDDMMAlg_t, syevdx). hipBLAS/hipSOLVER/
  hipSPARSE/hipRAND mirror the cu* APIs name-for-name.
- `cpp/hip_compat/` -- forwarding shims on the HIP include path (BEFORE) for the
  CUDA-named toolkit headers: `cublas_v2.h`/`cublasLt.h` -> hipblas/hipblaslt,
  `cusolverDn.h`/`cusolverSp.h` -> hipsolver, `cusparse.h`/`cusparse_v2.h` ->
  hipsparse, `curand.h`/`curand_kernel.h` -> hiprand, `cuda_fp16.h(pp)` /
  `cuda_bf16.h(pp)` / `library_types.h` / `vector_types.h` -> hip equivalents,
  `cub/` tree -> hipcub, `cooperative_groups.h` -> hip CG, and a small
  `math_constants.h` providing the CUDART_INF* RAFT uses. On NVIDIA the dir is
  absent so the real toolkit headers win.

## Fault classes hit (and fixes) -- all guarded so the CUDA path is unchanged

1. `__CUDA_ARCH__` undefined on HIP (the single highest-leverage fault).
   `core/math.hpp` and others gate device-vs-host on `#ifdef __CUDA_ARCH__`
   (`::cos(x)` device vs `std::cos(x)` host) AND numeric `#if __CUDA_ARCH__ >=
   530/800` for __half/bf16 intrinsics. hipcc does not define __CUDA_ARCH__, so
   device branches silently took the HOST path and faulted at runtime
   (MathDevice.Cos coredumped). Fix: define `__CUDA_ARCH__ 800` in the HIP
   device pass only (compat header). HIP provides ::hcos/::hsin/::hexp/::hsqrt/
   ::hlog so the >=800 half/bf16 branches are correct.
2. Enabling `__CUDA_ARCH__` activates previously-dead NVIDIA device code that
   needs HIP handling: `__dp4a` (cuda_utils dp4a), `__nanosleep` (interruptible
   test), wide-mul PTX (integer_utils wmul_64bit) -- each has a portable
   fallback, so guard the NVIDIA-intrinsic branch with `&& !defined(USE_HIP)` to
   take the fallback. `__ldcv`/`__stwt` (linewise_op) have no fallback -> add
   HIP device-fn templates (plain load/store) in the compat header.
3. wave64 warp masks. HIP's `__*_sync` intrinsics static_assert a 64-bit mask
   regardless of wave width, so the CUDA `0xffffffffu` literal will not even
   COMPILE. Central fixes: `util/warp_primitives.cuh` -> a `WarpMask`/
   `RAFT_LANE_MASK_ALL` portable type (uint64 on HIP); `util/cudart_utils.hpp`
   `warp_full_mask()` -> 64-bit on HIP and `warp_size()` -> 64 on __GFX9__;
   `util/cuda_dev_essentials.cuh` `WarpSize` -> 64 on __GFX9__ and `laneId()`
   -> `__lane_id()`; `util/reduction.cuh` `binaryBlockReduce` -> 64-bit ballot +
   `__popcll`.
4. hipBLAS getriBatched takes non-const A[]/ipiv where cuBLAS takes const ->
   const_cast at the HIP call site (cublas_wrappers.hpp).
5. hipBLAS status enum is not 1:1 -- `CUBLAS_STATUS_LICENSE_ERROR` has no
   hipBLAS value -> USE_HIP-guard that one `case` out (cublas_macros.hpp).
6. hipSOLVER gaps: no generic `cusolverDnXsyevd` and no cusolverSp batched-QR
   (`csrqrsvBatched`/`csrqrInfo_t`). Guarded those wrapper groups out of
   `cusolver_wrappers.hpp` under USE_HIP, and routed `eigDC` to the typed-syevd
   `eigDC_legacy` path (hipsolverDn{S,D}syevd) which computes the identical
   eigendecomp. hipSOLVER's device ::sincos is double-only -> route float to
   ::sincosf (math.hpp).

## Build (lead, gfx90a) -- repeatable script

Prereqs: the ported rmm installed at `_deps/raft-rmm/install` (see
projects/rmm/notes.md "Install as a dependency"); the rmm-vendored rapids_logger
clone at `agent_space/rapids_logger`; conda env py_3.12 (gtest/gmock/spdlog/fmt).

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/raft/src/cpp -B projects/raft/build -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/raft/install
cmake --build projects/raft/build -j$(nproc)
```

Arch is taken from `CMAKE_HIP_ARCHITECTURES` (defaults to gfx90a only when
unset), so a follower validates with only `-DCMAKE_HIP_ARCHITECTURES=<arch>`.

## Validation (gfx90a, HIP_VISIBLE_DEVICES=1)

Prioritized subset built + GPU-validated (real allocations/kernels/cuBLAS on an
MI250X). The crashing/blocked groups (below) are excluded via the
`RAFT_TEST_*` options in `raft_hip_tests.cmake`.

| Group         | GPU result        | Notes |
|---------------|-------------------|-------|
| LABEL_TEST    | 14/14 PASS        | full suite |
| RANDOM (sub)  | 250/250 PASS      | MakeBlobs 64, Rng 44, RngMdspan, RngNormalTable 8, Perm 66, RmatGen 24 (exercises libraft.so) |
| CORE (sub)    | 165/165 PASS      | MathDevice 17, OperatorsDevice 29, MathHost 17, OperatorsHost 29, Span/GPUSpan 27, NumPySerializer, MemoryType, BitmapTest 24, sparse-matrix containers, coordinate structure |
| UTILS (sub)   | 36/36 PASS        | integer_utils, pow2, device_atomics, cudart_utils, seive |
| LINALG (sub)  | 149/150 PASS      | element-wise (add/sub/mul/div/sqrt/pow/unary/binary/ternary), map, transpose 24, cuBLAS gemv 10 / dot 14 / axpy 14; 1 DotTestF float-tolerance fail |

Total: ~614 gtest cases pass on gfx90a. find_package(raft)+raft::raft also
validated end-to-end via a standalone consumer (agent_space/raft_consumer:
device_mdarray + raft::linalg::add on a GPU stream -> correct result).

### Deferred (documented, off the validated path)

GPU-crash subsets within otherwise-passing groups (wave64 algorithm-correctness
work remaining, not just compile): UTILS BitonicTest / PopcTest / ReductionTest /
MemoryTypeDispatcher; CORE DeviceResourcesManager / MDSpanCopy.*Cuda (device
transpose copy) / BitsetTest / Raft.InterruptibleOpenMP / MDSpan.AlignedMatrix
(host-alignment assert); RANDOM Bernoulli*; LINALG Reduce/gemm/eig/svd suites.

Whole test groups not yet building under HIP (`RAFT_TEST_* OFF`), each blocked
by a documented gap:
- MATRIX, STATS: `thrust::cuda::par` exec-policy delta (-> thrust::hip::par, the
  rmm/cupoch Thrust lesson; in scores/contingencyMatrix/adjusted_rand_index) +
  CUTLASS (distance epilogue pulled transitively) + cudaProfiler/L2-cache attrs.
- SPARSE: more NVIDIA PTX (`__ldcg`, `=l` asm in sparse/distance/detail/utils),
  `atomicAdd_block`, and `thrust::cuda::par` in the test sources.
- SOLVERS: cusolverSp batched-QR + cusolverDnXsyevd (no hipSOLVER equivalent).
- NEIGHBORS: CUTLASS fused-distance + faiss_select `static_assert(WarpSize==32)`
  (a hard wave32 assumption).
- DISTANCE: CUTLASS-based (fused_distance_nn) -- the largest single port item.

The compiled lanczos solvers (4 of libraft's stock 8 TUs) are deferred behind
`RAFT_COMPILE_LANCZOS=OFF` for the same cusolverSp/Xsyevd reasons.

## Install as a dependency

This is the contract cuvs / cugraph / cuml consume. raft is header-heavy: a
dependent gets the raft headers, the hip_compat shims, the generated
version_config/logger_macros, and links `libraft.so`. raft transitively pulls
rmm, so the dependent's CMAKE_PREFIX_PATH must include BOTH the raft and rmm
install prefixes.

### 1. Build + install (after rmm is installed at _deps/raft-rmm/install)

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S <raft>/cpp -B <raft>/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/raft/install
cmake --build <raft>/build-hip --target install -j$(nproc)
```

(For the MOAT deps workflow: clone `jeffdaily/raft @ moat-port` into
`_deps/raft/src`, apply the working-tree port if delivering via fork, install to
`_deps/raft/install`.)

### 2. Install-prefix layout

```
<prefix>/include/raft/...                       raft headers (+ version_config.hpp, core/logger_macros.hpp)
<prefix>/include/raft/util/hip/cuda_to_hip.h    the compat header (force-included into HIP consumers)
<prefix>/include/raft/hip_compat/...            cublas_v2.h / cuda_fp16.h / cub/ ... forwarding shims
<prefix>/include/raft_runtime/...               runtime API headers (for raft::compiled)
<prefix>/lib/libraft.so
<prefix>/lib/cmake/raft/raft-config.cmake       + raft-targets.cmake (+ version)
```

### 3. What a dependent (cuvs/cuml) sets to consume it

```
find_package(raft REQUIRED)                # resolves via CMAKE_PREFIX_PATH below
target_link_libraries(<tgt> PRIVATE raft::raft)            # header-only API
# raft::compiled (libraft.so) for the explicit-instantiation runtime symbols
```

Configure the dependent with BOTH prefixes (raft pulls rmm):
```
-DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft/install;/var/lib/jenkins/moat/_deps/raft-rmm/install;/opt/rocm;$CONDA_PREFIX"
```

The exported `raft::raft` propagates, automatically:
- include dirs: raft headers + the `hip_compat` shim dir (BEFORE);
- compile defs: `USE_HIP`, `__HIP_PLATFORM_AMD__` (and, via rmm::rmm,
  `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE`);
- a HIP-language force-include of the installed `raft/util/hip/cuda_to_hip.h`;
- link: rmm::rmm, hip::host, roc::hipblas, roc::hipsparse, and (if found)
  roc::hipsolver / roc::hipblaslt / hip::hiprand, OpenMP.

The dependent must itself `enable_language(HIP)` and compile any TU that
includes raft headers as HIP (`set_source_files_properties(... LANGUAGE HIP)`),
because raft headers reach rocThrust/hipCUB/libhipcxx device headers that only
clang-as-hipcc parses. raft-config does `find_dependency(hip rmm hipblas
hipsparse)` -- keep the rmm + conda + rocm prefixes on CMAKE_PREFIX_PATH. At
runtime put `<raft>/lib`, `<rmm>/lib`, and `$CONDA_PREFIX/lib` on
LD_LIBRARY_PATH.

### 4. Verified

A standalone consumer (`agent_space/raft_consumer/`) doing exactly the above --
`find_package(raft)` + `raft::raft`, a `raft::device_resources` +
`raft::make_device_vector` + `raft::linalg::add` on a GPU stream -- compiles
against the install tree (only CMAKE_PREFIX_PATH set) and runs correctly on
gfx90a: `raft consumer: add -> 11 22 33 44 (expect 11 22 33 44) via
find_package(raft)`.
