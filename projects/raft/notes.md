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

ROCm path for the CUTLASS-based items above (DISTANCE / NEIGHBORS, and the
MATRIX/STATS distance epilogue): CUTLASS does NOT port to ROCm. These kernels
must be REIMPLEMENTED against Composable Kernel (CK), preferring ck_tile -- not
a 1:1 API, so understand the CUTLASS kernel's intent and write the CK equivalent
from CK's in-repo examples. This is the real gate for cuvs/cuml, not a CUTLASS
port.

The compiled lanczos solvers (4 of libraft's stock 8 TUs) are deferred behind
`RAFT_COMPILE_LANCZOS=OFF` for the same cusolverSp/Xsyevd reasons.

## CUTLASS distance/neighbors -> CK design (gfx90a)

### What the CUTLASS distance kernels actually do (intent, not API)

raft's distance machinery has TWO implementations of every distance, selected at
runtime by `pairwise_matrix_dispatch` (dispatch-inl.cuh) and the fused-NN impls:

1. A CUTLASS path on SM80+ (`dispatch_sm80.cuh` -> `pairwise_distance_cutlass_base.cuh`,
   and `fused_distance_nn/cutlass_base.cuh`). It is a `GemmUniversal` whose EPILOGUE
   is the distance op: the GEMM computes `accVal = dot(x_i, y_j)`, and the epilogue
   `PairwiseDistanceEpilogueElementwise` applies `out[i][j] = distance_op(xn[i],
   yn[j], accVal)` where `xn`/`yn` are the per-row/per-col L2 norms passed as a C
   "vector" (row broadcast) and a broadcast vector (col broadcast). For expanded-L2:
   `||x_i||^2 + ||y_j||^2 - 2*dot`; for cosine: `1 - dot/(||x_i|| ||y_j||)`.
2. A "legacy" hand-written SIMT path (`pairwise_matrix/kernel_sm60.cuh` ->
   `PairwiseDistances` in `pairwise_distance_base.cuh`; fused-NN's
   `simt_kernel.cuh` / `fusedL2NNkernel`). NO CUTLASS: it tiles X and Y into shared
   memory, accumulates the same `acc += x*y` dot product in registers, and applies
   the SAME `distance_op.epilog(...)` formula. Mathematically identical to the
   CUTLASS path, just a generic SIMT GEMM instead of tensor-core MMA.

So "expanded-L2 = GEMM + epilogue" is literally the design of BOTH paths. The CK
reimplementation target is path (1): replace the CUTLASS tensor-core GEMM+epilogue
with a CK/ck_tile MFMA GEMM + the same distance epilogue. The mapping is:

| CUTLASS construct | CK / ck_tile equivalent |
|---|---|
| `cutlass::gemm::device::GemmUniversalAdapter` | ck_tile universal GEMM (tile GEMM pipeline) or classic CK `DeviceGemmMultipleD` |
| tensor-core MMA (sm80 `mma.sync`) | CK MFMA warp-gemm `WarpGemmMfmaF32F32F32M16N16K16` (gfx90a `v_mfma_f32_16x16x16f32`) |
| `PairwiseDistanceEpilogueElementwise(aNorm, bNorm, accVal)` | a CK CDE element op / ck_tile epilogue functor that reads the GEMM accumulator + two broadcast norm tensors (xn row, yn col) |
| GEMM batching over N (gridYZMax) | CK tile partitioner over M*N tiles (no 65535 grid-Y limit on the relevant axis) |

ck_tile DOES support fp32 MFMA on gfx90a: `WarpGemmDispatcher<float,float,float,16,16,16>`
-> `WarpGemmMfmaF32F32F32M16N16K16` (warp_gemm_dispatcher.hpp). Classic CK's
`DeviceGemmMultipleD` maps the epilogue even more directly: its `CDEElementOp(e, c,
d0, d1)` is exactly `out = distance_op(xn=d0_row, yn=d1_col, acc=c)` with `d0`/`d1`
as 1-D broadcast tensors -- the same shape as the CUTLASS C-vector + broadcast-vec.

### fused_distance_nn (the largest item)

`fusedDistanceNNImpl` -> {`fusedL2NNImpl`, `fusedCosineNN`}. Same two-path split:
CUTLASS `cutlassFusedDistanceNN` (a GEMM whose epilogue, instead of writing the
distance matrix, does an argmin-reduction per row into a KeyValuePair<idx,dist>)
vs. the SIMT `fusedDistanceNNkernel`/`fusedL2NNkernel` (simt_kernel.cuh) which fuses
the same row-argmin into the tiled SIMT GEMM epilogue. The CK reimplementation is a
CK GEMM with a REDUCING epilogue (row-wise argmin of `distance_op(xn,yn,acc)`),
i.e. the CK `DeviceGemmReduce`/`gemm + block-row reduce` family, writing the
KeyValuePair output + the per-row mutex update.

### wave64 considerations for the CK/neighbors work

- The CUTLASS path encodes NVIDIA warp=32 implicitly in its MMA fragment layout;
  the CK MFMA path is wave64-native on CDNA (16x16x16 MFMA uses all 64 lanes), so
  the GEMM tiling is correct on wave64 by construction -- no 32-lane assumption to
  port, unlike the faiss top-k below.
- faiss_select top-k (neighbors) DOES hardcode wave32: see the faiss_select +
  select_warpsort fixes below. These are the wave64-correctness gate for neighbors.
- The SIMT fused-NN fallback's per-row `updateReducedVal` uses a spin-lock, but it
  is acquired one lane at a time across the unrolled `j` loop (only lane
  `lid==j*AccThCols` enters per step), so it does NOT hit the RXMesh wave64
  all-lanes-contend deadlock; it serializes by construction. shfl widths are the
  sub-warp `AccThCols`, correct on wave64 via raft::shfl's width arg.

### Strategy chosen for this pass (correctness path + one CK kernel)

Layer A (ships now, unblocks the neighbors/distance BUILD on ROCm): the runtime
arch detect (`kernel_virtual_arch` reads `ptxVersion`, meaningless on HIP) plus the
`#include <cutlass/...>` in the 5 CUTLASS files is what breaks the build. On HIP,
(a) make every `#include <cutlass/...>` and CUTLASS-only body `#if !defined(USE_HIP)`,
(b) force `pairwise_matrix_dispatch` / the fused-NN dispatchers to ALWAYS take the
legacy SIMT/sm60 branch (correct identical math), and (c) flip the SIMT-fallback
`#if __CUDA_ARCH__ < 800` guards to also compile their body when `USE_HIP` (the
compat header sets `__CUDA_ARCH__=800` on gfx90a, which would otherwise EMPTY the
fallback kernel body). This makes distance + neighbors build and run correct on
gfx90a using raft's own non-CUTLASS kernels -- the same approach raft itself uses
on pre-Ampere NVIDIA GPUs.

Layer B (the CK reimplementation deliverable): a standalone CK MFMA GEMM + fused
expanded-L2 distance epilogue, GPU-validated numerically against a CPU reference
(agent_space/ck_l2/ck_l2_expanded.cpp). Proves the CUTLASS->CK mapping concretely
on gfx90a and is the template for swapping the high-performance path into
dispatch_sm80 on HIP in a follow-up (replacing Layer A's legacy-kernel routing for
the large-matrix regime where the MFMA GEMM wins).

NOTE on ck_tile vs classic CK at ROCm 7.2.1: the in-repo ck_tile GEMM EXAMPLES
(composable_kernel/example/ck_tile/03_gemm) target a NEWER ck_tile high-level
kernel/epilogue API than the ck_tile headers SHIPPED in /opt/rocm/include/ck_tile
(7.2.1): e.g. the shipped `CShuffleEpilogueProblem` is the multi-ABD form
(AsDataType/BsDataType/DsDataType/CDElementwise/...) while the example passes the
older `<ADataType,BDataType,AccDataType,CDataType,CLayout,...>` positional form, so
the examples do not compile verbatim against the system headers. Rather than pin a
matching ck_tile commit, the validated kernel uses CK's classic
`DeviceGemmMultipleD_Xdl_CShuffle` -- which IS the XDL/MFMA tensor-core path on
gfx90a and exposes a STABLE fused multi-D epilogue (`CDEElementOp(e,c,d0,d1)`),
the most direct 1:1 mapping of the CUTLASS `GemmUniversal + PairwiseDistance
epilogue`. ck_tile is still preferred for greenfield work; for THIS port the
classic DeviceGemmMultipleD is the robust choice on the installed ROCm. The
ck_tile route remains open once the system headers and examples realign.

### What this pass delivered + GPU-validated (gfx90a, MI250X)

1. DISTANCE/NEIGHBORS BUILD unblocked on ROCm: the `#include <cutlass/...>` wall in
   the 5 CUTLASS files (pairwise_distance_cutlass_base, dispatch_sm80,
   fused_distance_nn/cutlass_base, fused_l2_nn, fused_distance_nn/{fused_l2_nn,
   fused_cosine_nn}) is `#if !defined(USE_HIP)`-guarded; the dispatchers force the
   legacy SIMT/sm60 path on HIP; the SIMT-fallback kernel bodies (`#if __CUDA_ARCH__
   < 800`) also build under USE_HIP. Distance headers + neighbors now compile under
   hipcc.
2. CK expanded-L2 (L2Expanded + L2SqrtExpanded): MFMA GEMM + fused `xn+yn-2*dot`
   epilogue. PASS vs CPU ref at M,N,K in {512x384x128, 1024x1024x256, 100x200x64,
   2048x512x128} (max_rel ~1.7e-7, 0 mismatches). Build: clang++ -x hip
   --offload-arch=gfx90a -I/opt/rocm/include -lamdhip64. CONSTRAINT: this tuned
   instance needs N % 4 == 0 (CShuffle CDE vector width 4) and N%4 / K%4 for the
   A/B vector loads; arbitrary N needs a scalar (vec=1) instance or padding -- same
   alignment dispatch raft does on the CUDA side. Also: CK DeviceGemmMultipleD only
   supports Row-major D tensors and a stride-0 Row D broadcasts the N-vector (yn),
   so the per-M norm (xn) is materialized [M,N] here; a single-vector fused variant
   would transpose the GEMM (compute D[n,m]).
3. faiss/select_warpsort wave64 fix (the neighbors top-k): `ballot()` now returns
   the 64-bit WarpMask (was truncating lanes 32-63 into uint32) + `popc_mask`/
   `ffs_mask`/`lane_bit` WarpMask helpers; select_warpsort's two `add()` loops use
   them; faiss MergeNetworkWarp single-warp bitonic sort adds the stride-32 merge on
   wave64. select_k MATRIX_SELECT_TEST (matrix/select_k.cu, the kWarp* + radix
   algorithms, k from 1 to 1700): 607 tests, 0 failures, 0 errors on gfx90a (567
   run + the rest gtest-SKIPPED as not-applicable combos), HIP_VISIBLE_DEVICES=1.
4. THE ROOT-CAUSE wave64 fix -- host/device WarpSize mismatch: raft::WarpSize and
   warp_size() were 64 only when __GFX9__ is defined, which is ONLY the device
   pass; the HOST pass fell back to 32. select_warpsort computes its launch geometry
   (block_dim, smem, per-warp data partition) on the HOST from WarpSize, so a 32 on
   the host disagreed with the 64-lane device kernel and FAULTED (small-len top-k
   coredumped; large-len silently mis-partitioned). Fix: the HIP build derives
   RAFT_HOST_WARP_SIZE from CMAKE_HIP_ARCHITECTURES (gfx9->64, gfx10/11/12->32) and
   cuda_dev_essentials.cuh / cudart_utils.hpp use it in the host pass so host==device.
   Confirmed: same top-k inputs that coredumped now return identical, correct
   results across immediate/filtered/distributed (probe agent_space/raft_select_probe).

### faiss warp-select (knn_brute_force / ball_cover / fused_l2_knn) -- still deferred

The FAISS BlockSelect/WarpSelect (Select.cuh / key_value_block_select.cuh) is more
deeply wave32-coupled than select_warpsort: `kNumWarpQRegisters = NumWarpQ /
WarpSize` goes to 0 when a kernel requests NumWarpQ=32 on a 64-lane wavefront
(zero-length register arrays; `warpMergeAnyRegisters` no-match), and it has
`__any_sync(0xffffffff,...)` 32-bit masks. Making it wave64-correct needs the
warp-queue capacity raised to >= WarpSize at every instantiation in fused_l2_knn /
ball_cover plus the mask widening -- a larger change than this pass. NEIGHBORS_TEST
(haversine/ball_cover/epsilon) therefore stays RAFT_TEST_NEIGHBORS=OFF; select_k
(MATRIX_SELECT) is the validated neighbors-top-k surface. Other gaps surfaced while
building NEIGHBORS_TEST and are part of that follow-up: ball_cover.cu uses
`CUDART_PI_F` (add to the compat math_constants.h) and fused_l2_knn-inl.cuh:431 has
a `long`->key_type narrowing in a `{}` init (clang-strict).

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
