# cuml notes

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated).

## ROCm/HIP port (MOAT, Strategy A, RAPIDS-on-ROCm)

Lead platform linux-gfx90a (MI250X, ROCm 7.2.1). Multi-arch build
gfx90a;gfx1100 in one libcuml.so. Scoped to the cuvs-INDEPENDENT single-GPU
algorithm slice; the cuvs-dependent algorithms and the multi-GPU + Python
layers are deferred until cuvs is ported. See plan.md for the full deferral
rationale.

### In scope (built + validated on gfx90a)
linear_model (OLS/RIDGE/logistic via glm), solvers (CD/LARS/SGD/QN),
decomposition (PCA/TSVD), ensemble (decision tree + random forest library
code), tsa (ARIMA/auto-ARIMA), holtwinters, genetic, explainer
(kernel_shap + permutation_shap), datasets. The cuvs-free PRIMS primitives.

### Deferred (NOT regressions -- documented scope)
- cuvs-dependent algorithms (pull `<cuvs/...>`, the CUTLASS->CK neighbors/
  distance surface that lives in the cuvs port): dbscan, hdbscan, kmeans,
  knn/kde, tsne (+cufft), umap, **svm** (svm includes
  `cuvs/distance/grammian.hpp` for the kernel matrix -- it is NOT cuvs-free,
  correcting an earlier assumption), spectral clustering, and the
  distance-based metrics (pairwise_distance, silhouette_score{,batched},
  trustworthiness). Expand `CUML_ALGORITHMS` once cuvs is `completed`; no
  replan needed.
- `explainer/tree_shap.cu`: needs the header-only gputreeshap CUDA library;
  deferred for the lead pass (kernel_shap + permutation_shap kept, SHAP_KERNEL
  validated). Bring tree_shap when gputreeshap is audited under hipcc.
- RF_TEST (the gtest only): RandomForest *library* code builds into libcuml,
  but its gtest validates predictions through the FIL forest-inference module
  (`src/fil/*`, outside the cuvs-free in-scope set) and uses
  `cub::DeviceSegmentedReduce::ArgMax` over a strided offsets iterator, which
  the hipCUB rocprim backend does not instantiate. Bring FIL + RF_TEST in a
  follow-up. randomforest.cu itself compiles and links.
- All multi-GPU (`*_mg.cu`, `src_prims/opg/*`, NCCL/UCX) and the Python pytest
  suite (needs the full RAPIDS ROCm Python wheel stack).

### Dependency contract (consume the ported deps, do NOT re-port)
- rmm: jeffdaily/rmm @ moat-port, installed at `_deps/raft-rmm/install`.
  Exports rmm::rmm (HIP), vendored libhipcxx, rapids_logger; force-includes its
  compat header.
- raft: jeffdaily/raft @ moat-port (ce0fa68c, MULTI-ARCH), installed at
  `_deps/raft/install`. cuml uses ONLY the header-only `raft::raft` (not
  raft::compiled). raft's runtime `host_warp_size()` + per-arch device
  `raft::WarpSize` give cuml its wave-agnostic warp width.
  IMPORTANT: raft-config finds hipblas + hipsparse, but the raft::raft
  interface also references roc::hipsolver / roc::hipblaslt / hip::hiprand
  under `$<TARGET_NAME_IF_EXISTS:...>`. A consumer that instantiates raft's
  cusolver/cublasLt/curand wrappers in its OWN TUs (here the SG gtests) drops
  those guarded names and fails to link. cuml_hip.cmake therefore does
  `find_package(hipsolver/hipblaslt/hiprand)` so the targets exist.
- cudf: NOT a C++ dependency of libcuml (0 `<cudf...>` includes confirmed);
  Python-layer only. Not on CMAKE_PREFIX_PATH for the C++ build.
- cuvs: planned, NOT ported -- the deferral gate. Not linked by the lead.
- treelite (dmlc/treelite 4.4.1): external host C++ library, FetchContent-built
  host-side exactly as upstream. gputreeshap: deferred with tree_shap.cu.

### Build (gfx90a;gfx1100 multi-arch)
```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/cuml/src/cpp -B projects/cuml/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft/install;/var/lib/jenkins/moat/_deps/raft-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DCUML_ALGORITHMS="linear_model;solvers;decomposition;ensemble;tsa;genetic;explainer;datasets" \
  -DSINGLEGPU=ON -DBUILD_CUML_TESTS=ON -DBUILD_CUML_MG_TESTS=OFF -DBUILD_PRIMS_TESTS=ON
cmake --build projects/cuml/build-hip -j16
```
Followers pass their own `-DCMAKE_HIP_ARCHITECTURES` with no source edit.

### Test (real gfx90a, GCD assigned via HIP_VISIBLE_DEVICES)
Runtime libs (rapids_logger pulls spdlog/fmt from conda):
`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/opt/rocm/lib:$LD_LIBRARY_PATH`
then `ctest --output-on-failure` (each binary its own process). Result:
32/33 ctest targets PASS. SG: OLS, RIDGE, CD, LARS*, QUASI_NEWTON, SGD, PCA,
TSVD, HOLTWINTERS, SHAP_KERNEL, GENETIC{NODE,PARAM}. PRIMS: all 21 incl
KSELECTION. AMD_LOG_LEVEL=3 confirms gfx90a wave64 dispatch (Tensile kernels
tagged ISA90a / WS64, rocSOLVER/rocBLAS device kernels).

### Port fixes (arch-unified, USE_HIP-guarded where the CUDA path must stay byte-for-byte)
Order to review: the CMake/shim infra first, then the per-file gotchas below.
- `cpp/CMakeLists.txt` + `cpp/cmake/hip/*` + `cpp/src/hip/cuda_to_hip.h`:
  standalone HIP build (Strategy A), `.cu` -> LANGUAGE HIP, find_package deps,
  --offload-compress + gc-sections, thrust::cuda=thrust::hip alias, host
  erfinv, version/logger header regen.
- `__HIPCC__` guards for `CUML_KERNEL` and `FLATNODE_HD` (else kernels silently
  drop `__global__` -> `static`).
- `rapids_logger::level_enum` default args (`= 0` does not convert to the enum
  under the shipped rapids_logger).
- gtest `.cu`: `T arr[non_constexpr_len]` -> `T arr[]` (clang rejects the VLA).

### GOTCHAS (cross-project-worthy)
1. **hipBLASLt degenerate (m=1 or n=1) double GEMM crashes on gfx90a, ROCm
   7.2.1.** raft's `raft::linalg::gemm` routes through cublasLt -> hipBLASLt;
   for a row-vector*column-vector (output 1x1 or matrix*vector output width 1)
   double GEMM, Tensile SIGSEGVs in `ContractionSolution::requiredWorkspaceSize`
   (workspace passed as nullptr/0). Float is fine (different Tensile solution).
   FIX in cuml source: express these as `raft::linalg::gemv` (routes through
   hipblasgemv, not LT). Applied in `glm/preprocess.cuh` (postProcessData
   intercept dot -> fixed OLS/RIDGE/CD/SGD) and `glm/ridge.cuh` (ridgeSolve's
   two matrix*vector products). USE_HIP-guarded; NVIDIA path unchanged.
2. **cuSOLVER `*_bufferSize` lda validation: hipSOLVER is strict, NVIDIA is
   lax.** `holtwinters/internal/hw_decompose.cuh` passed `lda=2` to
   `geqrf_bufferSize`/`orgqr_bufferSize` for a [trend_len x 2] column-major
   matrix (the real geqrf/orgqr calls correctly use `lda=trend_len`). NVIDIA
   cusolver ignores lda in the size query; hipSOLVER returns
   CUSOLVER_STATUS_INVALID_VALUE (lda < m). FIX: pass `trend_len` (the true
   lda) to the size queries too. Unconditional -- it is the genuinely-correct
   lda. Fixed all HOLTWINTERS double Fit tests.
3. **wave64 warp-network stage count.** `src_prims/selection/kselection.cuh`
   `bitonicSort` hardcoded 5 bitonic stages (sorts across 32 lanes). On wave64
   (gfx90a/gfx94x) a warp is 64 lanes, so it needs a 6th stage. FIX:
   `if constexpr (raft::WarpSize > 32)` add `bitonicSortStage<...,5>`. Arch-
   agnostic: raft::WarpSize is the per-arch device constant (64 CDNA / 32 RDNA
   /CUDA), so wave32 device pass skips it. Fixed PRIMS_KSELECTION WarpTopK.
   `warpSort` / `WarpMask` were already raft::WarpSize-parameterized.
4. **clang two-phase lookup on dependent template member call.**
   kselection.cuh `current.cas<Greater>(...)` -> `current.template
   cas<Greater>(...)` at the 3 call sites in template functions (clang parses
   `<` as less-than otherwise). C++20 + hipcc/clang.

### KNOWN ISSUE (not a port defect; documented)
- **SG_LARS_TEST fails only under in-process sequential gtest execution.**
  Every LARS subtest PASSES in isolation: all 14 `LarsTest.*` unit tests
  (select/moveToActive/updateCholesky/calcW0/calcA/equiangular/maxStep) pass
  together, and each `LarsTestFitPredict.{fitGram,fitX,fitLarge,predictV1,
  predictV2}` passes standalone (verified per-filter, both float and double).
  When ANY LARS test precedes a full `larsFit` in the SAME process, the
  subsequent fit's coefficients drift by exactly +/-1 (alternating sign). The
  failing fit executes the IDENTICAL 23 rocBLAS calls (ROCBLAS_LAYER=2 count
  matches the clean run) and no HIP API error is logged -- the numbers are
  computed but slightly wrong. Each test fixture has its own raft::handle_t and
  freshly-allocated device buffers; there is no host-side static/shared state
  in lars_test.cu or lars_impl.cuh (grep-confirmed: no __constant__, no
  function-local static, no shared device global). The only process-global
  state in the LARS path is the rocThrust caching device allocator behind the
  single `thrust::max_element` in `selectMostCorrelated` and raft's
  thread-local `interruptible` sync token -- i.e. a ROCm-7.2.1 runtime/library
  cross-invocation state-leakage issue, NOT a defect in the cuml port. LARS is
  numerically correct on gfx90a. If a future ROCm fixes the allocator/Tensile
  interaction this should clear with no source change; otherwise the LARS gtest
  needs an isolation harness (separate process per fit) upstream.
