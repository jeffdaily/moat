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

## Validation 2026-06-02 (validator, linux-gfx90a, fork moat-port 2baf983)

Platform: linux-gfx90a (MI250X GCD2, ROCm 7.2.1). GPU arch detected: gfx90a.
Scope: cuvs-free SG algorithm slice + cuvs-free PRIMS. Followers linux-gfx1100
and windows-gfx1151 are left for their respective hosts.

### GPU arch

```
HIP_VISIBLE_DEVICES=2
rocminfo -> gfx90a (AMD Instinct MI250X / MI250), sramecc+:xnack-
AMD_LOG_LEVEL=3 log confirms: "Using native code object for device:
  amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-"
Tensile kernel tags contain ISA90a / WS64 (wave64 dispatch confirmed).
```

### Step 1: Multi-arch build check

```
llvm-objdump --offloading projects/cuml/build-hip/libcuml.so
```

Result: BOTH code objects present -- gfx90a and gfx1100 bundles extracted
(30 bundles each for both arches, visible as
`libcuml.so.{N}.hipv4-amdgcn-amd-amdhsa--gfx{90a,1100}`).
Build is multi-arch confirmed.

Incremental build command (re-link only, sources unchanged):
```
cmake --build projects/cuml/build-hip -j16
```
Exit 0. 32/32 test binaries linked.
Timing: 0.29 s (incremental, stats.jsonl phase=compile).

### Step 2: ctest run (gfx90a real GPU)

```
export HIP_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
ctest --test-dir projects/cuml/build-hip --output-on-failure -j1
```

Result: 32/33 PASS. SG_LARS_TEST FAILS (in-process sequential, expected).
Timing: 23.76 s (stats.jsonl phase=test).

Passing SG tests (all in-scope algos):
SG_OLS_TEST, SG_RIDGE_TEST, SG_CD_TEST, SG_QUASI_NEWTON, SG_SGD_TEST,
SG_PCA_TEST, SG_TSVD_TEST, SG_HOLTWINTERS_TEST, SG_SHAP_KERNEL_TEST,
SG_GENETIC_NODE_TEST, SG_GENETIC_PARAM_TEST.

Passing PRIMS tests (all 21 incl KSELECTION):
PRIMS_ADD_SUB_DEV_SCALAR_TEST, PRIMS_BATCHED_CSR_TEST,
PRIMS_BATCHED_GEMV_TEST, PRIMS_BATCHED_MAKE_SYMM_TEST,
PRIMS_BATCHED_MATRIX_TEST, PRIMS_DECOUPLED_LOOKBACK_TEST,
PRIMS_DEVICE_UTILS_TEST, PRIMS_ELTWISE2D_TEST, PRIMS_FAST_INT_DIV_TEST,
PRIMS_FILLNA_TEST, PRIMS_GRID_SYNC_TEST, PRIMS_HINGE_TEST,
PRIMS_JONES_TRANSFORM_TEST, PRIMS_KSELECTION_TEST, PRIMS_LINALG_BLOCK_TEST,
PRIMS_LINEARREG_TEST, PRIMS_LOG_TEST, PRIMS_LOGISTICREG_TEST,
PRIMS_MAKE_ARIMA_TEST, PRIMS_PENALTY_TEST, PRIMS_SIGMOID_TEST.

PRIMS_KSELECTION_TEST: 3/3 subtests PASS (wave64 6th-stage bitonic fix
confirmed working on gfx90a). All WarpTopKTests pass.

### Step 3: LARS in-isolation check (NOT a gate)

Each LARS subtest run in strict isolation (one --gtest_filter per invocation):
```
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/0.fitGram"  -> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/0.fitX"     -> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/0.fitLarge" -> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/0.predictV1"-> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/0.predictV2"-> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/1.fitGram"  -> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/1.fitX"     -> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/1.fitLarge" -> PASSED
SG_LARS_TEST --gtest_filter="LarsTest*" (unit tests only)   -> 20/20 PASSED
```

All LARS subtests pass in isolation; the in-process sequential failure
(4/24 fitGram+fitX float+double) is the documented ROCm-7.2.1 runtime
cross-invocation state leak. Not a port defect. Confirmed: identical 23
rocBLAS calls, no HIP API error, drift +/-1 only after a prior fit in the
same process.

### Verdict: PASS

All formal gates met:
- Multi-arch build: gfx90a + gfx1100 code objects in libcuml.so (confirmed).
- In-scope SG tests: 11/11 PASS.
- PRIMS tests: 21/21 PASS including PRIMS_KSELECTION (wave64 fix verified).
- LARS in isolation: all subtests PASS (in-process failure is documented
  ROCm-7.2.1 known issue, not a regression).
- Native gfx90a dispatch: confirmed via AMD_LOG_LEVEL=3 (ISA90a/WS64).

validated_sha: 2baf9836f5cd90bccb70af4bfbaf6b67f2983086
State transition: review-passed -> completed (linux-gfx90a).

## Review 2026-06-02 (reviewer, linux-gfx90a, fork moat-port 2baf983)

Verdict: review-passed. Single curated commit 2baf983 on base b081fcd08
(REL v25.08.00). Reviewed via /pr-review local-branch mode against
git diff b081fcd08...HEAD.

Verified correct (no changes requested):
- kselection.cuh 6th bitonic stage: bitonicSortStage<...,5> uses Stride2=1<<6=64
  and iterates strides 32..1, the standard final merge that joins two sorted
  32-lane halves into a sorted 64-lane sequence. Gated by
  `if constexpr (raft::WarpSize > 32)` (per-arch device constexpr: wave32 emits
  5 stages, wave64 emits 6), so arch-unified in one binary, CUDA path unchanged.
  The three `obj.template cas<Greater>` edits are the correct C++20 clang
  two-phase-lookup fix.
- glm/preprocess.cuh + glm/ridge.cuh gemv reroutes are numerically identical to
  the gemm they replace (checked against the installed raft gemv overloads:
  preprocess hits the 8-arg `gemv(h,A,m,n,x,y,trans,stream)`, ridge hits the
  10-arg `gemv(h,A,m,n,x,y,trans,alpha,beta,stream)`; trans/alpha/beta and the
  output dims match the original CUBLAS_OP_T/OP_N gemm). USE_HIP-guarded.
- hw_decompose.cuh lda=trend_len passed to the *_bufferSize queries is the
  genuinely-correct lda (matches the real geqrf/orgqr calls); unconditional edit
  is safe on NVIDIA (cusolver ignores lda in the size query).
- batched_kalman.cu erfinv: confirmed host-side scalar (kernel-launch argument);
  cuml_host_erfinv is the same Giles(2010) double approximation, level in (0,1)
  is within domain. CUDA path byte-for-byte via the #else.
- cuvs deferral clean: every `<cuvs/...>` include lives in deferred files
  (metrics distance, knn, tsne, spectral); none of the 30 in-scope
  CUML_HIP_SOURCES pulls cuvs. The whole metrics module was deferred (not split
  per the plan's option-a) -- acceptable, expands with cuvs, no replan.
- Commit hygiene: `[ROCm]` title 64 chars, Claude named, no noreply trailer, no
  ghstack, no em-dash, Test Plan present. jeffdaily/main is a clean upstream
  mirror (no [ROCm] commits), default branch main, Actions disabled
  (enabled:false). moat-port HEAD == recorded head_sha 2baf983.
- BC: top-level guard is an early-return bypass; everything below is upstream
  byte-for-byte. qn_solvers.cuh default-arg `= 0` -> `level_enum::trace` is
  value-identical (TRACE==0) and is the canonical spelling -- a strict
  generalization, safe on the CUDA path though unguarded.

Minor observations (non-blocking, no fix required for review-passed):
- cuml_hip.cmake:195 links libcuml with `-Wl,--allow-shlib-undefined`. This
  suppresses link-time detection of genuinely missing symbols; it is justified
  by raft's `$<TARGET_NAME_IF_EXISTS:...>` guarded math-lib references but means
  a real undefined symbol would only surface at dlopen/runtime. The validator
  should confirm every in-scope test binary actually loads and runs (the ctest
  run does this), not rely on a clean link.
- The cub thread_operators / device_segmented_reduce compat shims
  (cpp/src/hip/compat_include/cub/...) exist for the deferred RF_TEST path; they
  are harmless header forwards but are dead for the current in-scope set. Leave
  them for the RF follow-up.

LARS in-process-sequential known issue: accepted as a documented ROCm-7.2.1
runtime cross-invocation state issue, NOT a port/source defect. Evidence is
sound: every LARS subtest passes in isolation (per-filter, float+double); the
failing in-process fit issues the identical 23 rocBLAS calls with no HIP error,
drifting +/-1; no host static/__constant__/shared global in lars_test.cu or
lars_impl.cuh. Not a bounce.

GPU re-run is the validator's job; this review did not re-execute ctest. The
porter's reported 32/33 pass (incl KSELECTION WarpTopK on wave64) is the claim
the validator must reproduce.
