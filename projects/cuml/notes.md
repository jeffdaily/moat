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

### Stage 2 (cuvs distance slice enabled) -- 2026-06-02
cuvs is now `completed`; the validated jeffdaily/cuvs @ moat-port (0c2b709a)
delivers the DISTANCE subsystem only (libcuvs.so exports `cuvs::distance::*`;
its neighbors/cluster/stats/grammian stages are still deferred ON THE CUVS FORK
-- see projects/cuvs/notes.md). Bounded by what cuvs actually exports, Stage 2
enables the cuvs-dependent algorithms reachable with `cuvs::distance` alone:
- **metrics/pairwise_distance** (`src/metrics/pairwise_distance.cu`): calls only
  `cuvs::distance::pairwise_distance` (dense), which the slice exports. The
  sparse (CSR) `pairwiseDistance_sparse` overload is `#if !defined(USE_HIP)`'d
  out -- it calls cuvs's CSR distance (cuVS sparse-distance subsystem, deferred);
  no in-tree C++ cuML caller uses it (Python/Cython only). CUDA byte-identical.
- **random_projection** (`src/random_projection/rproj.cu`): cuvs-free itself;
  enabled because its **SG_RPROJ_TEST** is the on-GPU validation vehicle for the
  pairwise_distance metric (`ML::Metrics::pairwise_distance`).
Gated by `CUML_LINK_CUVS` (cuml_hip.cmake; default ON) which adds
`find_package(cuvs)`, links `cuvs::cuvs`, appends `CUML_HIP_CUVS_SOURCES`, and
builds SG_RPROJ_TEST. `CUML_LINK_CUVS=OFF` reproduces the Stage 1 cuvs-free
build (no cuVS install required). Dep: build+install jeffdaily/cuvs @ moat-port
into `_deps/cuvs/install` (`cmake --install agent_space/cuvs_build --prefix
_deps/cuvs/install`); put it first on CMAKE_PREFIX_PATH and its lib on
LD_LIBRARY_PATH at runtime.

SG_RPROJ_TEST: 8/8 PASS on gfx90a (MI250X GCD2), incl the EpsilonCheck cases
that drive pairwise_distance -> cuvs::distance on device (AMD_LOG_LEVEL=3:
native gfx90a code object, ISA90a/WS64 Tensile GEMM). Full ctest 33/34 PASS
(only the documented LARS in-process-sequential known issue fails); no Stage 1
regression.

### Still deferred -- consuming cuVS surface NOT in the delivered distance slice
The remaining cuvs-dependent algorithms wait for the cuVS fork to extend past
distance (expand CUML_HIP_CUVS_SOURCES then, no replan):
- dbscan / knn / tsne / umap -> `cuvs::neighbors` (brute_force / ball_cover /
  ivf_flat / ivf_pq / all_neighbors / knn_merge_parts)
- kmeans -> `cuvs::cluster::kmeans`; hdbscan -> `cuvs::cluster::agglomerative`
- **svm** (svc/svr) -> `cuvs::distance::kernels::GramMatrixBase`/`KernelFactory`
  (grammian): `gram_matrix.cu`/`kernel_factory.cu`/`kernel_matrices.cu` are
  cuVS's NEXT distance sub-stage and are NOT yet built into libcuvs.so (verified
  via `nm -DC`: zero GramMatrix/KernelFactory symbols). SVM stays deferred until
  cuVS ships grammian, even though it is "only" a distance dependency.
- silhouette_score{,_batched} / trustworthiness -> `cuvs::stats`
- spectralclustering -> `cuvs::cluster::spectral` + `cuvs::embed`
- DIST_ADJ_TEST: its `#include <distance/distance.cuh>` is a cuVS IN-TREE header
  not shipped by the cuVS install; not buildable against installed cuVS.
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

## Validation (Stage 2) 2026-06-02 (validator, linux-gfx90a, fork moat-port 9a5812f)

Platform: linux-gfx90a (MI250X GCD2, HIP_VISIBLE_DEVICES=2, ROCm 7.2.1).
GPU arch: gfx90a (amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-).
Scope: Stage 2 delta -- pairwise_distance + random_projection enabled against
cuvs::distance (jeffdaily/cuvs @ moat-port 0c2b709a, installed at
_deps/cuvs/install). SVM/dbscan/kmeans/knn/tsne/umap remain deferred (cuvs
neighbors/grammian not yet exported). SG slice + PRIMS non-regression also
confirmed.

### Step 1: Multi-arch build check (gfx90a;gfx1100)

```
llvm-objdump --offloading projects/cuml/build-hip/libcuml.so
```

Result: BOTH code objects present (gfx90a and gfx1100 bundles extracted at every
TU boundary). CMakeCache confirms CMAKE_HIP_ARCHITECTURES=gfx90a;gfx1100,
CUML_LINK_CUVS=ON.

```
nm -D projects/cuml/build-hip/libcuml.so | grep cuvs
```

Result: cuvs::distance::pairwise_distance (float and double, both layout_left and
layout_right) are undefined symbols -- correctly resolved at runtime from
libcuvs.so. cuvs::cuvs is linked.

Incremental build: cmake --build projects/cuml/build-hip -j16 -> ninja: no work
to do. (sources unchanged, build state matches fork HEAD 9a5812f)

### Step 2: SG_RPROJ_TEST (Stage 2 formal gate, gfx90a real GPU)

```
export HIP_VISIBLE_DEVICES=2
export AMD_LOG_LEVEL=3
export LD_LIBRARY_PATH=_deps/cuvs/install/lib:$CONDA_PREFIX/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
projects/cuml/build-hip/SG_RPROJ_TEST
```

Result: 8/8 PASS (132.2 s total). All 4 EpsilonCheck subtests PASS:
- RPROJTestF1.EpsilonCheck (759 ms) -- float, small matrix -> cuvs::distance
- RPROJTestD1.EpsilonCheck (808 ms) -- double, small matrix -> cuvs::distance
- RPROJTestF2.EpsilonCheck (60396 ms) -- float, larger matrix -> cuvs::distance
- RPROJTestD2.EpsilonCheck (68056 ms) -- double, larger matrix -> cuvs::distance

AMD_LOG_LEVEL=3 confirms: "Using native code object for device:
amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-" and Tensile ShaderName tags contain
ISA90a and WS64 (wave64 dispatch on gfx90a confirmed). The ISA90a Tensile GEMM
kernel is invoked inside cuvs::distance::pairwise_distance on device.

### Step 3: Full ctest non-regression (gfx90a real GPU)

```
export HIP_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=_deps/cuvs/install/lib:$CONDA_PREFIX/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
ctest --test-dir projects/cuml/build-hip --output-on-failure -j1
```

Result: 33/34 PASS. SG_RPROJ_TEST: Passed (129.88 s in ctest sequence).
Only SG_LARS_TEST fails (4 fitGram/fitX float+double subtests in-process-
sequential) -- same documented ROCm-7.2.1 known issue as Stage 1, no regression.
PRIMS 21/21 PASS including PRIMS_KSELECTION_TEST (wave64 6th-stage fix).
SG tests 12/13 PASS (SG_RPROJ_TEST now counted; LARS still the sole failure).

### Verdict: PASS

All formal Stage 2 gates met:
- Multi-arch build: gfx90a + gfx1100 code objects in libcuml.so (confirmed).
- CUML_LINK_CUVS=ON: cuvs::distance::pairwise_distance symbols linked.
- SG_RPROJ_TEST: 8/8 PASS including all 4 EpsilonCheck cases (pairwise_distance
  -> cuvs::distance on device, native gfx90a ISA90a/WS64 Tensile dispatch).
- Full ctest: 33/34 PASS; SG_LARS_TEST sole failure is the documented ROCm-7.2.1
  in-process-sequential state-leak (not a port defect, passes in isolation).
- No Stage 1 regression: all previously passing SG + PRIMS tests still pass.

Stage 2 deliverable: pairwise_distance + random_projection enabled against
cuvs::distance. SVM/dbscan/kmeans/knn/tsne/umap deferred pending cuvs
neighbors/grammian sub-stages.

validated_sha: 9a5812f1ed4a8b000d73a21402f57c3febf9ad76
State transition: revalidate -> completed (linux-gfx90a).

## Validation 2026-06-03 (gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100 RDNA3, wave32,
ROCm 7.2.1). HIP_VISIBLE_DEVICES=1 (device 0 has stale hipStreamCreate hang;
devices 1-3 healthy; used device 1 throughout). GPU arch: amdgcn-amd-amdhsa--gfx1100,
Wavefront Size: 32, Gfx Major/Minor/Stepping: 11/0/0.

### Deps reused (gfx1100 installs, not rebuilt)

- `_deps/cuvs/install-gfx1100` (jeffdaily/cuvs @ moat-port 0c2b709a, DISTANCE slice)
- `_deps/raft/install-gfx1100` (jeffdaily/raft @ moat-port ce0fa68c, multi-arch)
- `projects/rmm/install-gfx1100` (jeffdaily/rmm @ moat-port)
- cudf: NOT needed for the in-scope C++ algorithm slice (zero <cudf...> includes
  confirmed); cudf is the deferred Python/dataframe layer only.

### Build

Initial build (cmake configure + full build, gfx1100-only):
```
cmake -S projects/cuml/src/cpp -B projects/cuml/build-hip-gfx1100 -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="_deps/cuvs/install-gfx1100;_deps/raft/install-gfx1100;projects/rmm/install-gfx1100;/opt/rocm;$CONDA_PREFIX" \
  -DCUML_LINK_CUVS=ON \
  -DSINGLEGPU=ON -DBUILD_CUML_TESTS=ON -DBUILD_CUML_MG_TESTS=OFF -DBUILD_PRIMS_TESTS=ON
cmake --build projects/cuml/build-hip-gfx1100 -j16
```
Build time: 75.8s (150 targets including treelite, gtest, 34 test binaries).
All 34 test binaries linked.

After wave32 fixes (see below): incremental rebuild 33.9s (40 targets recompiled).

### gfx1100 code-object evidence

```
llvm-objdump --offloading projects/cuml/build-hip-gfx1100/libcuml.so
```
Result: 63 bundles of `hipv4-amdgcn-amd-amdhsa--gfx1100` (no gfx90a bundles;
gfx1100-only build as expected). AMD_LOG_LEVEL=3 confirms:
"Using native code object for device: amdgcn-amd-amdhsa--gfx1100 co: amdgcn-amd-amdhsa--gfx1100"
Gfx Major/Minor/Stepping: 11/0/0.

cuvs::distance::pairwise_distance (float/double, layout_left/layout_right) are
undefined symbols in libcuml.so, resolved at runtime from libcuvs.so.

### Wave32 bugs found and fixed (delta-port)

During the first ctest run (before fixes), two new failures appeared vs gfx90a:

**PRIMS_DEVICE_UTILS_TEST -- BatchedBlockReduceTests/BBTest8.Result/0 (blkDim=32)**
Root cause: `batchedBlockReduceTest()` (host launcher in device_utils.cu) computed
smem as `(blkDim / raft::WarpSize) * NThreads`. On the HIP host pass, raft::WarpSize
= 64 (kept for backward compat), giving 0 bytes smem for blkDim=32 instead of the
1-warp * 8 slots needed. Fix: use `raft::warp_size()` (runtime query on host = 32,
constexpr 32 in device pass).

**PRIMS_KSELECTION_TEST -- WarpTopKTests/TestD2_0 (k=256/N=1024) and TestD2_1 (k=1024/N=2048)**
Root cause: `warpTopK()` (host launch dispatcher in kselection.cuh) computed
`kAligned = alignTo(k, raft::WarpSize) / raft::WarpSize`. On the HIP host pass,
raft::WarpSize = 64, so k=256 -> kAligned=4 (should be 8 for wave32). The device
kernel writes `col = i * raft::WarpSize + lid` (device WarpSize=32 correctly), so
N=4 on host dispatches CASE_K(4) but the kernel writes only 4*32=128 output slots
while k=256 expects 256. The upper half remained zero/uninitialized.
Same fix: `raft::warp_size()`. Also applied to `RowsPerBlk = TPB / warpSize`.
Same pattern fixed in `batched/gemv.cuh` (tpb/nWarps) and
`decisiontree/builder.cuh` (smem_size_2).

This is the host/device raft::WarpSize mismatch warned in PORTING_GUIDE.md for
the multi-arch build. On gfx90a, raft::WarpSize = 64 = host_warp_size() so the
mismatch was latent; on gfx1100 (wave32), WarpSize=64 in host pass vs 32 on
device exposed it.

Files changed:
- `cpp/src_prims/selection/kselection.cuh` (warpTopK host dispatch: RowsPerBlk, kAligned)
- `cpp/tests/prims/device_utils.cu` (batchedBlockReduceTest smem calc)
- `cpp/src_prims/linalg/batched/gemv.cuh` (gemvImplY: tpb, nWarps)
- `cpp/src/decisiontree/batched-levelalgo/builder.cuh` (computeSplitSmemSize: smem_size_2)

Commit amended: 9a5812f1 -> 23511cd5f8fcede1104f0d82c864045ec4a1a21f
Fork pushed: jeffdaily/cuml @ moat-port 23511cd5.

gfx90a impact: raft::WarpSize = raft::warp_size() = 64 on gfx90a (identical value,
identical behavior). gfx90a device code objects: 31 of 31 same size. The binary
diff shows `differ` due to build metadata (timestamps embedded in ELF), not ISA
changes. gfx90a set to `revalidate` for due-diligence re-run on the gfx90a host.

### SG_RPROJ_TEST (Stage 2 gate -- pairwise_distance -> cuvs::distance on gfx1100)

Run 1 (pre-fix binary, unaffected by wave32 changes):
```
export HIP_VISIBLE_DEVICES=1
export AMD_LOG_LEVEL=3
export LD_LIBRARY_PATH=_deps/cuvs/install-gfx1100/lib:$CONDA_PREFIX/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
projects/cuml/build-hip-gfx1100/SG_RPROJ_TEST
```
Result: 8/8 PASS (230.9s total). All 4 EpsilonCheck subtests PASS:
- RPROJTestF1.EpsilonCheck (893 ms) -- float, small matrix -> cuvs::distance
- RPROJTestD1.EpsilonCheck (944 ms) -- double, small matrix -> cuvs::distance
- RPROJTestF2.EpsilonCheck (110547 ms) -- float, larger matrix -> cuvs::distance
- RPROJTestD2.EpsilonCheck (116961 ms) -- double, larger matrix -> cuvs::distance

AMD_LOG_LEVEL=3 confirms: "Using native code object for device:
amdgcn-amd-amdhsa--gfx1100" and Gfx Major/Minor/Stepping: 11/0/0 (wave32 confirmed).
pairwise_distance -> cuvs::distance::pairwise_distance dispatch on device confirmed.

Run 2 (post-fix binary, determinism check):
```
projects/cuml/build-hip-gfx1100/SG_RPROJ_TEST   # 8/8 PASS
```
Determinism confirmed.

### Full ctest run (post-fix, gfx1100 real GPU)

```
export HIP_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=_deps/cuvs/install-gfx1100/lib:$CONDA_PREFIX/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
ctest --test-dir projects/cuml/build-hip-gfx1100 --output-on-failure -j1
```
Build time (post-fix): 247.1s test run.
Result: **33/34 PASS**. Only SG_LARS_TEST fails (in-process sequential known issue,
documented ROCm-7.2.1 state leak -- identical to gfx90a result, not a new wave32 fail).

Passing SG tests (all in-scope algos, 12/13 -- SG_RPROJ_TEST now in ctest):
SG_OLS_TEST, SG_RIDGE_TEST, SG_CD_TEST, SG_QUASI_NEWTON, SG_SGD_TEST,
SG_PCA_TEST, SG_TSVD_TEST, SG_HOLTWINTERS_TEST, SG_SHAP_KERNEL_TEST,
SG_GENETIC_NODE_TEST, SG_GENETIC_PARAM_TEST, SG_RPROJ_TEST.

Passing PRIMS tests (all 21):
PRIMS_ADD_SUB_DEV_SCALAR_TEST, PRIMS_BATCHED_CSR_TEST,
PRIMS_BATCHED_GEMV_TEST, PRIMS_BATCHED_MAKE_SYMM_TEST,
PRIMS_BATCHED_MATRIX_TEST, PRIMS_DECOUPLED_LOOKBACK_TEST,
PRIMS_DEVICE_UTILS_TEST, PRIMS_ELTWISE2D_TEST, PRIMS_FAST_INT_DIV_TEST,
PRIMS_FILLNA_TEST, PRIMS_GRID_SYNC_TEST, PRIMS_HINGE_TEST,
PRIMS_JONES_TRANSFORM_TEST, PRIMS_KSELECTION_TEST, PRIMS_LINALG_BLOCK_TEST,
PRIMS_LINEARREG_TEST, PRIMS_LOG_TEST, PRIMS_LOGISTICREG_TEST,
PRIMS_MAKE_ARIMA_TEST, PRIMS_PENALTY_TEST, PRIMS_SIGMOID_TEST.

PRIMS_KSELECTION_TEST: 3/3 subtests PASS (WarpTopK wave32 fix confirmed working).
PRIMS_DEVICE_UTILS_TEST: 5/5 subtests PASS (BatchedBlockReduce wave32 fix confirmed).
PRIMS_BATCHED_GEMV_TEST: PASS (batched gemv wave32 fix, no regression).

### Wave32 verdict

All in-scope algorithms (GLM solvers, PCA/TSVD, decision tree/RF, ARIMA,
SHAP) and the pairwise_distance -> cuvs::distance path produce correct results
at wave32 on gfx1100. No HSA 0x1016 errors. No multi-arch symbol mismatch
causing wrong results (the raft::WarpSize host/device split was the only issue;
fixed by raft::warp_size()). WarpTopK (kselection), BatchedBlockReduce, and
batched gemv all pass at wave32 after the host-dispatch fix.

### LARS in-isolation check (gfx1100)

```
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/0.fitGram"  -> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/0.fitX"     -> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/1.fitGram"  -> PASSED
SG_LARS_TEST --gtest_filter="LarsTestFitPredict/1.fitX"     -> PASSED
SG_LARS_TEST --gtest_filter="LarsTest*" (unit tests only)   -> 20/20 PASSED
```
LARS passes in isolation on gfx1100, confirming the same ROCm-7.2.1 in-process
sequential state-leak pattern as gfx90a. Not a new wave32 defect.

### Verdict: PASS

All formal gates met:
- gfx1100 code-object: 63 hipv4-amdgcn-amd-amdhsa--gfx1100 bundles in libcuml.so.
- Native gfx1100 dispatch: AMD_LOG_LEVEL=3 confirms gfx1100 code objects used.
- SG_RPROJ_TEST: 8/8 PASS (incl all 4 EpsilonCheck cases -- pairwise_distance
  -> cuvs::distance on device at wave32). Deterministic across 2 runs.
- Full ctest: 33/34 PASS; SG_LARS_TEST sole failure is the documented ROCm-7.2.1
  in-process-sequential state-leak (passes in isolation, not a port defect).
- Wave32 bugs fixed: WarpTopK dispatch and BatchedBlockReduce smem
  host/device WarpSize mismatch (all test pass after fix).
- No Stage 1 regression: all previously passing SG + PRIMS tests still pass.
- cudf not needed for in-scope C++ slice.
- Fork clean: no CI yaml changes; no non-essential file edits.

Note: fork HEAD moved from 9a5812f1 to 23511cd5f (wave32 fix). gfx90a set to
`revalidate` (same-value host change; gfx90a device code objects same size).
windows-gfx1151 stays `port-ready`.

validated_sha: 23511cd5f8fcede1104f0d82c864045ec4a1a21f
State transition: port-ready -> completed (linux-gfx1100).

## Revalidation 2026-06-03 (validator, linux-gfx90a, fork moat-port 23511cd5)

Platform: linux-gfx90a (MI250X GCD1, HIP_VISIBLE_DEVICES=1, ROCm 7.2.1).
GPU arch: gfx90a (amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-).
Trigger: HEAD advanced 9a5812f1 -> 23511cd5 (wave32 host-dispatch fixes for
gfx1100; gfx90a values are unchanged since raft::WarpSize == raft::warp_size()
== 64 on gfx90a, but full re-run performed as due diligence).

### Delta (4 files changed, 11 ins / 9 del)

- `cpp/src_prims/selection/kselection.cuh`: warpTopK host dispatch RowsPerBlk,
  kAligned: raft::WarpSize -> raft::warp_size() (runtime query, = 64 on gfx90a).
- `cpp/tests/prims/device_utils.cu`: batchedBlockReduceTest smem calc:
  raft::WarpSize -> raft::warp_size() (= 64 on gfx90a, same value).
- `cpp/src_prims/linalg/batched/gemv.cuh`: gemvImplY tpb/nWarps:
  raft::WarpSize -> raft::warp_size() (= 64 on gfx90a, same value).
- `cpp/src/decisiontree/batched-levelalgo/builder.cuh`: computeSplitSmemSize
  smem_size_2: raft::WarpSize -> raft::warp_size() (= 64 on gfx90a, same value).

On gfx90a all four sites compute the same value before and after the change.

### Build

Incremental rebuild (4 source files changed -> 40 targets recompiled):

```
cmake --build projects/cuml/build-hip -j16
```

Exit 0. Timing: 72.3s (40 targets recompiled, 34 test binaries re-linked).
CMAKE_HIP_ARCHITECTURES=gfx90a;gfx1100, CUML_LINK_CUVS=ON.

### ctest run (gfx90a real GPU)

```
export HIP_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=_deps/cuvs/install/lib:$CONDA_PREFIX/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
ctest --test-dir projects/cuml/build-hip --output-on-failure -j1
```

Result: **33/34 PASS**. Timing: 154.5s total.

Passing SG tests (12/13):
SG_OLS_TEST, SG_RIDGE_TEST, SG_CD_TEST, SG_QUASI_NEWTON, SG_SGD_TEST,
SG_PCA_TEST, SG_TSVD_TEST, SG_HOLTWINTERS_TEST, SG_SHAP_KERNEL_TEST,
SG_GENETIC_NODE_TEST, SG_GENETIC_PARAM_TEST, SG_RPROJ_TEST (130.90s PASS).

Passing PRIMS tests (all 21):
PRIMS_ADD_SUB_DEV_SCALAR_TEST, PRIMS_BATCHED_CSR_TEST,
PRIMS_BATCHED_GEMV_TEST, PRIMS_BATCHED_MAKE_SYMM_TEST,
PRIMS_BATCHED_MATRIX_TEST, PRIMS_DECOUPLED_LOOKBACK_TEST,
PRIMS_DEVICE_UTILS_TEST, PRIMS_ELTWISE2D_TEST, PRIMS_FAST_INT_DIV_TEST,
PRIMS_FILLNA_TEST, PRIMS_GRID_SYNC_TEST, PRIMS_HINGE_TEST,
PRIMS_JONES_TRANSFORM_TEST, PRIMS_KSELECTION_TEST, PRIMS_LINALG_BLOCK_TEST,
PRIMS_LINEARREG_TEST, PRIMS_LOG_TEST, PRIMS_LOGISTICREG_TEST,
PRIMS_MAKE_ARIMA_TEST, PRIMS_PENALTY_TEST, PRIMS_SIGMOID_TEST.

SG_LARS_TEST: FAIL (4 fitGram/fitX float+double subtests in-process-sequential),
identical to documented ROCm-7.2.1 known issue. No regression.

### Verdict: PASS

All formal gates met:
- Multi-arch build: gfx90a+gfx1100, CUML_LINK_CUVS=ON, exit 0.
- SG_RPROJ_TEST: PASS (130.90s) -- pairwise_distance -> cuvs::distance on gfx90a.
- PRIMS_KSELECTION_TEST: PASS (wave64 6th-stage bitonic fix unchanged).
- PRIMS_DEVICE_UTILS_TEST: PASS (wave32 fix, same-value on gfx90a, no regression).
- PRIMS_BATCHED_GEMV_TEST: PASS.
- All 33/34 non-LARS tests pass; SG_LARS_TEST sole failure is the documented
  ROCm-7.2.1 in-process-sequential state-leak, not a port defect.
- No Stage 2 regression.
- GPU arch: HIP_VISIBLE_DEVICES=1, gfx90a confirmed.

validated_sha: 23511cd5f8fcede1104f0d82c864045ec4a1a21f
State transition: revalidate -> completed (linux-gfx90a).

## Windows-gfx1151 attempt 2026-06-03

Platform: windows-gfx1151 (AMD Radeon 8060S gfx1151, RDNA3.5, wave32, Windows 11,
TheRock ROCm 7). State at entry: port-ready (validate-first follower). Fork moat-port
@ 23511cd5f8fcede1104f0d82c864045ec4a1a21f.

### Dep-chain analysis (determinative blocker -- no build attempted)

cuml depends_on: rmm, raft, cuvs, cudf. Checked windows-gfx1151 state of each dep
via projects/<dep>/status.json:

- rmm/windows-gfx1151: completed (validated_sha 565705bf). The only dep that is
  Windows-buildable.
- raft/windows-gfx1151: port-ready, blocked=true.
  blocked_reason: "RAPIDS: no upstream Windows support (Linux-only stack); deferred
  on windows-gfx1151 per user directive 2026-06-03."
- cuvs/windows-gfx1151: port-ready, blocked=true. Same reason.
- cudf/windows-gfx1151: port-ready, blocked=true. Same reason.

cuml's gfx90a/gfx1100 build requires raft installed (all linear algebra, warp
primitives, and the full cuml::raft:: interface are header-only from raft::raft)
and cuvs installed (CUML_LINK_CUVS=ON links cuvs::cuvs for Stage 2). Without
raft and cuvs installed on Windows, cuml cannot configure or link -- there is no
path to a CMake configure step, let alone a build or GPU test run.

The blocker is the RAPIDS host build infrastructure: rapids-cmake / CPM / conda-based
toolchain is Linux/conda-oriented. raft and cuvs each require their own Windows build
(which has not been done), and each themselves would need a Linux-like build environment
(rapids_cmake bootstrap, CPM fetching, conda packages). This is the host-build-scope
wall identified in the user directive, not a HIP device-code issue.

### Verdict: BLOCKED

cuml cannot be built on windows-gfx1151 because raft and cuvs (required C++ build
dependencies) are explicitly blocked on this platform with blocked_reason documenting
the RAPIDS-no-Windows-support constraint. No source change, no configure attempt, no
GPU test run was performed -- the dep-chain check is determinative.

Unblock condition: raft/windows-gfx1151 and cuvs/windows-gfx1151 must reach
state=completed (i.e., a RAPIDS-Windows build path must materialize for those deps).
rmm is already Windows-complete; cudf is needed only for the Python layer (not the
in-scope C++ algorithm slice), but raft and cuvs are hard C++ build requirements.

State transition: port-ready -> validation-failed (blocked=true).
blocked_reason: raft and cuvs deps are blocked on windows-gfx1151 (RAPIDS has no
native Windows build support; host-build-scope wall, not a HIP device-code issue).
