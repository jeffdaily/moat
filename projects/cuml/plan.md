# cuml -- ROCm/HIP port plan (lead: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: cuml
- Upstream: https://github.com/rapidsai/cuml (default branch `main`)
- Clone HEAD at planning: cf6a54692b78bee4dc8c348194d5454af0a14c75 (VERSION 26.08.00)
- RAPIDS machine-learning library: C++ `libcuml` (algorithm implementations + a thin C/C++ ML API) and a Python layer (`libcuml`, `cuml`). scikit-learn-style estimators on GPU.
- MOAT deps (status.json depends_on): rmm, raft, cudf, cuvs. rmm/raft/cudf are consumable (rmm/cudf `completed`, raft moat-port fork GPU-validated and installed at `_deps/raft/install`). cuvs is `planned`, NOT yet ported -- it is the central deferral boundary of this plan.

## Existing AMD support
None. Upstream cuml is CUDA-only: `project(CUML ... LANGUAGES CXX CUDA)` (cpp/CMakeLists.txt:18-23), fetches NVIDIA CCCL/rmm/raft/cuvs via rapids-cmake CPM, no HIP/ROCm path, no USE_HIP, no OpenCL/Vulkan/SYCL alternative. No abandoned ROCm fork or PR found.
Decision: PROCEED with a fresh CUDA->HIP port (Strategy A), the same RAPIDS-on-ROCm pattern already proven for rmm/raft/cudf/cugraph in MOAT, scoped to the cuvs-INDEPENDENT algorithm slice for the lead deliverable and deferring the cuvs-dependent algorithms until cuvs lands.
A mechanical correctness-first port is appropriate: ZERO `cutlass`/`cute::` anywhere in cpp/ (grep-confirmed across src, src_prims, include) -- there is NO port-vs-AMD-native-rewrite question in cuml itself. The CUTLASS/CuTe-tuned distance/neighbors kernels that cuml would otherwise need live in cuVS+raft-NN, which is exactly why those algorithms are deferred to the cuvs port (per memory: cuvs needs the raft pairwise-distance/fused_distance_nn CUTLASS->CK reimplementation). cuml's own hand-written kernels (trees, SVM SMO, metrics, qn/glm) ride raft's wave-agnostic `raft::WarpSize`/`raft::shfl`/`raft::laneId` helpers, already validated by raft on both wave64 (gfx90a) and wave32 (gfx1100).

## Build classification: CMake (Strategy A)
Evidence:
- `cpp/CMakeLists.txt:8` `cmake_minimum_required(VERSION 4.0)`, `:18-23` `project(CUML VERSION ... LANGUAGES CXX CUDA)` -- a standalone CMake build with `.cu` sources, NOT a pytorch extension.
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension` anywhere in cpp/. The Python layer (`python/libcuml/pyproject.toml`, `python/cuml/pyproject.toml`) uses `rapids-build-backend` + `scikit-build-core` (not torch); the two `torch` hits in `python/cuml/cuml/internals/{array,input_utils}.py` are optional runtime dlpack/array interop, not a build dependency.
- Surface: 94 `.cu`, 129 `.cuh`, 2 `.cpp` under cpp/src + cpp/src_prims; C++/CUDA standard C++20 (`CXX_STANDARD 20`, `CUDA_STANDARD 20`), same as cugraph -- rocThrust/rocPRIM/hipCUB are fine at C++20.
=> Strategy A: `option(USE_HIP)` + early standalone HIP CMake bypass + `.cu` marked `LANGUAGE HIP` + a minimal compat shim. ext_type = `cmake`.

## Port strategy (Strategy A, RAPIDS-on-ROCm variant)
Apply the rmm/raft/cudf/cugraph RAPIDS recipe (PORTING_GUIDE "RAPIDS-on-ROCm" pattern; mirror projects/cugraph/plan.md):
1. Top of `cpp/CMakeLists.txt`: `option(USE_HIP "Build with HIP" OFF)`; `if(USE_HIP) include(cmake/hip/cuml_hip.cmake) return() endif()` placed BEFORE the rapids-cmake bootstrap (`include(../cmake/rapids_config.cmake)` at line 10 is fetched by rapids-cmake at configure time and is unavailable standalone). The NVIDIA path below the `return()` stays byte-for-byte unchanged.
2. `cpp/cmake/hip/cuml_hip.cmake` (NEW): standalone HIP build that
   - `enable_language(HIP)`; arch from `${CMAKE_HIP_ARCHITECTURES}` defaulted to `gfx90a` only when unset (never a literal -- followers pass their own arch with no source edit; PORTING_GUIDE arch-drift rule);
   - resolves deps by `find_package(rmm)`, `find_package(raft)` (`raft::raft` header-only -- cuml does NOT use `raft::compiled`; see Dependency contract) -- NOT CPM-fetch -- from the installed `_deps/*/install` prefixes on `CMAKE_PREFIX_PATH`;
   - satisfies CCCL by include paths: rocThrust/hipCUB from `/opt/rocm/include`, libhipcxx `cuda::std`/`cuda::mr` transitively via `rmm::rmm`'s exported include dirs + `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE`;
   - regenerates the header rapids-cmake would write: `include/cuml/version_config.hpp` (the CMakeLists writes it via `rapids_cmake_write_version_file` at line 30; replicate with configure_file);
   - builds treelite and gputreeshap host-side (see Dependency contract) only for the tree-model algos that need them;
   - drives the IN-SCOPE source list off the upstream `CUML_ALGORITHMS` algorithm-selection machinery (see Dependency contract) rather than hand-listing -- set `CUML_ALGORITHMS` to the cuvs-free slice so `ConfigureAlgorithms.cmake` leaves `LINK_CUVS=OFF` and never pulls cuvs;
   - marks all in-scope `.cu` `LANGUAGE HIP` (override `add_library`/`add_executable` at top scope per the MPPI-Generic lesson, since `cpp/tests/CMakeLists.txt` builds many targets);
   - force-includes the compat shim on HIP TUs and adds `--offload-compress` to the HIP compile options FROM THE START (cudf/cugraph lesson: a thrust/cub/rocPRIM-heavy RAPIDS library risks the +/-2 GiB x86-64 relocation wall) plus `-ffunction-sections -fdata-sections` + `--gc-sections`.
3. Compat shim (`cpp/src/.../cuda_to_hip.h`, minimal): cuml uses essentially no raw toolkit symbols in the cuvs-free slice except device-side curand in ONE file (kernel_shap, explainer) -- see Risk list. Most `cuda*` runtime aliasing arrives via rmm's/raft's force-included compat headers transitively. Add a `namespace cub = hipcub;` alias if any direct `cub::` use remains (cuml mostly uses raft's prims). Keep the shim minimal.

NVIDIA build remains the default (`USE_HIP=OFF`) and is untouched.

## Dependency contract + cuvs deferral boundary (the key handoff for the porter)
cuml has a built-in algorithm-selection knob that IS the deferral lever. `cpp/cmake/modules/ConfigureAlgorithms.cmake` maps each algorithm (or sklearn-style group) to link flags. `LINK_CUVS` is turned ON by exactly: `dbscan, hdbscan, kmeans, knn, metrics, tsne, umap` (ConfigureAlgorithms.cmake:108-116). `CUML_USE_RAFT_NN` is turned ON by `knn` (set transitively by hdbscan/tsne/umap). Each algorithm also gates its OWN `target_sources` block (cpp/CMakeLists.txt:334-496) and its OWN test (cpp/tests/CMakeLists.txt:99-191), so selecting a cuvs-free `CUML_ALGORITHMS` list compiles only the in-scope sources AND only the in-scope tests with no per-file editing.

cuvs API surface actually consumed (grep of `#include <cuvs/...>`): `cuvs::distance` (pairwise_distance, grammian/kernel matrix), `cuvs::neighbors` (brute_force, ball_cover, ivf_flat, ivf_pq, epsilon_neighborhood, knn_merge_parts, all_neighbors), `cuvs::cluster` (kmeans, agglomerative, spectral), `cuvs::stats` (silhouette_score, trustworthiness_score), `cuvs::preprocessing::spectral_embedding`, `cuvs::distance::kde`. Every one of these is a neighbors/distance primitive -- precisely the CUTLASS-tuned surface deferred to the cuvs port.

CUVS-DEPENDENT (DEFER until cuvs is `completed`): dbscan (cuvs/neighbors/distance), hdbscan (+hierarchicalclustering), kmeans, knn (+kde), tsne (also LINK_CUFFT), umap, the distance-based metrics (pairwise_distance, silhouette_score{,_batched_float,_double}, trustworthiness), spectralclustering (cuvs/cluster/spectral + cuvs/preprocessing/spectral_embedding), AND **svm** -- CORRECTION to the task's assumption: SVM is NOT cuvs-free. `svm/{svc_impl,svr_impl,kernelcache,sparse_util,smosolver}` all `#include <cuvs/distance/distance.hpp>` + `<cuvs/distance/grammian.hpp>` for the kernel matrix (RBF/poly/tanh Gram). SVM is therefore in the deferred set.

CUVS-INDEPENDENT (lead deliverable -- zero `cuvs/` includes, grep-confirmed):
- Linear models + solvers: glm (linearregression/ridge/lasso/logisticregression -> `src/glm/glm.cu`), solvers (lars/cd/sgd/qn -> `src/solver/{lars,solver}.cu`). cuSOLVER/cuSPARSE used here route through raft's `raft::linalg::detail::cusolver_wrappers`/`raft::sparse::detail::cusparse_wrappers` (already ported to hipSOLVER/hipSPARSE in raft), NOT direct toolkit calls.
- Decomposition: pca (`src/pca/pca.cu`), tsvd (`src/tsvd/tsvd.cu`) -- raft SVD/eig (lstsqEig/lstsqSvdQR etc.).
- Trees/ensembles: decisiontree (`src/decisiontree/...` 13 .cu), randomforest (`src/randomforest/randomforest.cu`). Need treelite (host) + see nvforest risk.
- Time series: arima/autoarima (`src/arima/*`, `src/tsa/*`), holtwinters (`src/holtwinters/holtwinters.cu`). cuBLAS via raft handle.
- genetic (`src/genetic/*`), explainer/treeshap (`src/explainer/{kernel_shap,permutation_shap,tree_shap}.cu`; needs gputreeshap header-only + device curand), datasets (`src/datasets/*`).
- Cluster-comparison metrics that are cuvs-free (split from the distance metrics): accuracy_score, adjusted_rand_index, completeness_score, entropy, homogeneity_score, kl_divergence, mutual_info_score, r2_score, rand_index, v_measure. NOTE the upstream metrics_algo source block is monolithic and pulls in pairwise_distance/silhouette/trustworthiness too AND sets LINK_CUVS, so to ship the cuvs-free metrics the porter must either (a) split the metrics `target_sources` block in the HIP CMake to exclude the 5 cuvs files, or (b) defer the whole metrics module. Prefer (a) for coverage; it is a HIP-CMake-only edit.

Recommended lead `CUML_ALGORITHMS` (cuvs-free): `linear_model;solvers;decomposition;ensemble;tsa;genetic;explainer;datasets` (+ the cuvs-free metrics subset via the split above). This leaves LINK_CUVS=OFF and LINK_CUFFT=OFF (cufft is tsne-only).

Consume the ported deps per DEPENDENCIES.md (do NOT re-port them):
- rmm: `jeffdaily/rmm @ moat-port`, installed at `_deps/raft-rmm/install` (present). Exports `rmm::rmm` (HIP), vendored libhipcxx, rapids_logger, the `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE` def, force-includes its compat header. notes.md "## Install as a dependency" present.
- raft: `jeffdaily/raft @ moat-port`, installed at `_deps/raft/install` (present; libraft.so + cmake). Exports `raft::raft` (header-only -- cuml needs ONLY this, not `raft::compiled`). Modules cuml uses are ALL in raft's delivered set: `raft/core/handle`, `raft/linalg/*` (lstsq/eig/svd/gemv + cusolver/cusparse wrappers), `raft/util/{cuda_utils,device_atomics,cudart_utils}` (WarpSize/shfl/laneId), `raft/core/nvtx` (gated, off by default), `raft/random/*`, `raft/stats/*`, `raft/matrix/*`. notes.md "## Install as a dependency" present.
- cudf: NOT a C++ dependency of libcuml (0 `<cudf...>` includes in cpp/src+src_prims+include; confirm at porter time). cudf is a PYTHON-layer dep only. notes.md "## Install as a dependency" present; documented scoped-out symbols (6: 2 binary_operation JIT + 4 scan_inclusive) are irrelevant to the cuml C++ slice.
- cuvs: `planned`, NOT ported. Its notes.md has NO "## Install as a dependency" section yet (expected -- not built). This is the gate for the deferred algorithm set; the lead deliverable does not link it.
- treelite (dmlc/treelite): an EXTERNAL host C++ library (model serialization, OpenMP only, NO CUDA). NOT a MOAT project. The HIP CMake CPM-fetches/builds it host-side exactly as upstream does (get_treelite.cmake). Portable as-is. Used by decisiontree, randomforest, explainer(treeshap).
- gputreeshap (rapidsai/gputreeshap): header-only CUDA library (treeshap kernels), CPM-fetched. Its device headers must compile under hipcc when treeshap_algo/explainer is in scope -- a small hipify-on-include risk (audit at porter time; if it does not port cleanly, defer tree_shap.cu only, keeping kernel_shap/permutation_shap).

NONE of rmm/raft/cudf is missing the "## Install as a dependency" section. cuVS is the only dep without it, and it is the deferral gate.

ACTION for the porter before building: confirm `_deps/raft/install` and `_deps/raft-rmm/install` exist (rebuild from the moat-port forks per their notes if stale), and set `-DCMAKE_PREFIX_PATH="<raft>/install;<rmm>/install;/opt/rocm;$CONDA_PREFIX"`.

## CUDA surface inventory (cuvs-free slice)
Build target topology (cpp/CMakeLists.txt): `cuml_objs` (OBJECT) -> `cuml` (SHARED) [+ `cuml_static` when not dynamic-only]; MG sources (`src/*/\*_mg.cu`, `src_prims/opg/*`) gated by `NOT SINGLEGPU` (out of single-GPU scope -- needs NCCL/UCX/MPI). Tests: `cpp/tests/CMakeLists.txt` SG `ConfigureTest` (the validatable slice) + PRIMS tests + MG tests (NCCL/MPI, out of scope).

GPU mechanisms (grep-confirmed across cpp/src + cpp/src_prims):
- `__global__` kernels: present in trees, qn/glm, svm, metrics, genetic, explainer; most numeric work is expressed via raft prims + thrust/cub.
- Warp intrinsics: very small surface, and the cuvs-free files ride raft's wave-agnostic abstraction. `raft::WarpSize`/`raft::shfl`/`raft::laneId`/`raft::shfl_xor` in `decisiontree/batched-levelalgo/{split,builder}.cuh`, `src_prims/common/device_utils.cuh`, `src_prims/selection/kselection.cuh`, `src_prims/linalg/batched/gemv.cuh` -- all parameterized on raft's warp size. The RAW intrinsics (`__shfl`, `__all_sync`, `__activemask`, hardcoded `WARPSIZE`) live ONLY in tsne (`tsne/cannylab/bh.cu`, `tsne/barnes_hut_kernels.cuh`) and umap (`umap/simpl_set_embed/optimize_batch_kernel.cuh`) -- BOTH cuvs-dependent and DEFERRED, so the wave64 raw-intrinsic risk is outside the lead slice.
- cub/hipCUB: via raft prims and a few direct uses; `namespace cub = hipcub` shim if any remain.
- thrust: pervasive (rocThrust drop-in). DIRECT `thrust::cuda::par` execution-policy uses in cuvs-free files: `arima/{arima_common,batched_arima,batched_kalman}.cu`, `tsa/auto_arima.cuh`, `solver/lars_impl.cuh` (and svm/tsne which are deferred) -- must become `thrust::hip::par` under USE_HIP (rmm lesson). `rmm::exec_policy` already handles its own.
- Math libraries: cuBLAS (20 files) and cuSOLVER (6) and cuSPARSE (1) are all reached through the raft handle / raft `*_wrappers` (already hipBLAS/hipSOLVER/hipSPARSE in raft) -- NO direct toolkit handle creation in the cuvs-free slice. cuFFT (tsne only -- deferred). cuRAND: device-side `<curand_kernel.h>` (`curandStatePhilox4_32_10_t`, `curand_init`, `curand_uniform`) in `explainer/kernel_shap.cu` (IN SCOPE) and host curand in `umap/simpl_set_embed/algo.cuh` (deferred). The kernel_shap device-RNG must map to hipRAND device API (`hiprandStatePhilox4_32_10_t`, `hiprand_init`, `hiprand_uniform`, `<hiprand/hiprand_kernel.h>`) -- a real, small porting item.
- nvtx: 17 files, ALL via `raft/core/nvtx.hpp` (raft handles it; `NVTX` option defaults OFF -> no-op).
- Cooperative groups: ONE file, `dbscan/adjgraph/algo.cuh` (dbscan is DEFERRED) -- not in the lead slice.
- CUTLASS/CuTe: ZERO across all of cpp/.
- Textures/surfaces, managed memory: none in algos (managed referenced only in `ml_cuda_utils.h` helper). rmm owns allocation.
- Streams/events: via rmm `cuda_stream_view` and raft handle.

## Risk list
1. cuvs deferral is the dominant scoping lever (not a bug). Build the cuvs-free `CUML_ALGORITHMS` slice; verify `ConfigureAlgorithms.cmake` leaves `LINK_CUVS=OFF`/`LINK_CUFFT=OFF`/`CUML_USE_RAFT_NN=OFF` for the chosen list. The single biggest correctness boundary: never let a cuvs `#include` reach the HIP compiler.
2. SVM is cuvs-dependent (CORRECTION to the dispatch assumption). `svm/*` includes `cuvs/distance/grammian.hpp` for the kernel matrix; defer SVC_TEST/SVR with the cuvs set. (If a future porter wants SVM sooner, the Gram matrix could be computed via raft `pairwise_distance`/`gemm` directly, but that is a non-trivial rewrite -- defer with cuvs.)
3. metrics module is split. `metrics_algo` source block is monolithic and pulls 5 cuvs files (pairwise_distance, silhouette_score{,_batched_float,_double}, trustworthiness) and sets LINK_CUVS. To ship the 10 cuvs-free comparison metrics, split the `target_sources` block in the HIP CMake (exclude the 5 cuvs files); also drop TRUSTWORTHINESS_TEST (cuvs). HIP-CMake-only edit, no source change.
4. device-side cuRAND in kernel_shap.cu (in scope). Map `<curand_kernel.h>` Philox device RNG to hipRAND device API in the compat shim or via `#if defined(USE_HIP)`. Small, isolated.
5. thrust execution-policy namespace (rmm lesson). `thrust::cuda::par` -> `thrust::hip::par` (and `<thrust/system/cuda/execution_policy.h>` -> `.../hip/...`) under USE_HIP in arima/tsa/solver files. Mechanical.
6. nvforest link for randomforest. `randomforest_algo` sets `LINK_NVFOREST=ON` and links `nvforest::nvforest++` (rapidsai/nvforest, a CUDA FIL/forest-inference engine), but NO `#include` of nvforest was found in cpp/src (FIL was externalized; the link may be inference-only). Audit at porter time: if `libcuml` builds + RF_TEST passes WITHOUT nvforest (set `LINK_NVFOREST=OFF` in the HIP path), keep randomforest in scope; if RF genuinely needs nvforest device code, defer randomforest (and RF_TEST) since nvforest is an unported external CUDA repo. treelite + gputreeshap are needed regardless (host + header-only).
7. gputreeshap header-only CUDA must compile under hipcc (treeshap/explainer). If it does not port cleanly, defer ONLY tree_shap.cu (keep kernel_shap/permutation_shap and SHAP_KERNEL_TEST). Audit at porter time.
8. Build-scale relocation wall (+/-2 GiB x86-64). cuml is RAPIDS-scale (94 .cu, heavy raft/thrust/cub template instantiation, C++20). Apply `--offload-compress` FROM THE START (cudf/cugraph lesson: ~55x device-fatbin shrink), single target arch, `-ffunction-sections -fdata-sections` + `--gc-sections`. The cuvs-free slice is materially smaller than full cuml (no neighbors/distance), so this should fit one `libcuml.so` comfortably.
9. wave64 vs wave32. Lead gfx90a is wave64. The cuvs-free slice rides raft's `raft::WarpSize`/`shfl`/`laneId`/`shfl_xor` (raft-validated on both widths). One audit point: `svm/linear.cu:129` `static_assert(BX <= 32, "BX must be not larger than warpSize")` -- but svm is deferred, so moot for the lead. No raw warp intrinsics in the cuvs-free slice. The deferred tsne/umap raw-intrinsic kernels are the real wave64 work and travel with the cuvs deferral.
10. hipCUB DeviceRadixSort nonzero begin_bit (cudaKDTree lesson) and device-functor-returning-reference (cudf lesson): latent, surface only at validation. Apply the documented localization probes if a sort/reduction returns wrong values.
11. C++20 + clang two-phase lookup / __CUDA_ARCH__ guard rewrites (cudaKDTree/MPPI lessons): expect a handful of `this->`/class-qualified-static/`typename T::template` fixes and `__CUDA_ARCH__` -> `(__CUDA_ARCH__||__HIP_DEVICE_COMPILE__)` rewrites in the heavily-templated trees/qn/genetic code. Mechanical, surfaced at compile.
12. raft lead state is `revalidate` (head_sha advanced past validated_sha), NOT `completed`. The moat-port fork IS GPU-validated and installed/consumable, so it does not block the cuml PORT, but flag to the orchestrator that raft should be re-validated. Soft state, not a missing deliverable.

## File-by-file change list (lead)
- `cpp/CMakeLists.txt`: add `option(USE_HIP)` + early `include(cmake/hip/cuml_hip.cmake); return()` before the rapids bootstrap. NVIDIA path untouched.
- `cpp/cmake/hip/cuml_hip.cmake` (NEW): standalone HIP build (deps via find_package from _deps prefixes; CCCL via include paths; version_config.hpp regen; treelite + gputreeshap host/header builds for in-scope tree algos; `.cu` -> LANGUAGE HIP via top-scope add_library/add_executable override; force-include compat header; --offload-compress + gc-sections; sets the cuvs-free `CUML_ALGORITHMS` and confirms LINK_CUVS=OFF; splits the metrics source block to the cuvs-free subset).
- `cpp/cmake/modules/ConfigureAlgorithms.cmake`: reuse as-is (it already produces the right link flags for a cuvs-free algorithm list); no edit needed beyond passing the right `CUML_ALGORITHMS`.
- Compat shim header (NEW, small): device cuRAND->hipRAND aliases for kernel_shap, optional `namespace cub = hipcub;`, any cuda* symbol not covered transitively by rmm/raft's force-included compat header. Force-included on HIP TUs only.
- Per-file source fixes (USE_HIP-guarded, CUDA path byte-for-byte): `thrust::cuda::par` -> `thrust::hip::par` in arima/tsa/solver (Risk 5); device cuRAND in `explainer/kernel_shap.cu` (Risk 4); a handful of clang/two-phase + __CUDA_ARCH__ guard rewrites (Risk 11); the nvforest/gputreeshap audits (Risks 6,7).
- `cpp/tests/CMakeLists.txt`: tests are already gated by the same `_algo` flags, so the cuvs-free `CUML_ALGORITHMS` automatically builds only in-scope tests. Prefer NOT editing it; if the metrics split needs it, scope TRUSTWORTHINESS_TEST out on the HIP path.

## Build commands (gfx90a lead)
```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
# deps installed first (per their notes "Install as a dependency"):
#   _deps/raft/install, _deps/raft-rmm/install  (rebuild from jeffdaily/{raft,rmm}@moat-port if stale)
cmake -S projects/cuml/src/cpp -B projects/cuml/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft/install;/var/lib/jenkins/moat/_deps/raft-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DCUML_ALGORITHMS="linear_model;solvers;decomposition;ensemble;tsa;genetic;explainer;datasets" \
  -DSINGLEGPU=ON \
  -DBUILD_CUML_TESTS=ON \
  -DBUILD_CUML_MG_TESTS=OFF \
  -DBUILD_PRIMS_TESTS=ON
cmake --build projects/cuml/build-hip -j16
```
(`-DCMAKE_HIP_ARCHITECTURES=gfx90a` is the only arch knob; a follower passes gfx1100/gfx1151 with no source change. `CUML_ALGORITHMS` is the cuvs deferral lever -- add cuvs-gated algos only after cuvs is `completed`. A CPU-only docker compile check (rocm/dev-ubuntu-24.04:7.2.4-complete) is allowed as a manual gate, never the validation gate.)

## Test plan (REAL GPU gate on gfx90a; the validation deliverable)
The SG C++ gtests for the cuvs-free algorithms are the correctness slice (they build small datasets and compare against sklearn/CPU references and analytic expectations). Each is auto-gated by its `_algo` flag, so the cuvs-free `CUML_ALGORITHMS` builds exactly this set. Run serially on a single assigned GCD (`ctest`, not `-jN`; set `HIP_VISIBLE_DEVICES`; SG_SVC_TEST is RUN_SERIAL but svm is deferred anyway):
- Linear models / solvers: OLS_TEST, RIDGE_TEST, CD_TEST, LARS_TEST, QUASI_NEWTON, SGD_TEST
- Decomposition: PCA_TEST, TSVD_TEST
- Trees / ensembles: RF_TEST (pending nvforest audit, Risk 6)
- Time series: HOLTWINTERS_TEST (ARIMA has no dedicated SG gtest; covered by MAKE_ARIMA_TEST in PRIMS + python)
- Explainer: SHAP_KERNEL_TEST (pending gputreeshap audit, Risk 7)
- Genetic: GENETIC_NODE_TEST, GENETIC_PARAM_TEST
- Metrics (cuvs-free subset, after the metrics source split): the comparison-metric gtests if present (rand_index/entropy/etc.); TRUSTWORTHINESS_TEST is cuvs-deferred.
- PRIMS tests (cuvs-free, validate the underlying primitives): LINEARREG_TEST, LOGISTICREG_TEST, PENALTY_TEST, HINGE_TEST, SIGMOID_TEST, LOG_TEST, MAKE_ARIMA_TEST, FILLNA_TEST, JONES_TRANSFORM_TEST, BATCHED_CSR_TEST, BATCHED_GEMV_TEST, BATCHED_MAKE_SYMM_TEST, BATCHED_MATRIX_TEST, LINALG_BLOCK_TEST, DEVICE_UTILS_TEST, KSELECTION_TEST, ELTWISE2D_TEST, FAST_INT_DIV_TEST, ADD_SUB_DEV_SCALAR_TEST, DECOUPLED_LOOKBACK_TEST, GRID_SYNC_TEST. (Exclude the cuvs-dependent prims tests KNN_CLASSIFY_TEST/KNN_REGRESSION_TEST/DIST_ADJ_TEST/distance_base.)
Validation bar: each in-scope gtest builds and passes on gfx90a; correctness is the gtests' own reference comparisons (coefficients vs analytic/sklearn, PCA/SVD reconstruction, RF accuracy, SHAP additivity, genetic program eval); run-to-run determinism where the algorithm is deterministic.
Deferred (documented, NOT regressions): all cuvs-dependent algorithm tests -- DBSCAN_TEST, HDBSCAN_TEST, KMEANS (mg), KNN_TEST, TSNE_TEST, UMAP_PARAMETRIZABLE_TEST, TRUSTWORTHINESS_TEST, SVC_TEST/SVR, the spectral-clustering and distance-based metrics tests; all MG tests (NCCL/MPI multi-GPU); the Python pytest suite (needs the full RAPIDS ROCm Python wheel stack -- cudf/pylibraft/dask -- which MOAT does not deliver). RF_TEST and SHAP_KERNEL_TEST are in-scope pending the nvforest/gputreeshap audits; if those externals do not port, defer those two specifically.
Non-GPU regression set: the NVIDIA build path must remain byte-for-byte (USE_HIP=OFF compiles unchanged); there is no separate CPU-only cuml test suite to regress (it is a GPU library; the CPU pieces are treelite/genetic host code exercised by the same gtests).

## Disposition
PROCEED -- fresh ROCm/HIP Strategy-A port, lead deliverable scoped to the cuvs-INDEPENDENT algorithm slice (linear models, solvers, PCA/TSVD, decision trees / random forest, time series, genetic, explainer, cuvs-free metrics) with the SG + cuvs-free PRIMS C++ gtests green on gfx90a. DEFER the cuvs-dependent algorithms (dbscan, hdbscan, kmeans, knn/kde, tsne, umap, svm, spectral clustering, distance-based metrics) and the multi-GPU + Python layers until cuvs is `completed`; expand `CUML_ALGORITHMS` then with no replan. Lead platform linux-gfx90a advanced to `planned`.

## Open questions
- Does randomforest build + RF_TEST pass with `LINK_NVFOREST=OFF` (nvforest not `#include`d in cpp/src), or does FIL inference still pull nvforest device code? Resolve at porter time; if it needs nvforest, defer randomforest until nvforest is ported.
- Does rapidsai/gputreeshap (header-only CUDA) compile cleanly under hipcc for tree_shap.cu, or must tree_shap.cu be deferred (keeping kernel_shap/permutation_shap)?
- Does the monolithic metrics `target_sources` split cleanly into a cuvs-free subset, or is it simpler to defer the whole metrics module for the first lead pass and add the comparison metrics in a follow-up?
- Confirm 0 `<cudf...>` includes in cpp at porter time (cudf is a python-layer dep) so cudf need not be on CMAKE_PREFIX_PATH for the C++ build.
