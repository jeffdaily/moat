# cugraph -- ROCm/HIP port plan (lead: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: cugraph
- Upstream: https://github.com/rapidsai/cugraph (default branch `main`)
- Clone HEAD at planning: e314f1e077fe84762a12807639f39985bb4fd564 (VERSION 26.08.00)
- RAPIDS graph-analytics library: C++ `libcugraph` (+ `cugraph_c` C-API) and a Python layer (`pylibcugraph`, `cugraph`).
- MOAT deps (status.json): rmm, raft, cudf -- all ported on gfx90a (rmm/cudf `completed`, raft `revalidate` but moat-port fork validated; see Dependency contract).

## Existing AMD support
None. Upstream cugraph is CUDA-only: `LANGUAGES C CXX CUDA`, fetches NVIDIA CCCL/rmm/cuco/raft/cuvs via rapids-cmake CPM, no HIP/ROCm path, no USE_HIP, no OpenCL/Vulkan/SYCL alternative. No abandoned ROCm fork or PR found.
Decision: PROCEED with a fresh CUDA->HIP port (Strategy A), the same RAPIDS-on-ROCm pattern already proven for rmm/raft/cudf in MOAT. A mechanical correctness-first port is appropriate: graph primitives are thrust/cub/cuco-bound, NOT CUTLASS/CuTe/MMA-tuned (confirmed: zero `cutlass`/`cute::` in the tree), so there is no port-vs-AMD-native-rewrite question. The library's own warp primitives already route through raft's wave-agnostic `raft::warp_size()` / `raft::warp_full_mask()` abstraction, which raft validated on both gfx90a (wave64) and gfx1100 (wave32).

## Build classification: CMake (Strategy A)
Evidence:
- `cpp/CMakeLists.txt:8` `cmake_minimum_required(VERSION 4.0)`, `:20` `project(CUGRAPH ... LANGUAGES C CXX CUDA)` -- a standalone CMake build with `.cu` sources, NOT a pytorch extension.
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension` anywhere. The Python layer (`python/*/pyproject.toml`) uses `scikit-build-core` + `rapids-build-backend` (not torch), and only wraps the C++ lib.
- 327 `.cu`, 165 `.cuh`, 47 `.cpp` under `cpp/`; C++/CUDA standard is C++20 (`CXX_STANDARD 20`, `CUDA_STANDARD 20`) -- note this is newer than raft/cudf (C++17); rocThrust/rocPRIM/hipCUB are fine at C++20.
=> Strategy A: `option(USE_HIP)` + standalone HIP CMake bypass + `.cu` marked `LANGUAGE HIP` + one compat shim. ext_type = `cmake`.

## Port strategy (Strategy A, RAPIDS-on-ROCm variant)
Apply the rmm/raft/cudf RAPIDS recipe (PORTING_GUIDE changelog 2026-05-30 "RAPIDS-on-ROCm, the rapids-cmake/CPM CCCL redirect"):
1. Top of `cpp/CMakeLists.txt`: `option(USE_HIP "Build with HIP" OFF)`; `if(USE_HIP) include(cmake/hip/cugraph_hip.cmake) return() endif()` placed BEFORE the rapids-cmake bootstrap (`include(../cmake/rapids_config.cmake)` at line 10 is itself fetched by rapids-cmake at configure time and does not exist statically). The NVIDIA path below the `return()` stays byte-for-byte unchanged.
2. `cpp/cmake/hip/cugraph_hip.cmake` (new): standalone HIP build that
   - `enable_language(HIP)`; arch from `${CMAKE_HIP_ARCHITECTURES}` defaulted to `gfx90a` only when unset (never a literal -- followers pass their own arch with no source edit; PORTING_GUIDE arch-drift rule);
   - resolves deps by `find_package(rmm)`, `find_package(raft)` (`raft::raft` + `raft::compiled`), `find_package(cuco)` -- NOT CPM-fetch -- from the installed `_deps/*/install` prefixes on `CMAKE_PREFIX_PATH`. rmm/raft already carry HIP-built configs (their notes "Install as a dependency" contracts). cuco comes from the cuCollections fork source tree (header-only, `-I`, as cudf consumes it: `-DCUDF_CUCO_SOURCE_DIR` analog);
   - satisfies CCCL by include paths: rocThrust/hipCUB from `/opt/rocm/include`, libhipcxx `cuda::std`/`cuda::mr` vendored (comes transitively via `rmm::rmm`'s exported include dirs + `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE` def);
   - regenerates the headers rapids-cmake would write: `include/cugraph/version_config.hpp` and `include/cugraph_c/version_config.hpp` (the CMakeLists already writes these via `rapids_cmake_write_version_file`; replicate with a small configure_file);
   - marks all in-scope `.cu` `LANGUAGE HIP` (or override `add_library`/`add_executable` at top scope per the MPPI-Generic lesson, since `cpp/tests/CMakeLists.txt` globs/lists `.cu` into many targets);
   - force-includes the compat shim on HIP TUs and adds `--offload-compress` to the HIP compile options (cudf lesson: a thrust/cub/rocPRIM-heavy 540+MB-class library risks the +/-2 GiB x86-64 relocation wall; cugraph instantiates the whole prim machinery over v32_e32 and v64_e64 -- 327 .cu -- so the uncompressed fatbin reach is the first build risk).
3. Compat shim (`cpp/src/.../cuda_to_hip.h` or reuse rmm/raft's force-included compat header transitively): cugraph uses essentially no raw toolkit symbols (no direct cublas/cusparse/cusolver/curand/cufft, no nvtx, no cuda_fp16 -- all confirmed zero). It needs the `cuda*`->`hip*` runtime aliasing (which it already gets via rmm's/raft's force-included `cuda_to_hip.h` on every HIP TU that includes their headers) plus a `namespace cub = hipcub;` alias for the direct `cub::` uses (cudaKDTree lesson). Keep the shim minimal.

NVIDIA build remains the default (`USE_HIP=OFF`) and is untouched.

## Dependency contract (the key handoff for the porter)
cugraph `status.json depends_on = [rmm, raft, cudf]`. C++/Python split discovered during analysis:
- C++ `libcugraph` HARD deps: rmm, raft (`raft::raft` header-only + `raft::compiled`), cuco (cuCollections), and cuvs (ONE file only -- see Risk list). cudf is NOT a C++ dependency of libcugraph: 0 `<cudf...>` includes in `cpp/src`+`cpp/include`; cudf appears only as a README link. The CMake links `raft::raft`, `raft::compiled`, `cuco::cuco`, `rmm::rmm`, `cuvs::cuvs` (cpp/CMakeLists.txt:555-565, 613-623, 740) -- no cudf target.
- cudf is a PYTHON-layer dep (pylibcugraph/cugraph pyproject require `cudf==26.8.*`, `pylibraft`, `dask-cudf`, `pylibcudf`) consumed as GPU DataFrames at runtime, not by the C++ build.

Consume the ported deps per DEPENDENCIES.md (do NOT re-port them):
- rmm: `git clone -b moat-port https://github.com/jeffdaily/rmm _deps/rmm/src`; build+install to `_deps/rmm/install` per projects/rmm/notes.md "## Install as a dependency" (verified present, line 421). Exports `rmm::rmm` (HIP), vendored libhipcxx, rapids_logger, the `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE` def, and force-includes its compat header.
- raft: clone `jeffdaily/raft @ moat-port`; build+install to `_deps/raft/install` per projects/raft/notes.md "## Install as a dependency" (verified present, line 612). Exports `raft::raft` + `raft::compiled` (libraft.so), needs BOTH raft and rmm prefixes on `CMAKE_PREFIX_PATH`, links hipblas/hipsparse/hipsolver. raft's lead state is `revalidate` (head_sha advanced past validated_sha) NOT `completed`, but the moat-port fork is GPU-validated (DISTANCE/FUSED_NN/MATRIX_SELECT/LINALG/CORE/LABEL/RANDOM green on gfx90a) and consumable (standalone consumer verified, notes line 679). The raft modules cugraph actually uses are all in raft's DELIVERED set: `raft/core/handle` (144 uses), `raft/random/*` (rng/rng_state, 28), `raft/util/*` (cudart_utils/integer_utils/device_atomics/cuda_utils, ~36), `raft/core/device_span`/`host_span` (57), `raft/comms` (5), `raft/lap` (2). The DEFERRED raft modules (sparse/spectral/solver) are touched by only 3 legacy files (see Risk list), which are deferred.
- cudf: NOT needed for the C++ GPU-validatable slice. Clone+build only if the porter later attempts the Python layer (out of scope for lead validation). projects/cudf/notes.md "## Install as a dependency" exists (verified, line 979). cudf carries documented scoped-out deferred-dispatch symbols (2 binary_operation JIT + 4 scan_inclusive); irrelevant to cugraph C++.
- cuCollections (cuco): MOAT delivered PARTIAL (`jeffdaily/cuCollections @ moat-port`, `ported`); cudf already consumes it. cugraph's cuco usage (8 files) is `static_map`/`static_set`/`insert_and_find`/`find`/`murmurhash3_32`/`linear_probing<1>` with key = vertex type (int32 or int64, ALL >=4-byte) and arithmetic values -- exactly inside the delivered slice (>=4-byte keys; sub-word keys and >8-byte CAS are the deferred parts and cugraph does not use them). Point cugraph's HIP CMake at the cuco fork source dir (header-only).

NONE of the three depended-on notes are missing the "## Install as a dependency" section. cuCollections is not in `depends_on` (it is a transitive dep via raft/cudf in upstream); the porter must still clone+supply the cuco fork source dir.

ACTION for the porter before building: confirm `_deps/rmm/install` and `_deps/raft/install` exist (rebuild from the moat-port forks if stale per their notes), and set `-DCMAKE_PREFIX_PATH="_deps/raft/install;_deps/rmm/install;/opt/rocm;$CONDA_PREFIX"` plus the cuco source dir.

## CUDA surface inventory
Build target topology (cpp/CMakeLists.txt):
- `cugraph_common` (SHARED), `cugraph` (SG), `cugraph_mg` (multi-GPU, NCCL), `cugraph_mtmg` (multi-thread/multi-GPU), `cugraph_c` (C-API over the above), `cugraph_etl` (libcugraph_etl/, depends on cudf). MG/MTMG/ETL are out of the single-GPU validation slice (NCCL/UCX/cudf).
- Tests: `cpp/tests/CMakeLists.txt`, 63 `ConfigureTest` (SG, `GPUS 1`, no NCCL) + several `ConfigureTestMG` (NCCL+MPI). SG gtests are the validatable slice.

Graph primitives (the "prims" template engine in `cpp/include/cugraph/prims/`) implement traversal (BFS/SSSP/extract_bfs_paths/od_shortest_distances), centrality (PageRank/Katz/eigenvector/betweenness/HITS), community (Louvain/Leiden/ECG/k-truss/triangle-count/egonet), components (WCC/SCC), cores (core_number/k_core), link-prediction (jaccard/sorensen/overlap/cosine), link-analysis, sampling (neighbor/negative/random-walks), generators (rmat). Each is instantiated over vertex/edge type widths as `*_sg_v32_e32.cu` and `*_sg_v64_e64.cu` (these v32/v64 suffixes are 32-/64-bit VERTEX/EDGE INDEX TYPES, NOT wavefront width).

GPU mechanisms (grep-confirmed across cpp/src + cpp/include):
- `__global__` kernels: 14 files (most compute is expressed via thrust/cub transforms inside the prims, not raw kernels).
- Warp intrinsics: only `__shfl_sync` (4), `__ballot_sync` (3), `__activemask` (1), `__all_sync` (1) across 6 files (5 prims headers + 1 legacy `layout/legacy/bh_kernels.cuh`). The prims sites pair the shuffle with `raft::warp_full_mask()` and size lanes via `raft::warp_size()` (e.g. `sample_and_compute_local_nbr_indices.cuh`), i.e. they ride raft's wave-agnostic abstraction.
- cub: `cub::DeviceSegmentedSort` (44 -- the dominant primitive; -> hipCUB `DeviceSegmentedSort`/rocPRIM segmented radix sort), `cub::DeviceReduce` (12), `cub::BlockReduce` (7), `cub::BlockScan` (4), `cub::WarpScan<edge_t, raft::warp_size()>` (3), `cub::DeviceSelect` (3), `cub::WarpReduce` (1). All map to hipCUB; `namespace cub = hipcub` shim.
- cuCollections: `static_map`/`static_set`/`insert_and_find`/`find`/`murmurhash3_32`/`linear_probing<1>` (8 files; keys >=4-byte) -- inside the delivered cuco slice.
- thrust: pervasive (rocThrust drop-in; watch exec-policy namespace `thrust::cuda::par` -> `thrust::hip::par`, rmm lesson).
- Cooperative groups: ZERO uses.
- Math libraries direct: ZERO curand/cusparse/cublas/cusolver/cufft, ZERO nvtx, ZERO cuda_fp16/__half. (All library math is reached through raft.)
- CUTLASS/CuTe: ZERO.
- Streams/events: via rmm `cuda_stream_view` and raft handle; `CUDA_API_PER_THREAD_DEFAULT_STREAM` is a public compile def (PTDS) -- keep as-is on HIP (HIP supports per-thread default stream).
- Textures/surfaces, pinned/managed memory: none directly (rmm owns allocation).

## Risk list
1. cuVS hard-link, but used by ONE file. `cuvs::cuvs` is linked PRIVATE into `cugraph`/`cugraph_mg`/`cugraph_common` (cpp/CMakeLists.txt:561,619), yet the only translation unit that includes cuVS is `cpp/src/community/legacy/spectral_clustering.cu` (`<cuvs/cluster/spectral.hpp>`, `<cuvs/preprocessing/spectral_embedding.hpp>`). cuVS is MOAT-blocked (needs deferred raft neighbors/distance + CUTLASS->CK reimplementation per memory). DECISION: defer spectral clustering on HIP. Drop `spectral_clustering.cu` (and the `cuvs::cuvs` link) from the in-scope source list in the HIP CMake; scope out `c_api/legacy_spectral.cpp`, `tests/community/balanced_edge_test.cpp` (BALANCED_TEST), and `tests/c_api/legacy_spectral_test.c`. Same legacy file also pulls deferred raft `raft/spectral/*` + `raft/sparse/convert` + `raft/solver`, so it is the natural deferral boundary. This is the single biggest scope-down lever and removes the cuVS dependency entirely.
2. Deferred raft modules in two more legacy files. `cpp/src/tree/legacy/mst.cu` includes `raft/sparse/solver/mst.cuh` (deferred raft sparse/solver) -> defer MST (scope out, plus MST_TEST and c_api/legacy_mst.cpp). `cpp/src/linear_assignment/legacy/hungarian.cu` includes `raft/solver/linear_assignment` + `raft/lap/lap_kernels`/`lap_functions`; raft/lap IS in raft's delivered set but `raft/solver/linear_assignment` routing needs confirmation -- if it pulls a deferred raft path, defer HUNGARIAN_TEST too (low-value legacy algo). Confirm at porter time by attempting the include against the installed raft.
3. wave64 vs wave32. Lead gfx90a is wave64. cugraph rides raft's `raft::warp_size()`/`warp_full_mask()`/`WarpSize` abstraction, which raft validated host+device on both wave widths (RAFT_HOST_WARP_SIZE derived from CMAKE_HIP_ARCHITECTURES; 64-bit ballot masks). The `cub::WarpScan<edge_t, raft::warp_size()>` instantiations are parameterized on the runtime-correct warp size. RISK is the one legacy hand-rolled warp file `layout/legacy/bh_kernels.cuh` (Barnes-Hut FA2) -- audit its `__shfl`/`__ballot` for hardcoded 32 / 32-bit masks (ROCm 7.x static_asserts a 64-bit mask on `__shfl_sync`/`__ballot_sync`; AutoDock-GPU/popsift lessons). FA2 is `layout/legacy/force_atlas2.cu` (LEGACY_FA2_TEST) -- treat as a wave64 audit point, not necessarily a defer.
4. Build-scale relocation wall (+/-2 GiB x86-64). cugraph is the largest RAPIDS surface yet (327 .cu, full prim template matrix x {v32_e32, v64_e64}, thrust/cub/rocPRIM-heavy). Apply `--offload-compress` to the HIP compile options FROM THE START (cudf lesson: 55x device-fatbin shrink), build a single target arch (no multi-arch fatbin), and add `-ffunction-sections -fdata-sections` + `--gc-sections`. The library splits into cugraph_common + cugraph + (deferred mg/mtmg), which also helps keep any single .so under 2 GiB.
5. rocPRIM 4.2.0 GFX10/11 DPP bug -- FOLLOWERS ONLY (gfx1100/gfx1151), not the lead. cugraph's heavy `cub::DeviceSegmentedSort` (44) + `BlockScan`/`WarpScan` lower to rocPRIM `block_radix_sort`/`warp_exchange`/`lookback_scan_state`, which on GFX10+ emit wavefront-shift DPP (`DPP_WF_SL1 0x130`, `DPP_WF_RL1 0x134`, `ROW_SL1 0x101`) that the GFX10+ backend rejects ("wavefront shifts are not supported on GFX10+") -- the exact wall raft hit on gfx1100 (raft notes 779/962: MATRIX_SELECT_TEST and BALL_COVER_TEST excluded on gfx1100). `ROCPRIM_DISABLE_DPP=1` is only a partial mitigation (does not cover `lookback_scan_state`). On gfx90a (CDNA) this does NOT occur. RECORD for the follower delta-plan: cub-segmented-sort-bearing cugraph TUs may fail to COMPILE on gfx1100/gfx1151 until upstream rocPRIM fixes it; expect a per-arch build exclusion, not a source change.
6. hipCUB DeviceRadixSort nonzero `begin_bit` correctness (cudaKDTree lesson). If any cugraph sort selects a bit sub-range, it mis-sorts on gfx90a; prefer full-width sort. Audit the `DeviceSegmentedSort` call sites for `begin_bit`/`end_bit` (likely all full-width; confirm).
7. rocThrust exec-policy namespace (rmm lesson): any `thrust::cuda::par`/`<thrust/system/cuda/execution_policy.h>` must become `thrust::hip::par`/`<thrust/system/hip/execution_policy.h>` under USE_HIP. rmm's `exec_policy` (which cugraph uses via `rmm::exec_policy`) already handles this; audit any direct cugraph use.
8. Device functor returning a reference to a by-value/forwarded param (cudf lesson): rocThrust/rocPRIM read it as garbage where CUB tolerates it. cugraph's prims are functor-heavy (reduce_op, property_op_utils). If a reduction returns wrong/identity values, apply the cudf 3-way localization probe. Latent risk, surfaces only at validation.
9. C++20 + clang two-phase lookup / std-in-device traps (cudaKDTree, gsplat, MPPI lessons): the prims use heavy templates + dependent bases; expect `this->`/`typename T::template`/class-qualified-static fixes and `__CUDA_ARCH__`->`(__CUDA_ARCH__||__HIP_DEVICE_COMPILE__)` guard rewrites. Mechanical, surfaced at compile.
10. raft lead state `revalidate`. The selector gates cugraph on deps `completed`; raft is `revalidate`. The moat-port fork IS validated and consumable, so this does not block the PORT, but flag to the orchestrator that raft should be re-validated (its head_sha advanced) -- it is a soft state, not a missing deliverable.

## File-by-file change list (lead)
- `cpp/CMakeLists.txt`: add `option(USE_HIP)` + early `include(cmake/hip/cugraph_hip.cmake); return()` before the rapids bootstrap. NVIDIA path untouched.
- `cpp/cmake/hip/cugraph_hip.cmake` (NEW): standalone HIP build (deps via find_package from _deps prefixes + cuco source dir; CCCL via include paths; version_config.hpp regen; .cu -> LANGUAGE HIP via top-scope add_library/add_executable override; force-include compat header; --offload-compress + gc-sections; in-scope source list EXCLUDING the deferred legacy files).
- `cpp/cmake/hip/cugraph_hip_sources.cmake` (NEW): in-scope `.cu` list = CUGRAPH_SG_SOURCES + CUGRAPH_COMMON_SOURCES minus the deferred trio (spectral_clustering, mst, possibly hungarian) and minus the MG/MTMG sets; the cugraph_c sources minus legacy_spectral/legacy_mst.
- Compat shim header (NEW, small): `namespace cub = hipcub;` + any cuda* symbols not covered by rmm/raft's force-included compat header. Force-included on HIP TUs only.
- `cpp/tests/CMakeLists.txt`: under USE_HIP, build only the SG `ConfigureTest` targets, excluding BALANCED_TEST, MST_TEST, (HUNGARIAN_TEST if deferred), and the MG/MTMG tests. Prefer a HIP-side test list file over editing the macro.
- Per-file source fixes: expect a handful of clang/two-phase + guard-rewrite fixes (Risk 9), the FA2 wave64 audit (Risk 3), and any functor-return-by-reference fix (Risk 8). Keep diffs USE_HIP-guarded; CUDA path byte-for-byte.

## Build commands (gfx90a lead)
```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
# deps installed first (per their notes "Install as a dependency"):
#   _deps/rmm/install, _deps/raft/install, cuco fork source at _deps/cugraph-cuco/src
cmake -S projects/cugraph/src/cpp -B projects/cugraph/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft/install;/var/lib/jenkins/moat/_deps/rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DCUGRAPH_CUCO_SOURCE_DIR=/var/lib/jenkins/moat/_deps/cugraph-cuco/src \
  -DBUILD_TESTS=ON \
  -DBUILD_CUGRAPH_MG_TESTS=OFF
cmake --build projects/cugraph/build-hip -j16
```
(`-DCMAKE_HIP_ARCHITECTURES=gfx90a` is the only arch knob; a follower passes gfx1100/gfx1151 with no source change. A CPU-only docker compile check (rocm/dev-ubuntu-24.04:7.2.4-complete) is allowed as a manual gate, never the validation gate.)

## Test plan (REAL GPU gate on gfx90a; the validation deliverable)
The SG C++ gtests are the correctness slice (they build small graphs and compare against host/reference implementations). Required-green core (one process per case via ctest; run serially `ctest` not `-jN` on a single assigned GCD, MPPI lesson; set `HIP_VISIBLE_DEVICES` for GCD isolation):
- Traversal: BFS_TEST, SSSP_TEST, EXTRACT_BFS_PATHS_TEST, OD_SHORTEST_DISTANCES_TEST, MSBFS_TEST
- Link-analysis/centrality: PAGERANK_TEST, KATZ_CENTRALITY_TEST, EIGENVECTOR_CENTRALITY_TEST, HITS_TEST, BETWEENNESS_CENTRALITY_TEST, EDGE_BETWEENNESS_CENTRALITY_TEST
- Community: LOUVAIN_TEST, LEIDEN_TEST, ECG_TEST? (egonet), TRIANGLE_COUNT_TEST, EDGE_TRIANGLE_COUNT_TEST, K_TRUSS_TEST
- Components/cores: WEAKLY_CONNECTED_COMPONENTS_TEST, STRONGLY_CONNECTED_COMPONENTS_TEST, CORE_NUMBER_TEST, K_CORE_TEST
- Structure/utilities: SYMMETRIZE_TEST, TRANSPOSE_TEST, COARSEN_GRAPH_TEST, DEGREE_TEST, RENUMBERING_TEST, INDUCED_SUBGRAPH_TEST, WEIGHT_SUM_TEST
- Similarity: SIMILARITY_TEST, WEIGHTED_SIMILARITY_TEST
- Sampling/generators: GRAPH_GENERATORS_TEST, GENERATE_RMAT_TEST, RANDOM_WALKS_TEST, SAMPLING_POST_PROCESSING_TEST, NEGATIVE_SAMPLING_TEST
- Misc graph ops: MIS_TEST, VERTEX_COLORING_TEST, TOPOLOGICAL_SORT_TEST, K_HOP_NBRS_TEST, LOOKUP_SRC_DST_TEST, COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST, HAS_EDGE_AND_COMPUTE_MULTIPLICITY_TEST, REMOVE_MULTI_EDGES_TEST
Validation bar: each in-scope SG gtest builds and passes on gfx90a; correctness is the gtests' own reference comparisons (PageRank vs reference solver, BFS/SSSP distances vs CPU graph traversal, Louvain modularity, etc.); run-to-run determinism for the algorithms that should be deterministic.
Deferred (documented, NOT regressions): BALANCED_TEST (spectral/cuVS), MST_TEST (raft sparse-solver), legacy spectral C-API test; HUNGARIAN_TEST and LEGACY_FA2_TEST pending the raft/solver and FA2-wave64 audits; all MG/MTMG tests (NCCL/MPI multi-GPU, out of single-GPU scope); the Python pytest suite (needs the full RAPIDS ROCm Python wheel stack -- cudf/pylibraft/dask-cudf -- which MOAT does not deliver).
Non-GPU regression set: the NVIDIA build path must remain byte-for-byte (USE_HIP=OFF compiles unchanged); there is no separate CPU-only cugraph test suite to regress (it is a GPU library).

## Disposition
PROCEED -- fresh ROCm/HIP Strategy-A port. Lead platform linux-gfx90a advanced to `planned`. The deliverable is HIP-built `libcugraph` + `cugraph_c` with the SG C++ gtests green on gfx90a, deferring the cuVS-gated (spectral) and deferred-raft-module (mst, possibly hungarian/fa2) legacy algorithms and the multi-GPU + Python layers.

## Open questions
- Does `raft/solver/linear_assignment` (used by hungarian.cu) route entirely through raft's DELIVERED `raft/lap` path, or pull a deferred raft module? Resolve by attempting the include against the installed raft at porter time; if deferred, defer HUNGARIAN_TEST.
- Is `layout/legacy/bh_kernels.cuh` (Barnes-Hut FA2) a clean wave64 audit (just 64-bit masks) or does it pack two 32-lane rows positionally (popsift class)? Determines whether LEGACY_FA2_TEST is in-scope or deferred.
- Confirm the cuCollections fork source dir is wired the same way cudf wires it (the `-DCUDF_CUCO_SOURCE_DIR` analog); cugraph upstream CPM-fetches cuco via `rapids_cpm_cuco` -- the HIP CMake must point at the fork source instead.
- Whether `--offload-compress` alone keeps every cugraph .so under the 2 GiB host-image reach, or whether the prim template matrix needs a split beyond the existing common/sg/mg target boundaries.
