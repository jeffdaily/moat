# cuvs ROCm/HIP port plan (lead platform linux-gfx90a, MI250X, ROCm 7.2.1)

## Project

- Name: cuvs (RAPIDS vector search: brute-force, IVF-Flat, IVF-PQ, CAGRA, NN-Descent, Vamana/DiskANN, HNSW interop, kmeans, pairwise distance, SCANN).
- Upstream: https://github.com/rapidsai/cuvs
- Pinned tag: **v25.08.00** (commit 9ce11a0f3b29eb8e1dca154bfb710dc89f5af7b6), checked out in `projects/cuvs/src` (gitignored, shallow). CRITICAL: the upstream default branch (`main`) is VERSION 26.08.00; the ported rmm and raft are pinned at v25.08.00, and cuvs CMake requires an EXACT-MAJOR.MINOR raft (`find_and_configure_raft(VERSION ${RAPIDS_VERSION}.00 ...)`). cuvs MUST be ported at v25.08.00 so it links our 25.08 rmm/raft ABI. Do not use `main`.
- Fork: not yet created. The deliverable is `jeffdaily/cuvs @ moat-port` off the v25.08.00 tag (fork default branch stays a clean upstream mirror).
- depends_on: rmm (completed), raft (completed). See "Existing AMD support" for the nuance: raft is `completed` but the modules cuvs leans on hardest (neighbors top-k / ball_cover) are DEFERRED in raft.

## Existing AMD support

Finding: **none**. No `hip`/`rocm`/`amd` branch upstream, no ROCm fork among forks, and no `USE_HIP`/`__HIP_PLATFORM_AMD__`/`hip_runtime` token anywhere in `cpp/`. NVIDIA-only (CUDA + CUTLASS + cuBLAS/cuSOLVER/cuSPARSE/cuRAND). cuvs is also NVIDIA-affiliated (RAPIDS), so upstreaming is unlikely; this is a from-scratch CUDA-to-HIP port kept for the MOAT exercise.

Decision: **proceed with a from-scratch Strategy A port**, reusing the rmm and raft ports as the substrate and the raft CUTLASS->CK work as a direct template. This adds clear value (no HIP path exists) and is the next RAPIDS layer.

### The deps reality (read this before planning effort)

DEPENDENCIES.md and `projects/raft/notes.md` are blunt: raft is delivered PARTIAL. raft's `completed` state covers core/linalg/random/label/utils plus the CUTLASS->CK pairwise-distance and fused-distance-NN fast paths (DISTANCE 11/11, FUSED_NN 12/12, MATRIX_SELECT 607 on gfx90a). But raft's **NEIGHBORS** module is DEFERRED (`RAFT_TEST_NEIGHBORS OFF`): ball_cover's FAISS `KeyValueBlockSelect` mis-sorts on wave64 (reconvergence artifact), and `faiss_select` carries a hard `static_assert(WarpSize==32)`. cuvs's vector-search algorithms sit on exactly the neighbors/top-k primitives that raft deferred.

Bigger structural fact discovered by inventory: **cuvs v25.08 forked the entire dense-distance subsystem out of raft into its own `cuvs::distance::detail` namespace.** cuvs does NOT call `raft::distance::pairwise_distance`; it ships its own copy of every CUTLASS distance file, its own `pairwise_matrix_dispatch`, its own fused-distance-NN, AND its own copy of the FAISS select / brute-force / knn_merge_parts code. So cuvs cannot simply inherit raft's CK distance fix -- it needs the SAME CUTLASS->CK reimplementation redone in the cuvs namespace. raft's `dispatch_ck.cuh` + `dispatch_fused_nn_ck.cuh` are a proven, near-line-for-line template, but they are raft-namespaced and not installed for reuse.

Consequence: a fully-validated cuvs (all algorithms passing on gfx90a) is a LARGE port that re-treads raft's hardest walls plus adds CAGRA/NN-Descent wave64 work raft never had to do. A correctness-first staged port (below) with a clearly-scoped deferral set is the realistic deliverable, mirroring how raft itself was marked `completed` with ball_cover deferred.

## Build classification: Strategy A (pure CMake RAPIDS library)

Evidence:
- `cpp/CMakeLists.txt:13-18` bootstraps rapids-cmake (`include(rapids-cmake/rapids-cpm/rapids-export/rapids-find)`), `project(CUVS LANGUAGES CXX CUDA)` (line 31), `rapids_cuda_init_architectures` (line 27). No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`. The python layer (`python/cuvs`, 17 `.pyx`) is Cython over the C API, not a torch extension.
- `rapids_cpm_init()` + `rapids_cpm_cccl()` + `include(get_raft.cmake)` + `include(get_cutlass.cmake)` (lines 173-180): identical RAPIDS CPM bootstrap to rmm and raft.

This is the same pure-CMake RAPIDS shape as rmm and raft -> **Strategy A** (compat header + `enable_language(HIP)` + `.cu` marked `LANGUAGE HIP`), using the exact `option(USE_HIP) ... include(cmake/hip/cuvs_hip.cmake) return()` early-guard pattern raft/rmm used to bypass the CCCL/CUTLASS CPM fetch.

## Port strategy (Strategy A; correctness-first with a CUTLASS->CK fast path follow-on)

Rationale: cuvs is not a torch extension, so Strategy B does not apply. The diff stays minimal by (a) gating the whole rapids-cmake/CPM bootstrap behind a top-of-file `USE_HIP` guard that hands off to a standalone `cuvs_hip.cmake` (consuming our installed rmm+raft via `find_package` instead of CPM-fetching them), (b) marking the existing `.cu` `LANGUAGE HIP` (never renaming), and (c) force-including a small cuvs compat header on HIP TUs. cuvs inherits rmm's and raft's compat headers transitively (both are force-included by their exported targets), so the cuvs-specific compat header only adds what cuvs uses beyond raft's surface.

For the CUTLASS-based hot kernels (pairwise distance, fused-distance-NN): a **correctness-first mechanical port** that forces the non-CUTLASS SIMT/sm60 path on HIP ships first (raft "Layer A"), then a **CK fast-path** wired into cuvs's dispatch (raft "Layer B", reusing raft's validated `DeviceGemmMultipleD_Xdl_CShuffle` CDE-epilogue design). CUTLASS does NOT port to ROCm (PORTING_GUIDE, per jeff) -- do not attempt a CUTLASS shim. CAGRA/NN-Descent are hand-written SIMT kernels (no CUTLASS); they need wave64 correctness work, not a GEMM rewrite.

## CUDA surface inventory

Counts are over `cpp/` at v25.08.00. 356 `.cu` total (264 under `src/`, of which ~201 compile into `libcuvs.so` via `cuvs_objs` and 64 into the separate `cuvs-cagra-search` static lib), 196 `.cuh`, 70 `.hpp`.

| Surface | Where | ROCm/HIP mapping | Risk |
|---|---|---|---|
| CUTLASS GEMM + distance epilogue (pairwise distance) | `src/distance/detail/pairwise_distance_cutlass_base.cuh`, `pairwise_distance_gemm.h`, `pairwise_distance_epilogue*.h`, `pairwise_matrix/dispatch_sm80.cuh`, `predicated_tile_iterator_normvec.h`, `distance_ops/cutlass.cuh` | NO CUTLASS port. Reimplement against CK (classic `DeviceGemmMultipleD_Xdl_CShuffle` MFMA + CDE epilogue) exactly as raft did. Layer A first: `#if !defined(USE_HIP)` the CUTLASS includes/bodies and force the sm60 SIMT path (`dispatch_sm60.cuh`, present in cuvs's own namespace). | HIGH (but raft template exists) |
| CUTLASS fused-distance-NN (argmin reducing epilogue) | `src/distance/detail/fused_distance_nn/{cutlass_base,epilogue,epilogue_elementwise,gemm.h,persistent_gemm.h,custom_epilogue_with_broadcast.h,predicated_tile_iterator_*}.cuh/.h`, `fused_distance_nn.cu(h)`, `fused_l2_nn.cuh`, `fused_cosine_nn.cuh`, `simt_kernel.cuh` | CK reimplementation = CK distance GEMM materializing D[M,N] + a plain-shared-mem `row_argmin_kernel` (NO warp shuffles -> wave64-safe), per raft's `dispatch_fused_nn_ck.cuh`. Layer A forces the SIMT `simt_kernel.cuh` path. | HIGH (raft template exists) |
| 119 `#include <cutlass/...>` / `<cute/...>` lines across 26 files | distance + `neighbors/detail/knn_brute_force.cuh` + `sparse/neighbors/detail/cross_component_nn.cuh` | All become `#if !defined(USE_HIP)` guarded; the dispatchers take SIMT/CK on HIP. `get_cutlass.cmake` is skipped entirely on HIP (not fetched). | MED (mechanical, follows raft) |
| FAISS warp/block select top-k (neighbors) | `neighbors/detail/{knn_brute_force,fused_l2_knn,knn_merge_parts}.cuh`, `neighbors/ball_cover/registers.cuh`, `internal/cuvs_internal/matrix/select_k.cuh` | hipify-clean spellings; the HARD part is wave64 correctness (32-bit masks, `kNumWarpQRegisters=NumWarpQ/WarpSize`, KeyValueBlockSelect reconvergence). Reuse raft's `faissWarpQ`/`WarpMask`/`select_k` fixes; route brute-force through the wave64-safe tiled path on CDNA (raft pattern). | HIGH (this is the exact raft NEIGHBORS deferral) |
| CAGRA search kernels | `neighbors/detail/cagra/search_{single,multi}_cta_kernel-inl.cuh`, `search_multi_kernel.cuh`, `topk_for_cagra/topk_core.cuh`, `graph_core.cuh`, `hashmap.hpp`, `bitonic.hpp`, `device_common.hpp` | `__shfl`/`__ballot` (6-8 files) + hardcoded `constexpr unsigned warp_size = 32` (`device_common.hpp:37`, `search_single_cta_kernel-inl.cuh:463,841`) + dim/team-size templating (`dim128_t8`/`dim256_t16`/`dim512_t32` = threads-per-query tied to warp=32). This is a popsift-class warp-width coupling inside a complex graph search. | HIGH (new; raft has no CAGRA) |
| NN-Descent | `neighbors/detail/nn_descent.cuh` | `static_assert(raft::warp_size()==32)` (lines 163,452) guarding a hardcoded 32-lane `warp_bitonic_sort`; explicit `asm("bfe.u32 ...")` PTX (line 156). Needs a wave64 bitonic network (or force a 32-lane sub-group) + `__bfe`/shift replacement for the PTX. | HIGH (new) |
| IVF-PQ sub-warp reductions | `neighbors/ivf_pq/ivf_pq_process_and_fill_codes*.cuh`, `ivf_pq_codepacking.cuh` | Uses `raft::WarpSize`-parameterized `kSubWarpSize`/`SubWarpSize` with `raft::shfl_xor(..., SubWarpSize)` and `raft::Pow2<SubWarpSize>::mod(raft::laneId())` -- raft's wave-width-correct primitives. Adapts to wave64 via raft's abstraction; SOFTER than CAGRA. | MED |
| cub/thrust | throughout (DeviceScan/Reduce/RadixSort/Select via raft+cuvs) | rocThrust/hipCUB from `/opt/rocm/include` (hipcc default) + raft's `hip_compat/cub` shim (inherited). rocPRIM requires C++17 (cuvs already CXX_STANDARD 17). Watch the rocPRIM cuda::std::tuple-key shims and the wave64 BlockReduce/RadixSort TempStorage-reuse `__syncthreads()` race. | MED |
| cuBLAS / cuSOLVER / cuSPARSE / cuRAND | `CUVS_CTK_MATH_DEPENDENCIES` (CMakeLists 598-603); kmeans, spectral, sparse | hipBLAS/hipSOLVER/hipSPARSE/hipRAND via raft's installed `hip_compat` shims + `raft_mathlib_aliases.inc` (inherited from raft::raft). hipBLAS v2 enum / status-enum-not-1:1 gaps already solved in raft. | LOW-MED |
| cuco (cuCollections) | `src/distance/detail/sparse/coo_spmv_strategies/hash_strategy.cuh` ONLY (`cuco::legacy::static_map`, block-scope) | Module-level (sparse SpMV hash strategy), off the core vector-search path. jeffdaily/cuCollections is delivered PARTIAL (>=4-byte keys). Defer the sparse hash path if it does not build cleanly; it does not gate IVF/CAGRA. | LOW (scope-limited) |
| Inline PTX asm | `nn_descent.cuh` (bfe), `dynamic_batching.cuh`, `distance/detail/sparse/utils.cuh` (`__ldcg`, `=l` asm) | Replace with portable intrinsics/builtins on HIP (raft did `__ldcg`/`__stwt`->plain load/store in the compat header). | MED |
| jitify / NVRTC / driver API | none | `__global__`/`__device__` everywhere is statically compiled. NO `cuLaunchKernel`/`nvrtc`/`jitify`. hipRTC not needed. | NONE (positive) |
| textures / surfaces | none | No `texture<>`/`cudaTextureObject`/`surf*`. The colmap/popsift texture fault classes do not apply. | NONE (positive) |
| atomics | atomicMin/Max (6 files), atomicCAS (7), one `atomicAdd_block` | Watch the cudaKDTree class: int `atomicMin`/`atomicMax` are silently dropped on COARSE-GRAINED (managed) memory on gfx90a -- emulate with atomicCAS loop where they touch managed memory. Most cuvs allocations are `hipMalloc` device memory (via rmm) where the RMW works; audit any managed/`prefetch` path. | MED |
| NCCL (multi-GPU) | `BUILD_MG_ALGOS` -> `neighbors/mg/*`, `NCCL::NCCL` (CMakeLists 613) | RCCL is the ROCm NCCL drop-in. The MG (multi-GPU) algos are optional (`-DBUILD_MG_ALGOS=OFF`). Defer for the lead bringup; not part of single-GPU correctness. | DEFER |

## Risk list (project-specific, keyed to PORTING_GUIDE fault classes)

1. **Version skew (build-breaking if wrong).** cuvs must be v25.08.00 to match the installed rmm/raft 25.08. `main`=26.08 will fail `find_package(raft 26.08)`. Already handled by the pinned checkout; the porter must keep it.

2. **wave64 in CAGRA (the headline new risk).** `device_common.hpp` `constexpr unsigned warp_size = 32`, the `t8/t16/t32` team-size template instantiations, and the search-kernel `__ballot`/`__shfl` assume a 32-lane warp. On CDNA a 64-lane wavefront either packs two teams into one wavefront (popsift two-rows-per-wavefront trap: per-row leader election / ballot must operate per 32-lane half) OR the team size must grow to use 64 lanes. Decide per kernel: prefer routing team-size through `raft::WarpSize` so a wave64 team is 64 lanes (matches IVF-PQ's approach) where the algorithm allows; otherwise treat the wavefront as two independent 32-lane groups. This is the single largest wave64 unknown and the likely first hard wall after the build compiles.

3. **wave64 in NN-Descent bitonic sort.** `static_assert(raft::warp_size()==32)` + a hand-unrolled 32-lane `xor_swap` bitonic network + `bfe.u32` PTX. Needs either a 64-lane bitonic network (add the stride-32 merge step, as raft did for FAISS MergeNetworkWarp) or a forced 32-lane sub-group. Replace the PTX `bfe` with `__bfe`/shifts.

4. **wave64 in FAISS select / ball_cover (inherited raft deferral).** cuvs's own copies of `Select`-family code carry the same `NumWarpQ/WarpSize -> 0` collapse, 32-bit masks, and the KeyValueBlockSelect partial-warp reconvergence mis-sort that raft deferred (`-DNDEBUG` gives wrong top-k, debug SIGABRTs via `__hip_check_mask`). Reuse raft's `faissWarpQ=max(warp_q,WarpSize)`, `WarpMask`/64-bit ballot, and the "route brute-force through the wave64-safe tiled select_k path on CDNA" pattern. ball_cover itself may have to be deferred in cuvs exactly as in raft.

5. **CUTLASS->CK fidelity (distance + fused-NN).** The CK CDE epilogue must reproduce the cuvs op bodies byte-for-byte: `l2_exp_cutlass_op` (relu only inside sqrt branch + per-dtype self-neighbor clamp 1e-3/1e-6/1e-15) for pairwise distance, and `l2_exp_distance_op::epilog` (relu ALWAYS, then optional sqrt) for fused-NN argmin -- these are DIFFERENT ops (raft's review caught exactly this). Cosine uses NON-squared norms. Validate against a double-precision CPU reference. Never validate argmin against a periodic/low-entropy data generator (raft's exact-tie debugging time-sink): use a splitmix hash embedding the row index.

6. **Large-`.so` x86-64 relocation wall (`--offload-compress`).** 264 heavily-templated `.cu` (51 CAGRA `compute_distance_*` permutations alone, plus the full distance metric x dtype x int-width matrix) compile into `libcuvs.so` + `cuvs-cagra-search`. cudf hit `R_X86_64_PC32`/TLS/`.eh_frame` overflow from uncompressed HIP offload bundles at this scale. Apply the proven cudf lever: `target_compile_options(<tgt> PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-compress>)` (54.9x device-fatbin shrink) + single target arch + `-ffunction-sections -fdata-sections` + `--gc-sections`. NOTE: `cuvs-cagra-search` is `CUDA_SEPARABLE_COMPILATION ON` (RDC = `-fgpu-rdc` on HIP) and `cuvs_c` uses a `fatbin.ld` linker script -- so unlike cudf there IS a device-link step; flag `--offload-compress` on BOTH the per-TU compile AND the device-link for the RDC target. Also drop `-maxrregcount=64` (nvcc spelling; HIP wants `-Xoffload-linker`/`--offload-arch` reg control or just remove it for first bringup) and the nvcc-only `-Xfatbin=-compress-all`, `--expt-extended-lambda`, `--expt-relaxed-constexpr`, `-static-global-template-stub` flags under `USE_HIP`.

7. **`__CUDA_ARCH__` undefined on HIP.** Inherited from raft's compat header (defines `__CUDA_ARCH__ 800` in the HIP device pass only). cuvs's sm60-vs-sm80 dispatch and `raft::util::arch::SM_*` checks rely on this; the cuvs compat header force-include must occur before any cuvs TU (same `-include` mechanism raft uses). Enabling `__CUDA_ARCH__=800` activates `__dp4a`/cache-intrinsic device branches -- the raft compat header already handles these; cuvs adds only its own extras.

8. **Two-phase lookup / `this->member` / dependent-base (clang strictness).** cuvs is heavily templated; expect the cudaKDTree/MPPI clang-vs-nvcc class (`this->` for inherited dependent-base members, `typename Dep::X`, host/device attribute matching on explicit specializations, `CLASS::STATIC_CONST` not `this->STATIC_CONST`). Mechanical, fix as the compiler surfaces them.

9. **Narrowing conversions are hard errors under clang-as-hipcc** (warnings on nvcc). raft hit `{colId, acc}` long->uint32 narrowing in fused_l2_knn and simt_kernel; cuvs's copies of those files will need the same `static_cast`.

10. **ffp-contract / fsqrt 1-ULP** (CV-CUDA class). If any cuvs gtest does bit-exact float compares, pin `-ffp-contract=on` in `CMAKE_HIP_FLAGS` and route f32 `sqrt` through f64. Most cuvs ANN tests use recall/tolerance metrics, not bit-exactness, so this is lower-risk here than in CV-CUDA -- confirm per failing test.

## File-by-file change list (additive; NVIDIA path untouched)

New files (HIP-only, mirror raft's layout):
- `cpp/CMakeLists.txt` -- add top-of-file `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`; immediately after the rapids includes, `if(USE_HIP) include(cmake/hip/cuvs_hip.cmake) return() endif()` to bypass `rapids_cpm_init`/`rapids_cpm_cccl`/`get_raft`/`get_cutlass`. This is the ONLY edit to the stock CMakeLists.
- `cpp/cmake/hip/cuvs_hip.cmake` (new) -- the standalone HIP build: `enable_language(HIP)`; arch from `${CMAKE_HIP_ARCHITECTURES}` defaulted to gfx90a only when unset; `find_package(rmm)` + `find_package(raft)` against `_deps/*/install`; regenerate the rapids-cmake-generated headers cuvs needs (`version_config.h` -- cuvs writes it via `rapids_cmake_write_version_file`, replicate); define the `cuvs_objs` / `cuvs-cagra-search` / `cuvs` / `cuvs_c` targets with the stock source lists marked `LANGUAGE HIP`; `--offload-compress` + gc-sections + (for cuvs-cagra-search) RDC device-link compress; force-include the cuvs compat header; strip nvcc-only flags. Plus `cuvs_hip_tests.cmake` with per-test-group `CUVS_TEST_*` options (so crashing groups can be gated OFF like raft's `RAFT_TEST_*`).
- `cpp/include/cuvs/util/hip/cuda_to_hip.h` (new) -- the cuvs compat header (force-included on HIP TUs). Adds only what cuvs uses beyond what rmm::rmm and raft::raft already alias (cuvs inherits both compat headers transitively). Likely minimal: any cuvs-specific cache intrinsic / datatype enum / math alias not already covered.
- `cpp/hip_compat/` (new, if needed) -- forwarding shims for any CUDA-named toolkit header cuvs includes directly that raft's shim dir does not already cover (raft already shims cublas_v2/cusolverDn/cusparse/curand/cub/cooperative_groups/fp16/bf16). Add cuvs's include path BEFORE only under `if(USE_HIP)`.

Guarded edits to cuvs sources (all `#if defined(USE_HIP)` / `#if !defined(USE_HIP)`; CUDA path byte-identical):
- Distance CUTLASS files (26): `#if !defined(USE_HIP)` the `#include <cutlass/...>`/`<cute/...>` and CUTLASS-only bodies; in `pairwise_matrix/dispatch-inl.cuh` force `cutlass_op_unavailable=true` (or take CK) on HIP; flip the sm60 fallback `#if __CUDA_ARCH__ < 800` bodies to also compile under `USE_HIP`.
- `distance/detail/pairwise_matrix/dispatch_ck.cuh` + `fused_distance_nn/dispatch_fused_nn_ck.cuh` (new, cuvs-namespaced) -- ported from raft's validated headers; wired into cuvs's dispatchers (Layer B).
- CAGRA (`device_common.hpp`, `search_*_kernel-inl.cuh`, `topk_core.cuh`, `bitonic.hpp`, `hashmap.hpp`, `graph_core.cuh`) -- wave64: route `warp_size` through `raft::WarpSize`/a per-arch constant; fix `__ballot`/`__shfl` masks to 64-bit; handle team-size-vs-wavefront.
- `neighbors/detail/nn_descent.cuh` -- wave64 bitonic network + `__bfe`/shift for the PTX; relax/parameterize the `static_assert`.
- FAISS select copies (`knn_brute_force.cuh`, `fused_l2_knn.cuh`, `knn_merge_parts.cuh`, `ball_cover/registers.cuh`, `select_k.cuh`) -- reuse raft's `faissWarpQ`/`WarpMask`/64-bit-ballot fixes; route brute-force to tiled select_k on CDNA.
- PTX/cache-intrinsic sites (`dynamic_batching.cuh`, `distance/detail/sparse/utils.cuh`) -- portable intrinsics on HIP.
- Narrowing/`this->`/dependent-base fixes as the compiler surfaces them.

## Build commands (gfx90a)

Prereqs: ported rmm + raft installed (see deps recipe). conda env `py_3.12` (gtest/gmock/spdlog/fmt), the rmm-vendored libhipcxx + rapids_logger clones.

```bash
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/cuvs/src/cpp -B projects/cuvs/build -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft/install;/var/lib/jenkins/moat/_deps/raft-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=ON \
  -DBUILD_C_LIBRARY=OFF \
  -DBUILD_MG_ALGOS=OFF \
  -DBUILD_CAGRA_HNSWLIB=OFF \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/cuvs/install
cmake --build projects/cuvs/build -j$(nproc)
```

Arch is read from `CMAKE_HIP_ARCHITECTURES` (defaulted to gfx90a only when unset), so a follower validates with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` and no source edit. Turn `BUILD_C_LIBRARY`/`BUILD_MG_ALGOS`/`BUILD_CAGRA_HNSWLIB` back ON in later passes once core algorithms validate (C API needs dlpack fetched; HNSW needs hnswlib fetched; MG needs RCCL). Bring the library up in stages via the `CUVS_TEST_*` gating options in `cuvs_hip_tests.cmake` -- do not try to compile all 264 `.cu` green at once.

## Test plan

cuvs ships gtest C++ tests (`cpp/tests/`) plus pytest python tests (`python/cuvs/cuvs/tests/`, 15 files) and C-API/Java/Rust/Go bindings (out of scope for the lead). The validator's GPU correctness gates are the gtest binaries; the python tests are a secondary gate once the C API + python build.

GPU gtest targets (from `cpp/tests/CMakeLists.txt`, run on real gfx90a with `HIP_VISIBLE_DEVICES` GCD isolation, SERIALLY -- one GPU):

| Test target | Covers | Staging |
|---|---|---|
| DISTANCE_TEST | pairwise distance (all metrics x dtypes) | Stage 1 (CK + SIMT, the raft template) |
| NEIGHBORS_TEST | brute_force, brute_force_prefiltered | Stage 2 (faiss select wave64) |
| NEIGHBORS_ANN_BRUTE_FORCE_TEST | ANN brute force | Stage 2 |
| NEIGHBORS_ANN_IVF_FLAT_TEST | IVF-Flat build/search | Stage 3 |
| NEIGHBORS_ANN_IVF_PQ_TEST | IVF-PQ build/search | Stage 3 (raft sub-warp primitives) |
| NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST (+half/int8/uint8, +BUGS) | CAGRA | Stage 4 (wave64 graph search -- hardest) |
| NEIGHBORS_ANN_NN_DESCENT_TEST | NN-Descent graph build | Stage 4 (wave64 bitonic) |
| NEIGHBORS_ANN_VAMANA_TEST | Vamana/DiskANN build | Stage 4 |
| NEIGHBORS_ALL_NEIGHBORS_TEST | all-neighbors | Stage 4 |
| NEIGHBORS_BALL_COVER_TEST | ball_cover | Likely DEFER (raft's open wave64 blocker) |
| NEIGHBORS_TIERED_INDEX_TEST | tiered index | Stage 3/4 |
| NEIGHBORS_ANN_SCANN_TEST | SCANN | Stage 4 |
| CLUSTER_TEST | kmeans, kmeans_balanced, find_k | Stage 2 (kmeans on raft primitives) |
| PREPROCESSING_TEST | scalar/binary quantization | Stage 2 (binary.cuh hardcodes warp_size=32 -> wave64) |
| STATS_TEST | trustworthiness / silhouette etc. | Stage 2 |
| SPARSE_TEST | sparse distance / cross_component_nn | Likely DEFER (cuco + CUTLASS sparse + raft sparse deferred) |
| C-API tests (BRUTEFORCE/CAGRA/IVF_*/DISTANCE/HNSW/INTEROP) | C bindings | After core, with `BUILD_C_LIBRARY=ON` |

ANN correctness is measured by **recall against a brute-force/CPU reference at a tolerance**, not bit-exactness -- so the validator asserts recall thresholds (the tests embed them) rather than bitwise match. Determinism where claimed: re-run and compare.

Non-GPU regression set that must not regress: the host-only portions of the above gtests (parameter validation, serialization round-trips, interop/dlpack shape checks). There is no separate CPU-only cuvs test suite; the gtests mix host and device. The rmm/raft installs are dependencies, not part of cuvs's regression set, but their `_deps` builds must remain intact.

Realistic `completed` bar (mirroring raft): DISTANCE + the CK fast path + IVF-Flat + IVF-PQ + brute-force + kmeans + preprocessing + at least the float CAGRA path GPU-validated on gfx90a, with ball_cover / sparse / SCANN / MG / NN-Descent-on-wave64 deferred behind documented `CUVS_TEST_*` options if their wave64/CUTLASS-sparse/cuco walls are not cleared in the first port. State the deferral set explicitly in notes.md as raft did.

## Deps-install recipe (the porter follows this first)

rmm and raft forks are at `jeffdaily/<dep> @ moat-port`; the installs already exist on this host at `_deps/raft-rmm/install` (rmm) and `_deps/raft/install` (raft) with `librmm.so`/`libraft.so` + their `cuda_to_hip.h` compat headers + cmake configs present. To (re)build from the forks per DEPENDENCIES.md:

1. rmm (per `projects/rmm/notes.md` "Install as a dependency"):
```bash
git clone -b moat-port https://github.com/jeffdaily/rmm _deps/rmm/src   # if re-cloning
# vendored deps (once per host): libhipcxx (amd-develop) + rapids-logger (release/0.2.0)
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S _deps/rmm/src/cpp -B _deps/rmm/build -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=<X>/libhipcxx/include \
  -DRAPIDS_LOGGER_SOURCE_DIR=<X>/rapids_logger \
  -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/raft-rmm/install
cmake --build _deps/rmm/build --target install -j$(nproc)
```

2. raft (per `projects/raft/notes.md` "Install as a dependency", after rmm is installed):
```bash
git clone -b moat-port https://github.com/jeffdaily/raft _deps/raft/src   # if re-cloning
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S _deps/raft/src/cpp -B _deps/raft/build -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/raft/install
cmake --build _deps/raft/build --target install -j$(nproc)
```

3. cuvs consumes both via the configure line above: `find_package(raft)` (pulls `raft::raft` which transitively pulls `rmm::rmm`, both compat headers force-included), with `CMAKE_PREFIX_PATH` listing BOTH the raft and rmm install prefixes plus `/opt/rocm` and the conda env. At runtime put `_deps/raft/install/lib`, `_deps/raft-rmm/install/lib`, and `$CONDA_PREFIX/lib` on `LD_LIBRARY_PATH`.

cuvs's own CUTLASS dependency is NOT a MOAT dep: `get_cutlass.cmake` is skipped entirely on HIP (CUTLASS does not port to ROCm). dlpack (C API), hnswlib (CAGRA-HNSW), NCCL/RCCL (MG) are optional and OFF for the lead bringup.

## Open questions

1. CAGRA team-size vs wavefront: can the `t8/t16/t32` threads-per-query templates be widened to use 64 lanes on CDNA (cleaner, matches IVF-PQ), or must each kernel treat the 64-lane wavefront as two 32-lane teams (popsift pattern)? Resolve empirically once CAGRA compiles -- this is the largest correctness unknown.
2. How much of cuvs's distance subsystem is reachable WITHOUT the CK fast path? If the SIMT/sm60 fallback covers all metrics correctly (it should, per raft), Layer A alone may pass DISTANCE_TEST and unblock the ANN algorithms, deferring CK to a perf follow-on. Confirm the cuvs sm60 path compiles under USE_HIP the way raft's did.
3. ball_cover: cuvs's `neighbors/ball_cover/registers.cuh` is the same FAISS KeyValueBlockSelect raft deferred on wave64. Expect to defer NEIGHBORS_BALL_COVER_TEST unless the raft "wave_barrier at mergeWarpQ" or "route through select_k" fix is landed here first.
4. Does cuvs build its python/Cython layer against the C API on ROCm, or is the gtest GPU suite the sole validation gate? Lead bringup targets gtests; python is a stretch goal after `BUILD_C_LIBRARY=ON`.
