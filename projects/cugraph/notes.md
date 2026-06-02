# cugraph notes

RAPIDS graph analytics. ROCm/HIP port (Strategy A), lead linux-gfx90a, ROCm 7.2.1.
Upstream rapidsai/cugraph @ e314f1e (VERSION 26.08.00). Fork jeffdaily/cugraph,
branch moat-port. The C++ libcugraph SG (single-GPU) graph-primitive slice is the
target; MG/MTMG (NCCL/MPI), cugraph_c (links the MG lib), and the Python layer are
out of scope. See plan.md for the full scope/deferral rationale.

## Status: IN PROGRESS (blocked, library nearly compiling)

Session 2 resolved ~14 systematic interop/multi-arch fault classes (see "Session 2
progress" below). cugraph_common now compiles entirely EXCEPT the 2 cuco-blocked
renumber_utils TUs (class C); the SG cugraph library has ~92 of ~124 .cu compiling
with 32 primitive groups remaining (classes A/B/D). The library does not yet link.
Remaining is the rest of the zip-WRITE boundary + rocPRIM block_load value-type
mismatch across the frontier/per-prim impls, the cuCollections-fork heterogeneous_value
delta (a second repo we own), then link + the SG gtests + GPU validation on gfx90a.
Kept blocked (no false-port) per the no-thrash rule; the continuation plan below is
precise and the fixes are mechanical/bounded.

## Build recipe (gfx90a lead)

CMake 4.0+ is REQUIRED (cugraph's cmake_minimum_required(VERSION 4.0); the conda
base ships 3.31). Install once: `pip install "cmake==4.0.*"` into the py_3.12 env.

```
export HIP_VISIBLE_DEVICES=0
export CONDA_PREFIX=/opt/conda/envs/py_3.12
export PATH=$CONDA_PREFIX/bin:/opt/rocm/bin:$PATH
RMM_INSTALL=/var/lib/jenkins/moat/_deps/raft-rmm/install     # rmm raft was built against
RAFT_INSTALL=/var/lib/jenkins/moat/_deps/raft/install
CUCO_SRC=/var/lib/jenkins/moat/projects/cuCollections/src    # moat-port fork, header-only
LIBHIPCXX_INC=/var/lib/jenkins/moat/_deps/libhipcxx/include
export LD_LIBRARY_PATH=$RAFT_INSTALL/lib:$RMM_INSTALL/lib:$CONDA_PREFIX/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

cmake -S projects/cugraph/src/cpp -B projects/cugraph/build-hip -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$RAFT_INSTALL;$RMM_INSTALL;/opt/rocm;$CONDA_PREFIX" \
  -DCUGRAPH_CUCO_SOURCE_DIR=$CUCO_SRC \
  -DLIBHIPCXX_INCLUDE_DIR=$LIBHIPCXX_INC \
  -DBUILD_TESTS=ON
cmake --build projects/cugraph/build-hip -j16
```

The env is captured in agent_space/cugraph/env.sh.

## Dependency contract (verified)

- rmm:  `_deps/raft-rmm/install` (jeffdaily/rmm @ moat-port, the build raft used).
        Exports rmm::rmm. NOTE this rmm is ~25.x: memory-resource headers live
        under `rmm/mr/device/` and pool_memory_resource is the TEMPLATED form.
- raft: `_deps/raft/install` (jeffdaily/raft @ moat-port). Exports raft::raft +
        raft::raft_compiled (NOT `raft::compiled`). find_dependency(rmm) pulls
        rmm, so BOTH prefixes on CMAKE_PREFIX_PATH. raft was built at C++17;
        cugraph builds at C++20 (see assume_aligned / span fixes below).
- cuco: cuCollections fork source `projects/cuCollections/src` @ moat-port
        (47ae24d). add_subdirectory()'d with USE_HIP=ON; provides cuco::cuco +
        the cub/cooperative_groups hip_compat shim. Needs
        -DLIBHIPCXX_INCLUDE_DIR=_deps/libhipcxx/include.
- cudf: NOT a C++ dependency of libcugraph (Python-layer only). Not used.

## What is done (committed to moat-port)

CMake (NVIDIA path byte-for-byte unchanged behind `option(USE_HIP)` + early
include+return in cpp/CMakeLists.txt):
- cpp/cmake/hip/cugraph_hip.cmake -- standalone HIP build: enable_language(HIP),
  find_package(rmm/raft), add_subdirectory(cuco), generated version_config.hpp,
  builds cugraph_common (SHARED) + cugraph (SG, SHARED), --offload-compress +
  gc-sections, force-includes the compat header, install/export.
- cpp/cmake/hip/cugraph_hip_sources.cmake -- SG + COMMON source lists minus the
  deferred legacy trio (spectral_clustering, mst, hungarian) + force_atlas2; the
  cugraph_c C-API is deferred (it links cugraph_mg).
- cpp/cmake/hip/cugraph_hip_tests.cmake -- cugraphtestutil + the in-scope SG
  gtests (BFS/SSSP/PAGERANK/.../SIMILARITY); deferred legacy + MG/C-API tests out.
- compat header cpp/src/hip/cuda_to_hip.h, force-included on every HIP TU.
- rmm path-forwarding shims cpp/src/hip/rmm_compat/rmm/mr/*.hpp (BEFORE rmm on
  the include path).

Source fixes (all USE_HIP-guarded; CUDA byte-for-byte):
1. assume_aligned: cugraph at C++20 makes raft's vendored mdspan aligned_accessor
   take the std::assume_aligned branch, but that header does not include <memory>.
   Fixed by `#include <memory>` in the compat header (ahead of raft).
2. rmm header relocation: cugraph 26.08 includes rmm/mr/{cuda_memory_resource,
   per_device_resource,polymorphic_allocator,pool_memory_resource}.hpp at the flat
   path; the ported rmm has them under rmm/mr/device/. Path-forwarding shims.
3. rmm non-templated pool_memory_resource: host_staging_buffer_manager.hpp uses
   rmm 26.08's rmm::mr::pool_memory_resource (no template arg); the ported rmm has
   only the templated form -> use pool_memory_resource<pinned_host_memory_resource>
   on HIP.
4. raft span.hpp converting-ctor enable_if defect surfaced at C++20: raft's
   span converting constructor uses `class = typename std::enable_if<...>` WITHOUT
   `::type`, so it is never SFINAE-disabled; libstdc++ 14's C++20 std::variant
   probe then hard-errors in arithmetic_variant_types.hpp. Worked around on the
   cugraph side with std::in_place_type at the 2 variant-of-span construction
   sites (avoids the converting-ctor probe). The real defect is in raft's header.
5. clang two-phase: kv_store.cuh needs `template ref_type<...>` (dependent
   template name). (More such `template`/`this->` fixes are expected across the
   prims; class 5 below.)
6. rocThrust cuda::std::get on thrust tuples -- THE systemic read-boundary fix.
   This rocThrust uses its OWN classic 10-slot thrust::tuple (NOT aliased to
   cuda::std::tuple, unlike CCCL thrust), and zip iterators dereference to
   thrust::detail::tuple_of_iterator_references. cugraph (CCCL-native) calls the
   qualified cuda::std::get<I>(x) pervasively on these. Fix: add cuda::std::get
   overloads (for thrust::tuple AND tuple_of_iterator_references) forwarding to
   thrust::get, in the compat header. This resolved ALL the read-boundary sites
   (e.g. utility_wrappers now compiles). On CUDA the overloads never participate.
7. rocThrust zip-write boundary (shuffle_comm.cuh, 2 sites): a thrust::transform
   functor returning cuda::std::make_tuple cannot write through a zip output whose
   reference is a classic thrust::tuple ("no viable operator="). Fixed with a
   file-local cugraph_zip_make_tuple = thrust::make_tuple on HIP / cuda::std on
   CUDA. (This class recurs; see below.)

## Session 2 progress (fork advanced; ~14 systematic classes resolved)

State after this session: `cugraph_common` compiles ENTIRELY except the 2
renumber_utils_common TUs (the cuco delta, class C below). The SG `cugraph`
library has ~92 of ~124 .cu TUs compiling; 32 SG primitive groups remain. The
deps wiring + the version-skew tier were already done in session 1; this session
resolved the systemic interop classes. The build recipe and deps are unchanged.

Resolved this session (all USE_HIP-guarded; CUDA path byte-for-byte):
1. __CUDACC__ guards -> `#if defined(__CUDACC__) || defined(__HIPCC__)` in
   partition_manager.hpp + thrust_tuple_utils.hpp (18 guards). Without this every
   __host__ __device__ helper degraded to host-only under clang HIP (HIP defines
   __HIPCC__, not __CUDACC__), breaking all device callers.
2. Multi-arch compile-time warp size. raft::warp_size() is constexpr only in a
   ROCm DEVICE pass (per-arch 64/32); the HOST pass is a runtime query, but clang
   compiles __global__ bodies in BOTH passes, so kernel static_asserts / __shared__
   array bounds / cub::Warp{Scan,Reduce}<,N> failed to parse. Added
   cugraph::warp_size_ct() (new header cugraph/utilities/warp_size_ct.hpp): device
   per-arch, host = CUGRAPH_HOST_WARP_SIZE (-D from CMAKE_HIP_ARCHITECTURES, 64 for
   CDNA / 32 for RDNA). Applied at the ~12 static_assert + WarpScan + array-bound
   sites. Runtime lane uses keep calling raft::warp_size(). Also low_degree_threshold
   in graph_view.hpp (a stored-segment heuristic) fixed to the upstream-tuned 32.
3. transform_e_packed_bool wave-coupling: the kernel mapped 1 lane -> 1 bit of a
   32-bit packed-bool word assuming warp==32. Regrouped by packed_bools_per_word()
   (32) lanes so on wave64 a wavefront spans two words; each group leader extracts
   its 32 ballot bits and writes its uint32. Arch-unified (correct wave32 AND 64).
4. Device-extended-lambda invocability trap (THE prims SFINAE class): clang treats
   a [] __device__ lambda's operator() as device-only, so std::is_invocable_v /
   std::invoke_result evaluated in a host-instantiated template report NOT invocable
   (NVCC reports invocable), dropping the only viable prim overload. Added
   cugraph::hip_compat::is_device_invocable_v / device_invoke_result_t (a
   __host__ __device__ SFINAE test that sees the device operator) in the compat
   header; used in property_op_utils.cuh (cast_edge_op_bool_to_integer,
   intersection_op_result_type, edge_op_result_type) and transform_reduce_v.cuh.
5. raft span.hpp SFINAE defect (surfaces only at C++20): raft's span converting
   ctor uses `class = typename std::enable_if<COND>` WITHOUT `::type`, so it is
   never SFINAE-disabled; std::variant/optional<variant<span...>> resolution then
   hard-errors. Fixed by a one-line-corrected verbatim shadow header in
   src/hip/raft_compat/raft/core/span.hpp on the include path BEFORE raft. Fixes
   ALL variant<span> sites (replaces the per-site std::in_place_type whack-a-mole).
6. cub:: namespace: `#include <hipcub/hipcub.hpp>` + `namespace cub = hipcub;` in
   the compat header (cugraph uses cub::Device*/Block*/Warp* with headers relying
   on transitive cub:: visibility). Cleared ~22 TUs.
7. cuda::std::get / tuple_size / tuple_element bridge for the classic thrust::tuple,
   thrust::pair, AND tuple_of_iterator_references (extended the session-1 get
   bridge with pair + the tuple_size/tuple_element specializations; the latter
   fixed `undefined template cuda::std::tuple_element<I, thrust::tuple<...>>`).
8. is_thrust_tuple / is_thrust_tuple_of_arithmetic specialized for the classic
   thrust::tuple under USE_HIP (the iterator value_type at rocThrust algorithm
   boundaries is thrust::tuple, not cuda::std::tuple) -- unblocked
   allocate_dataframe_buffer for zip value types (shuffle_vertex_pairs_common).
9. rmm flat-path shims for every rmm/mr/device/*.hpp at rmm/mr/ (managed, binning,
   etc.) + a cuda/std/algorithm umbrella shim (libhipcxx ships the underlying
   header but not the umbrella cuGraph includes).
10. Comparator/op functors taking cuda::std::tuple by value/const& -- templated the
    tuple param under USE_HIP so rocPRIM/rocThrust's thrust::tuple binds (the body
    uses the bridged cuda::std::get). Done for ~16 functors incl
    kv_pair_group_id_less/greater_equal_t, value_group_id_less/greater_equal_t,
    is_not_lower_triangular_t, is_not_self_loop_t, compare_*_triangular_edges_t
    (two independent template params -- lhs may be thrust::tuple while rhs is a
    tuple_of_iterator_references), to_lower_triangular_t, hash_*_pair_t,
    check_*_t, cluster_update_op_t, count_updown_moves_op_t, pick_min_degree_t.
11. sssp_impl exemplar for two recurring per-prim classes: local `constexpr`
    invalid_* not implicitly captured by a device lambda under clang (add to the
    capture list), and the zip-write boundary (detail::cugraph_zip_make_tuple).

## Remaining fault classes (the bounded work to finish)

A. rocThrust zip-WRITE boundary across the rest of the prims (`no viable
   overloaded '='`). The pattern is established (sssp_impl.cuh): a device lambda /
   functor returning cuda::std::make_tuple into a zip output reference (classic
   thrust::tuple). Convert each return site to detail::cugraph_zip_make_tuple (or
   thrust::make_tuple under USE_HIP). Plus the paired "local constexpr not
   captured" capture-list fix. Surfaces in the frontier prims
   (extract_transform_if_v_frontier_e.cuh has several at lines ~78/260/388/505)
   and per-prim impls. Mechanical, ~tens of sites; do per failing TU.

B. rocPRIM block_load value-type mismatch: in extract_transform_if /
   per_v_transform_reduce paths rocPRIM block_load static_asserts
   is_convertible<thrust::tuple<...>, int> -- the iterator value_type resolves to
   a thrust::tuple where a scalar is expected (or vice versa). Needs the specific
   iterator's value_type aligned (likely a transform/zip iterator whose declared
   reference/value_type disagrees on ROCm). Diagnose per site; may need a
   proclaim_return_type or an explicit value_type on the cugraph iterator wrapper.

C. cuCollections fork delta (a SECOND repo -- jeffdaily/cuCollections, which we
   own): cuco open_addressing_ref_impl.cuh heterogeneous_value() does `.first` /
   `.second` on the inserted value when it is a (classic) thrust::tuple<K,V> from a
   rocThrust zip deref -- thrust::tuple has no `.first`. cugraph's kv_store insert
   (renumbering) hits this -- it is the ONLY remaining cugraph_common blocker
   (renumber_utils_common_v32/v64). Add a thrust::tuple branch (thrust::get<0>/<1>)
   to the cuco fork's is_cuda_std_pair_like / heterogeneous_value, push the cuco
   fork (--force-with-lease), and re-consume from projects/cuCollections/src.

D. clang two-phase lookup (`template`/`this->`/class-qualified static) across the
   template-heavy prims -- mechanical, surfaced per-TU at compile (a few seen).

E. After cugraph_common + cugraph link: build the in-scope SG gtests, then
   GPU-validate on gfx90a (the real gate). NOT started. Note the tests cmake hits
   a separate concept error in tests/utilities/base_fixture.hpp:99
   (`too many template arguments for concept 'async_resource'`) -- a libhipcxx
   cuda::mr concept-arity mismatch in the test harness, to resolve before the
   gtests build.

## Build hygiene note (session 2)

Build on GCD 2 only: `export HIP_VISIBLE_DEVICES=2` (GCD0 is LEAP). The env.sh in
agent_space/cugraph/env.sh is updated to GCD 2. Incremental keep-going build:
`cmake --build projects/cugraph/build-hip -j8 -- -k 0`. Reconfigure
(`cmake projects/cugraph/build-hip`) after editing cugraph_hip.cmake.

## Deferred surface (documented, NOT regressions)

- spectral_clustering.cu (cuVS spectral + raft spectral/solver; cuVS not ported)
  -> drops cugraph's only cuVS dependency. Plus c_api/legacy_spectral, BALANCED_TEST.
- tree/legacy/mst.cu (raft sparse-solver), linear_assignment/legacy/hungarian.cu
  (raft solver linear-assignment), layout/legacy/force_atlas2.cu (FA2, wave64
  audit pending). Plus MST_TEST/HUNGARIAN_TEST/LEGACY_FA2_TEST.
- cugraph_c (C-API) -- links cugraph_mg; deferred with the MG layer.
- cugraph_mg / cugraph_mtmg (NCCL/UCX multi-GPU); the Python pytest suite.

## Followers (gfx1100/gfx1151) -- record for the delta plan

cugraph's heavy cub::DeviceSegmentedSort (44 sites) + BlockScan/WarpScan lower to
rocPRIM block_radix_sort / lookback_scan_state, which on GFX10+ emit wavefront-
shift DPP that the backend rejects ("wavefront shifts not supported on GFX10+").
Expect per-arch COMPILE exclusions on gfx1100/gfx1151 (the raft gfx1100 wall), NOT
a source change. Does not affect gfx90a (CDNA).
