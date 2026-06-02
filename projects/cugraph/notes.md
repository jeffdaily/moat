# cugraph notes

RAPIDS graph analytics. ROCm/HIP port (Strategy A), lead linux-gfx90a, ROCm 7.2.1.
Upstream rapidsai/cugraph @ e314f1e (VERSION 26.08.00). Fork jeffdaily/cugraph,
branch moat-port. The C++ libcugraph SG (single-GPU) graph-primitive slice is the
target; MG/MTMG (NCCL/MPI), cugraph_c (links the MG lib), and the Python layer are
out of scope. See plan.md for the full scope/deferral rationale.

## Status: IN PROGRESS (blocked, SG library compiling 101/124 TUs)

Session 3: cugraph_common now compiles ENTIRELY (the 2 renumber_utils TUs cleared
via the collect_comm zip-construction fix -- the cuco delta was a red herring on
the SG path). The SG cugraph library compiles 101 of 124 .cu (was 85); 52 TUs
remain, all gated by ONE class (A' below): the rocThrust classic-tuple zip iterator
value_type vs cugraph's cuda::std::tuple value algebra in the edge-property-view
stack. The KEYSTONE write-boundary class is SOLVED this session (a thrust_compat
shadow header adding cuda::std::tuple assignment to tuple_of_iterator_references).
The library does not yet link. bfs_sg / pagerank_sg objects DO compile, but no SG
test can run yet because create_graph_from_edgelist (graph construction) is in the
A' set. Kept BLOCKED (no false-port) per the no-thrash rule; the continuation plan
(Session 3 progress + Remaining fault classes A'/B/D/E) below is precise and names
the decision needed.

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

## Session 3 progress (cugraph_common DONE; SG lib 85 -> 101/124 TUs)

Net-positive, zero-regression. All USE_HIP-guarded; CUDA path byte-identical.

Resolved this session:
1. cugraph_common now compiles ENTIRELY (the 2 renumber_utils TUs cleared). The
   actual blocker was NOT the cuco heterogeneous_value `.first` delta (class C,
   never hit on the SG renumber path); it was a zip-iterator CONSTRUCTION site:
   collect_comm.cuh:71 `thrust::make_zip_iterator(cuda::std::make_tuple(it0,it1))`.
   rocThrust's make_zip_iterator takes a classic thrust::tuple or a variadic
   iterator pack, so a single cuda::std::tuple is taken as ONE iterator. Fixed
   generically (see item 4) + the explicit collect_comm / wcc sites converted to
   detail::cugraph_zip_make_tuple.
2. KEYSTONE -- the rocThrust zip-WRITE boundary (the broad class A) solved ONCE,
   not per-site: rocThrust's thrust::detail::tuple_of_iterator_references (a zip
   deref lvalue) defines operator= only for thrust::tuple / thrust::pair /
   thrust::reference, NOT cuda::std::tuple, so the central prim write
   `*(result_value_output + i) = val` (val a cuda::std::tuple) failed everywhere
   (per_v_transform_reduce_e.cuh 218/366/663/877, etc.). Added a SHADOW header
   cpp/src/hip/thrust_compat/thrust/iterator/detail/tuple_of_iterator_references.h
   (byte-identical to rocThrust's except one added operator=(cuda::std::tuple<Us...>)
   that rebuilds a classic thrust::tuple and delegates), wired on the include path
   BEFORE /opt/rocm/include via target_include_directories(... BEFORE ...) in
   cugraph_hip.cmake -- exactly the raft_compat / rmm_compat shim pattern. This
   cleared the entire per-prim `= val` write class in one move. NOTE: the value-
   variable writes (val a variable, not make_tuple) can ONLY be fixed this way;
   detail::cugraph_zip_make_tuple does not apply to them.
3. make_zip_iterator(cuda::std::tuple<Its...>) overload added to the compat header
   (namespace thrust) that splats the cuda::std::tuple into rocThrust's variadic
   make_zip_iterator. Fixes view_concat's `make_zip_iterator(thrust_tuple_cat(...))`
   and all explicit-tuple zip construction generically.
4. dataframe_buffer_{,const_}iterator_type<cuda::std::tuple<Ts...>> trait: declared
   the zip iterator over a classic thrust::tuple under USE_HIP (matching what
   thrust::make_zip_iterator actually returns; a zip_iterator<cuda::std::tuple<...>>
   cannot even instantiate -- rocThrust's tuple_meta_accumulate/value_type are
   undefined for cuda::std::tuple). NOTE this is the source of the remaining wide
   front (class A' below): it pushes a CLASSIC thrust::tuple value_type into the
   property views' ValueIterator, which then disagrees with the cuda::std value_t.
5. cugraph::is_equivalent_value_type_v<A,B> (thrust_tuple_utils.hpp): treats a
   classic thrust::tuple<Ts...> and a cuda::std::tuple<Us...> with the same element
   prefix as equal (= std::is_same on CUDA). Applied at edge_src_dst_property.hpp:51
   static_assert. (More property-view asserts/typedefs need the same treatment --
   see A' below.)
6. to_thrust_iterator_tuple (tuple-valued overload): normalize the classic
   get_iterator_tuple() result to cuda::std::tuple under USE_HIP (new detail helper
   classic_tuple_to_cuda_std_tuple) so view_concat can mix scalar- and tuple-valued
   views in thrust_tuple_cat.
7. thrust_tuple_get functor operator() templated under USE_HIP (class-10 pattern)
   so the classic-tuple zip deref binds (unblocked the shuffle_comm path in
   create_graph_from_edgelist).

## Remaining fault classes (the bounded work to finish)

A'. THE DOMINANT REMAINING CLASS (gates ~all 52 failing SG TUs incl
   create_graph_from_edgelist, common_methods, sssp, the centralities, similarity).
   Root: item 4 above makes the edge property buffers' ValueIterator a classic-
   thrust::tuple zip iterator, whose value_type is thrust::tuple<...,null...> while
   the property's logical value_t is cuda::std::tuple<...>. That mismatch then
   breaks, in cascade: edge_endpoint_property_view_t (value_type typedef + asserts),
   edge_partition_endpoint_property_device_view_t conversion (the
   `no matching conversion ... to edge_partition_dst_input_device_view_t` at
   transform_reduce_e_by_src_dst_key.cuh:592), reduce_op::plus<cuda::std::tuple>
   invoked on classic-tuple args inside rocPRIM/rocThrust reduce_by_key, and
   to_thrust_iterator_tuple's enable_if on single-element zips
   (`zip_iterator<thrust::tuple<cuda::std::tuple<const float*>>>`).
   DECISION NEEDED (pick ONE and apply consistently across the property-view stack):
   (a) Keep value_iterators classic-tuple-based (current item-4 direction) and push
       is_equivalent_value_type through EVERY property-view value_type compare +
       teach edge_partition_endpoint_property_device_view_t + reduce_op to accept
       a classic-tuple value_type (template their tuple args like the class-10
       functors, use the cuda::std::get bridge). Broadest but mechanical.
   (b) Revert item 4 (value_iterators stay cuda::std::tuple) and instead make
       rocThrust's zip machinery instantiate for cuda::std::tuple by shadowing
       zip_iterator_base.h's tuple_meta_accumulate/tuple_of_value_types for
       cuda::std::tuple (deep thrust shadow, narrow blast radius if it works).
   Estimate: option (a) is ~1-2 sessions of per-site templating; (b) is riskier but
   could collapse the whole class. The underlying truth: this rocThrust is classic-
   tuple-based and cugraph 26.08 is fully CCCL/cuda::std::tuple-native, so every
   zip<->value boundary needs a bridge. The write side (keystone item 2) is done;
   the READ/value-type side (A') is the remaining half.

B. rocPRIM block_load value-type mismatch (block_load_func.hpp:301 `assigning to
   'int' from incompatible type tuple_of_iterator_references<float&&>`): a sub-case
   of A' surfacing inside rocPRIM block primitives once the property value_type is
   classic. Should fall out once A' is resolved; recheck per TU.

C. cuCollections fork delta -- NOT on the SG critical path after all (cugraph_common
   compiles without it). Leave the jeffdaily/cuCollections fork as-is unless a
   non-SG path later needs it.

D. clang two-phase lookup (`template`/`this->`/class-qualified static) across the
   template-heavy prims -- mechanical, ~2 TUs left (od_shortest_distances).

E. After the SG lib links: the in-scope SG gtests, then GPU-validate on gfx90a
   (the real gate). NOT reachable yet -- create_graph_from_edgelist (graph
   construction, needed by EVERY test incl BFS/PageRank) is in the A' set, so even
   though bfs_sg / pagerank_sg .cu OBJECTS already compile, no test can build a
   graph until A' is cleared. Tests cmake also hits a libhipcxx cuda::mr
   async_resource concept-arity issue in tests/utilities/base_fixture.hpp:99 to
   clear first.

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
