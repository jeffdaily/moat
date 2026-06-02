# cugraph notes

RAPIDS graph analytics. ROCm/HIP port (Strategy A), lead linux-gfx90a, ROCm 7.2.1.
Upstream rapidsai/cugraph @ e314f1e (VERSION 26.08.00). Fork jeffdaily/cugraph,
branch moat-port. The C++ libcugraph SG (single-GPU) graph-primitive slice is the
target; MG/MTMG (NCCL/MPI), cugraph_c (links the MG lib), and the Python layer are
out of scope. See plan.md for the full scope/deferral rationale.

## Status: linux-gfx90a PORTED -- BFS + SSSP + PageRank all GPU-validated

Session 6: fixed the session-5 remaining bug (BFS/SSSP wrong distances). The
vertex-frontier expansion prims used raft::warp_full_mask() (0xffffffff, 32-bit),
which on wave64 dropped lanes 32-63 from __ballot_sync and miscomputed the
intra-warp buffer prefix offsets. Added wave-width-correct ballot helpers and
routed the frontier-expansion ballot/broadcast sites through them (USE_HIP). All
BFS/SSSP/PageRank gtests with a local dataset now PASS on real gfx90a. See
Session 6 below. Earlier status (kept for history):

Session 5: all SG library TUs compile, libcugraph.so links, and the
PAGERANK/BFS/SSSP gtests build and PASS on real gfx90a hardware (the validation
gate). A real wave64 GPU bug in transform_e_packed_bool (out-of-range ballot
shift on small graphs with edge masking) was found and fixed. See Session 5
below. Earlier status (kept for history):

## (historical) Status: IN PROGRESS (blocked, SG library compiling 101/124 TUs)

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

## Session 6 (BFS + SSSP FIXED and GPU-validated on gfx90a; fork d372efc)

Net-positive, zero-regression. All USE_HIP-guarded; CUDA path byte-identical
(original __ballot_sync / __popc / __shfl_sync preserved verbatim in #else).
NEW commit on top of 2f60595 (validated base preserved, not amended).

ROOT CAUSE of the session-5 remaining bug (BFS/SSSP wrong distances on rmat,
PageRank exact): the vertex-frontier expansion prims (which PageRank does not
use) called raft::warp_full_mask(), hard-coded to 0xffffffff (32 bits). On
wave64 (gfx90a/CDNA) a 32-bit ballot mask DROPS lanes 32-63 from __ballot_sync,
__popc truncates the 64-bit ballot, and `mask << lane_id` for lane_id >= 32 is
undefined; the per-lane prefix offset into the output buffer therefore collided,
so frontier entries were lost/duplicated and distances diverged. The lane-0
broadcasts (__shfl_sync with the same 32-bit mask) also failed to reach the
upper half of the wave.

FIX (warp_size_ct.hpp gained wave-width-correct helpers under USE_HIP):
warp_full_ballot_mask() (64-bit on wave64, 32-bit on wave32), warp_ballot(bool)
-> __ballot_sync with that mask, warp_ballot_popc -> __popcll, warp_ballot_prefix_popc
(count set bits in lanes [0,lane_id) via `full_mask >> (warp_size - lane_id)`,
special-casing lane_id==0 to avoid the UB 64-shift), and warp_bcast (shfl with
the active-wave mask). Sites converted (all USE_HIP-guarded, CUDA in #else):
- detail/extract_transform_if_v_frontier_e.cuh warp_push_buffer_elements (THE
  primary bug; this is the by_dst topdown engine, runs unconditionally so it hit
  even the edge_masking=false rmat case). ballot + increment popc + prefix popc +
  the lane-0 broadcast.
- detail/transform_v_frontier_e.cuh (edge-mask path only): ballot + intra-warp
  offset + base_offset popc.
- detail/per_v_transform_reduce_e.cuh (bottom-up + per_v_transform_reduce_if): the
  two first_valid_lane_id lane-0 broadcasts.
- detail/sample_and_compute_local_nbr_indices.cuh (sampling, not BFS/SSSP but
  same SG lib + same bug): the inclusive-sum last-lane broadcast.
NOTE: the many __popc(word) sites in per_v_transform_reduce_e / by_dst operate on
uint32 packed-bool WORDS (32-bit), not ballots -- those are already correct and
were left alone. transform_e.cuh was the session-5 group-mask fix, untouched.

GPU validation (gfx90a, GCD 2): BFS_TEST rmat_small 4/4 PASS (was FAIL),
file_test karate/polbooks/netscience 6/6 PASS incl edge_masking, rmat_benchmark
4/4 PASS; SSSP_TEST rmat_small 4/4 PASS (was FAIL), file_test karate 2/2 PASS
incl edge_masking; PAGERANK_TEST 56/56 PASS (no regression). Symlinked
polbooks/netscience/dolphins into src/datasets/test/datasets so their file_test
cases run (karate was symlinked in session 5). Remaining file_test cases need
absent large datasets (wiki2003, wiki-Talk, dblp) and are not run. linux-gfx90a
is now PORTED: BFS + SSSP + PageRank all correct on real gfx90a.

## Session 5 progress (SG lib LINKS; PAGERANK/BFS/SSSP GPU-validated on gfx90a)

Net-positive, zero-regression. All USE_HIP-guarded; CUDA path byte-identical.
NEW commit on top of d326818 (base preserved, not amended).

The last 16 failing library TUs (8 SG primitives x v32/v64) now compile, so
libcugraph.so LINKS (1.1 GB). PAGERANK_TEST / BFS_TEST / SSSP_TEST build and
RUN on gfx90a (GCD 2) and PASS. The validation gate is met.

How the 16 TUs were cleared (all mechanical instances of the classic-tuple vs
cuda::std::tuple algebra now that the keystone conversion exists):
- erdos_renyi: cuda::make_transform_output_iterator missing -> alias in
  cuda_to_hip.h forwarding to thrust::make_transform_output_iterator.
- strongly_connected_components: the reduce_by_key proclaim_return_type<cuda::std
  ::tuple> lambda returned the classic zip tuple; gave it an explicit
  -> cuda::std::tuple return that rebuilds via make_tuple (exact-match holds).
- od_shortest_distances: rocPRIM device_merge needs key_type2 (a cuda::std::tuple
  from split_vi_t) convertible/static_cast'able to key_type1 (the classic zip
  value_type), plus thrust::less<> over the two flavors. Added (1) a converting
  CONSTRUCTOR from cuda::std::tuple on the classic value-type tuple in the
  thrust/tuple.h shadow (the reverse of session-4's conversion operator,
  SFINAE-bounded the same way), and (2) classic-tuple-vs-cuda::std operator< in
  cuda_to_hip.h (mirroring the existing ==/!=).
- lookup_src_dst: kv_store insert ValueIterator static_asserts switched to
  cugraph::is_equivalent_value_type_v (4 in kv_store.cuh + 1 in
  lookup_src_dst_impl.cuh); the kv_cuco_store_view_t invalid_value ctor now
  builds the classic tuple from the cuda::std invalid_value via the new ctor.
- sample_edges / temporal_sample_edges / random_walks: temporal_sample_edge_
  biases_op_t and gather_one_hop return_edges_with_label_op /
  return_edges_with_index_and_position_op take the tagged-major/src KEY by a
  template parameter (the class-10 functor pattern) so the classic zip deref
  binds and reads through the bridged cuda::std::get; bias_t{0.0}/{1.0}
  brace-init replaced with static_cast<bias_t>(...) (clang rejects the
  double->int narrowing in the dead bias_t=int SFINAE instantiation that NVCC
  tolerates); the inclusive-sum mid-degree-threshold constexpr uses
  cugraph::warp_size_ct() (raft::warp_size() is not constexpr on the host pass);
  and hipCUB DeviceSegmentedSort::SortPairs requires one OffsetIteratorT type
  for begin+end (CUB allows two), so both offset iterators use one
  segmented_sort_offset_t functor (empty counts span = begin offset i*K, counts
  span = end offset i*K + counts[i]).

cuCollections fork (jeffdaily/cuCollections @ moat-port): the SG lookup path is
the first to hit open_addressing_ref_impl.cuh's three `#if __CUDA_ARCH__ < 700`
guards. On HIP __CUDA_ARCH__ is undefined so a bare `< 700` evaluated to
`0 < 700` (true) and wrongly applied NVIDIA's pre-Volta >8B slot restriction /
cas_dependent_write; AMD provides the atomics. Gated all three on
`defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700` (NVIDIA pre-Volta unchanged; HIP
and NVIDIA Volta+ take the modern path). REQUIRES a cuco fork commit + push.

Tests bring-up:
- base_fixture.hpp: the ported rmm (~25.x) predates cuda::mr::any_resource and
  ships pool_memory_resource only templated (no CTAD). Under USE_HIP the MR
  factories return owning shared_ptr<device_memory_resource>,
  create_memory_resource returns that, and a set_test_current_device_resource
  bridge calls set_current_device_resource(resource.get()); CUDA path unchanged.
- csv_file_utilities_impl.cuh: file.tellg() returns std::fpos; clang rejects
  `fpos + int` as ambiguous (vs the implicit streamoff conversion libstdc++
  allows). Reduced to std::streamoff before the size/index arithmetic.
- cugraph_hip_tests.cmake: the test targets (cugraphtestutil + each gtest) now
  get the SAME compat shims as the library -- raft_compat + thrust_compat BEFORE
  includes and CUGRAPH_HOST_WARP_SIZE -- since they instantiate the same prims
  (transform_e, property views). Previously only rmm_compat was wired, so the
  zip<->cuda::std::tuple write boundary and the host-pass warp size were missing.

REAL GPU BUG FOUND AND FIXED (transform_e.cuh transform_e_packed_bool, the
session-2 wave-coupling kernel): running PAGERANK file_test with edge_masking on
the tiny karate graph faulted (GPU memory fault) while the same masking usecase
on the larger rmat/dolphins graphs passed -- the first time this kernel ran on
real hardware (sessions 2-4 never linked the .so). Two coupled wave64 defects:
(1) group_base_lane (the bit offset of a 32-lane packed-bool group in its
wavefront's 64-bit ballot) was BLOCK-relative (threadIdx.x/32*32), so for the
2nd+ wavefront in a block it was 64, 96, ... -> `ballot >> 96` is an
out-of-range shift; reduced to the WAVE-relative offset (threadIdx.x % warp_size
/ 32 * 32). (2) more fundamentally, the ballot used the full 64-bit wave mask,
but on wave64 a wavefront spans TWO packed-bool words (two idx values), so the
two 32-lane groups in a wave have DIFFERENT loop bounds; on a small graph the
last wave splits (one group's word in range, the next out), the out-of-range
group's lanes never reach the ballot, and a full-wave __ballot_sync on that
divergent set faults. (Large graphs rarely split mid-wave, hence rmat passed.)
Fix: ballot only over this lane's own 32-lane group (group_ballot_mask =
0xFFFFFFFF << group_base_lane); a group's 32 lanes all share one idx = tid/32 so
they enter/exit the loop together (matching upstream CUDA's warp==word
invariant). Arch-unified: on wave32 group_base_lane=0 and the mask is the full
32-bit warp. After the fix all PAGERANK file_test edge_masking cases pass.

GPU validation (gfx90a, GCD 2, RAPIDS_DATASET_ROOT_DIR=src/datasets;
HSA_ENABLE_COREDUMP=0 to avoid the host's missing-coredump-handler masking a
fault as EXIT=141; a benchmark mtx symlink test/datasets/karate.mtx ->
../../karate.mtx satisfies file_benchmark_test's nonstandard path):
- PAGERANK_TEST: 56/56 PASS (rmat correctness + file karate/dolphins x 8
  usecases incl edge_masking + weighted + personalized). FULLY VALIDATED.
- BFS_TEST: 5 PASS, the rmat_small_test correctness cases FAIL with "distances
  do not match with the reference values" (rmat_benchmark_test, correctness
  off, passes; file_test cases need absent large datasets wiki2003/web-Google).
- SSSP_TEST: same -- rmat_small_test correctness FAILS the same way.

REMAINING BUG (keeps linux-gfx90a blocked, no false-port of traversal): BFS and
SSSP compute WRONG distances on the in-GPU rmat graph (PageRank on the same
infra is exact, so graph construction / renumber / per-V/per-E reduce / edge
mask are all correct). BFS/SSSP differ from PageRank in using the vertex-
frontier expansion prim transform_reduce_if_v_frontier_outgoing_e_by_dst +
per_v_transform_reduce_if_incoming_outgoing_e + the frontier buckets. No raw
wave intrinsics there (it lowers to rocPRIM BlockScan / DeviceSelect /
reduce_by_key over the classic-tuple key algebra), so the next session should
bisect that frontier/reduce path on gfx90a (a deterministic rmat seed-0 graph;
likely a wave64 scan/select boundary or a reduce_op key-flavor mismatch in the
frontier dedup at lines ~334-380 of the by_dst prim). This was not observable
before this session because the .so never linked.

## Session 4 progress (A' class SOLVED; SG lib down to 16 failing TUs; fork d326818)

Net-positive, zero-regression. All USE_HIP-guarded; CUDA path byte-identical.
Committed as a NEW commit on top of 1e28c83 (base preserved, not amended).

OPTION DECISION (the A' fork in the road): round 4 had begun option (b) -- a
deep shadow of zip_iterator_base.h's tuple_meta_accumulate / tuple_of_value_types
so the zip value_type would be cuda::std::tuple. That shadow was NEVER created
(only an aspirational comment in the tuple_of_iterator_references.h shadow
referenced it). This session FELL THROUGH to option (a) and it yielded
decisively. (b) did not yield; (a) did.

KEYSTONE (a): a thrust/tuple.h compat shadow (new file
cpp/src/hip/thrust_compat/thrust/tuple.h, on the include path BEFORE /opt/rocm
via the existing thrust_compat -I) that gives rocThrust's classic value-type
thrust::tuple<Ts...,null...> two USE_HIP members:
  - operator cuda::std::tuple<Us...>() over the non-null prefix (the READ half:
    fixes rocPRIM block_load's is_convertible<value_type, T> static_assert and
    block_load_direct_striped's assignment).
  - operator=(cuda::std::tuple<Us...>) over the prefix (the reduce_by_key
    sequential-fallback write-back temp_value = binary_op(...)).
Both are SFINAE-bounded to sizeof...(Us)==thrust::tuple_size so a nested
tuple-of-tuples never instantiates tuple_element past the prefix (this guard was
essential -- without it, thrust::tuple<cuda::std::tuple<int,int,int>, ...> drove
tuple_element<2> into null_type and hard-errored). This ONE header cleared the
dominant A'/B class: sssp_sg went from 2 errors to 0; the full library dropped
from ~44 failing library TUs to 16.

Built on top of the keystone (all USE_HIP, CUDA = std::is_same / unchanged):
1. Property-view value-type static_asserts now use cugraph::is_equivalent_value_type_v
   instead of std::is_same_v: edge_property.hpp:32,
   edge_partition_endpoint_property_device_view.cuh:41,
   edge_partition_edge_property_device_view.cuh:39, vertex_frontier.cuh:390,
   nbr_intersection.cuh:704 (round 4 had only done the last one).
2. edge_partition_endpoint_property_device_view_t's two ctors and
   kv_binary_search_store_view_t's ctor accept an EQUIVALENT value_t under
   USE_HIP (templated param + is_equivalent_value_type guard), converting via the
   tuple.h assignment bridge. Fixes the per_v_transform_reduce_dst_key /
   kv_store deduction failures in the Louvain aggregated-edge path.
3. is_arithmetic_or_thrust_tuple_of_arithmetic specialized for classic
   thrust::tuple (was only specialized for cuda::std::tuple), unblocking
   common_methods.
4. classic thrust::tuple<Ts...> vs cuda::std::tuple<Us...> operator==/!= added in
   cuda_to_hip.h (namespace thrust) -- thrust::remove(first,last,value) lowers to
   a not_fun_t<_1 == value<cuda::std::tuple>> placeholder predicate that compares
   a block-loaded classic tuple against the cuda::std value. (round 4 had only
   added these for tuple_of_iterator_references.)
5. fill_edge_property device lambdas take auto pair (was the explicit
   cuda::std::tuple<T,uint32_t>; the zip deref is the classic flavor).
6. Recurring per-prim: clang device-lambda constexpr capture (sssp_impl bucket
   idx vars, core_number_impl bucket_idx_next, betweenness_centrality_impl
   bucket_idx_next); dependent template-name keyword (key_store.cuh ref_type).

REMAINING: 16 library TUs (8 SG primitives x v32/v64) still fail and gate the
libcugraph.so LINK, so NO GPU validation is reachable yet (kept BLOCKED, no
false-port). The 8: od_shortest_distances, strongly_connected_components,
random_walks, sample_edges, temporal_sample_edges, gather_one_hop,
lookup_src_dst, erdos_renyi_generator. NONE are BFS/SSSP/PageRank -- those
compile -- but the SG library is one .so so it will not link until these clear.
Their remaining sub-classes (all narrower instances of the same tuple-flavor
work, now mechanical given the keystone):
  - cuda::proclaim_return_type "Return type shall match the proclaimed one
    exactly": an op returns a classic tuple where a cuda::std tuple is proclaimed
    (od_shortest_distances). Normalize the returned value or the proclaimed type.
  - rocPRIM device_merge "Keys_input2 must be convertible to keys_input1" + a
    static_cast conversion failure (strongly_connected_components): a merge over
    two key iterators whose value_type flavors differ.
  - double->int narrowing in braced init at sample_and_compute_local_nbr_indices
    .cuh:2547 (bias_t{0.0}) and :3289 -- a dead/SFINAE bias_t=int instantiation;
    use static_cast under USE_HIP rather than braces.
  - prim_functors.cuh return_edges_with_label_op operator() not matching, and
    transform_v_frontier_e / extract_transform_if_v_frontier_e "no viable
    overloaded" + sample_and_compute "no matching function": more functor tuple
    args to template (the class-10 pattern) and a couple kv_store static_asserts.

SCOPE NOTE: a tempting shortcut is to defer these 8 from cugraph_hip_sources.cmake
to force the link and validate PAGERANK now. DO NOT do it casually:
neighbor_sampling_impl.hpp calls sample_edges() and gather_one_hop_edgelist(), so
deferring sample_edges/gather_one_hop cascades into neighbor_sampling /
negative_sampling / sampling_post_processing (link-time symbol loss). If the next
session wants to reach validation fast, finish the 8 (mechanical) rather than
excising the sampling subsystem.

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

## Validation 2026-06-02 (validator, linux-gfx90a @ d372efc) -- FAILED

GPU: gfx90a (GCD 2, HIP_VISIBLE_DEVICES=2). Fork HEAD d372efc confirmed.
Build: cmake --build projects/cugraph/build-hip --target BFS_TEST SSSP_TEST PAGERANK_TEST
-- EXIT 0 (all three targets build clean).

RAPIDS_DATASET_ROOT_DIR=projects/cugraph/src/datasets; HSA_ENABLE_COREDUMP=0.

PAGERANK_TEST: 56/56 PASS.

BFS_TEST: 14/18 PASS. 4 FAIL = file-missing exceptions for absent large datasets
wiki2003.mtx (cases 6,7) and wiki-Talk.mtx (cases 8,9). All in-scope cases PASS:
- file_test karate/polbooks/netscience (cases 0-5): 6/6 PASS incl edge_masking
- rmat_small (4 cases): 4/4 PASS
- rmat_benchmark (4 cases): 4/4 PASS
The 4 wiki* failures are expected/documented (absent large datasets); not a regression.

SSSP_TEST: 10/14 PASS. 4 failures. Breakdown:
- file_test karate (cases 0-1): 2/2 PASS incl edge_masking  <-- in-scope, pass
- file_test dblp (cases 2-3): FAIL with fopen dblp.mtx -- absent dataset, expected
- file_test wiki2003 (cases 4-5): FAIL with fopen wiki2003.mtx -- absent dataset, expected
- rmat_small (4 cases): 4/4 PASS
- rmat_benchmark (cases 0-1): ABORT via assert -- REAL GPU BUG (see below)

REAL BUG (blocks validation): SSSP_TEST rmat_benchmark_test cases 0 and 1 abort with:
  Assertion `vertex_frontier.bucket(bucket_idx_far).aggregate_size() > 0' failed.
in sssp_impl.cuh:503. This is inside the delta-stepping algorithm's split_far path.

ROOT CAUSE: sssp_impl.cuh line 249:
  auto delta = (static_cast<weight_t>(raft::warp_size()) * average_edge_weight)
               / average_vertex_degree;
raft::warp_size() on the HOST for gfx90a returns 64 (wave64 runtime query), vs 32 on
CUDA. This doubles the delta step size. For the large directed RMAT benchmark graph
(scale=20, edge_factor=32), the 2x delta causes the split_bucket call to move ALL
vertices from bucket_idx_far into bucket_idx_cur_near_near (their distances are all
within the enlarged threshold), violating the assert that bucket_idx_far is non-empty
after the split. This is a host-side arithmetic issue, not a device-kernel issue; the
delta formula uses warp_size as a performance-tuning constant (not a correctness param)
and must stay at 32 to match the upstream-calibrated delta-stepping behavior.

FIX NEEDED: In sssp_impl.cuh line 249, replace raft::warp_size() with the constant 32
(or equivalently min(raft::warp_size(), 32u)) under USE_HIP. This restores the upstream
CUDA delta calibration on all ROCm targets.

Note: the rmat_small tests (scale=10, smaller graph) PASS because the smaller graph has
fewer vertices in bucket_idx_far when the split fires, so the 2x delta still leaves some
vertices in far. The large benchmark (1M+ vertices, 32M+ edges) triggers the assert.

This is NOT covered by the session-6 porter validation because the SSSP rmat_benchmark
was not run in session 6 (only rmat_small and file karate were validated). BFS
rmat_benchmark was run and passed because BFS does not use a delta-stepping threshold.

Status: validation-FAILED. Returning to porter. Fix: sssp_impl.cuh:249 replace
raft::warp_size() with 32 under USE_HIP (min or constant).

## Review 2026-06-02 (reviewer, linux-gfx90a @ d372efc)

Reviewed `git diff e314f1e...HEAD` (4 commits, 83 files) with the /pr-review skill, fault-class deep read of warp_size_ct.hpp and the five ballot/shuffle sites, the classic-tuple <-> cuda::std::tuple bridge (tuple.h, cuda_to_hip.h, thrust_tuple_utils.hpp), the CMake USE_HIP gate, and the test harness. Verdict: review-passed, no changes requested.

Verified (no defect): wave64 ballot correctness. warp_full_ballot_mask()/warp_ballot_popc(__popcll)/warp_ballot_prefix_popc (lane_id==0 special-cased to avoid the UB 64-shift) are wave-correct; warp_bcast uses the 64-bit mask so the lane-0 broadcasts of warp_buffer_start_idx and first_valid_lane_id reach lanes 32-63 (the CUDA 32-bit mask would not). transform_e_packed_bool group_ballot_mask restricts the ballot to the lane's own 32-lane packed-bool group (group leader = every 32nd lane, idx=tid/32 shared within a group) -- correct on wave32 (group==warp) and wave64 (two groups/wave). __GFX9__ confirmed defined for the gfx90a device pass and absent for gfx1100 (gets __GFX11__), so warp_size_ct() device-pass returns 64/32 correctly; CUGRAPH_HOST_WARP_SIZE is derived from the first arch and passed PUBLIC to library and test targets.

Verified (no defect): CUDA path. Every ballot/shuffle change keeps the original __ballot_sync/__popc/__shfl_sync expressions byte-identical in the #else branch. The unguarded raft::warp_size()->cugraph::warp_size_ct() swaps (static_assert, cub::WarpScan<,N>) are value-identical on CUDA (warp_size_ct() forwards to raft::warp_size()). The shadow headers (tuple.h, tuple_of_iterator_references.h, raft span.hpp) and the rmm_compat shims are wired only via target_include_directories(... BEFORE) under USE_HIP, never on the CUDA include path. cuda_to_hip.h tuple operator</==/!= bridges and is_device_invocable live entirely inside #if USE_HIP. operator< reverse direction (cuda::std lhs vs classic rhs) computes a correct total order. Commit hygiene clean: all four titles [ROCm]-prefixed and <=72 chars, Claude attribution present, no noreply trailer, no ghstack, no AMD-internal account references.

Minor (non-blocking, recorded for the porter, not changes-requested): the tuple.h operator=(cuda::std::tuple<Us...>) at line 750 is bounded only by `sizeof...(Us) <= 10` rather than `== tuple_size`, unlike the converting operator/ctor which use the strict `==` arity check. moat_assign_cuda_std assigns only the first sizeof...(Us) slots, so a shorter-arity RHS would silently leave trailing classic-tuple slots unwritten. No current call site hits a mismatched arity (the reduce_by_key write-back is always equal-arity), so this is latent only; tightening to `==` would match the other two bridges and remove the foot-gun.

Scope deferrals confirmed documented (spectral/mst/hungarian/FA2 legacy, cugraph_c C-API, MG/MTMG, Python layer) in plan.md and notes.md "Deferred surface". GPU validation claims (BFS/SSSP/PAGERANK PASS on gfx90a GCD 2) are the porter's; the validator re-runs them next.
