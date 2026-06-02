# cugraph notes

RAPIDS graph analytics. ROCm/HIP port (Strategy A), lead linux-gfx90a, ROCm 7.2.1.
Upstream rapidsai/cugraph @ e314f1e (VERSION 26.08.00). Fork jeffdaily/cugraph,
branch moat-port. The C++ libcugraph SG (single-GPU) graph-primitive slice is the
target; MG/MTMG (NCCL/MPI), cugraph_c (links the MG lib), and the Python layer are
out of scope. See plan.md for the full scope/deferral rationale.

## Status: IN PROGRESS (blocked, not yet building)

The HIP CMake scaffold, dependency wiring, and the version-skew + first-tier
rocThrust-interop fixes are done and committed to moat-port. The build does NOT
yet produce a library: the remaining work is the rocThrust<->cuda::std tuple
boundary fault family across the prims tree plus a cuCollections-fork delta. This
is a genuine multi-session, multi-repo effort; the fault taxonomy below is
complete and the fixes are mechanical.

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

## Remaining fault classes (the bounded, mechanical work to finish)

A. rocThrust zip-write boundary (class 7) across the rest of the prims/shuffle/
   structure tree: ~10-45 `cuda::std::make_tuple` sites that return into a
   thrust zip output. Each needs the cugraph_zip_make_tuple treatment (or a
   per-site thrust::make_tuple under USE_HIP). Currently blocking
   shuffle_vertices_common, shuffle_vertex_pairs_common,
   shuffle_and_organize_output_common.

B. make_zip_iterator(cuda::std::make_tuple(it1,it2)): rocThrust's zip_iterator is
   parameterized on thrust::tuple-of-iterators; a cuda::std::tuple of iterators
   does not form an iterator (iterator_traits<cuda::std::tuple<int*,int*>> has no
   value_type/reference). Fix: variadic make_zip_iterator(it1, it2) (canonical on
   CUDA too) -- found in renumber_utils and the shuffle code.

C. cuCollections fork delta (a SECOND repo): cuco
   open_addressing_ref_impl.cuh heterogeneous_value() falls into a `.first`/
   `.second` branch when the inserted value is a (classic) thrust::tuple<K,V>
   from a rocThrust zip deref -- thrust::tuple has no `.first`. cugraph's kv_store
   insert (renumbering) hits this. Needs a thrust::tuple branch (thrust::get<0>/
   <1>) in the cuco fork's is_cuda_std_pair_like / heterogeneous_value, then
   re-consume. This is the renumber_utils blocker.

D. clang two-phase lookup (`template`/`this->`/class-qualified static) across the
   template-heavy prims -- mechanical, surfaced per-TU at compile.

E. After cugraph_common + cugraph link: build the ~50 SG gtests, then GPU-validate
   on gfx90a (the real gate). NOT started.

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
