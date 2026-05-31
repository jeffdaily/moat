# cudf notes (ROCm/HIP port)

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated
RAPIDS). `depends_on: [rmm, cuCollections]` (both COMPLETED/ported). cudf is
itself a dependency of cugraph and cuml.

Pinned upstream: tag `v25.08.00` (commit 6cea374, in `projects/cudf/src`,
gitignored), matching the rmm/raft/cuCollections v25.08 line.

Fork: jeffdaily/cudf @ `moat-port`. The port is a standalone HIP CMake build
reached by a top-of-CMakeLists `if(USE_HIP) include(cmake/hip/cudf_hip.cmake)
return()` guard; the CUDA build is untouched.

## State: PORTING on linux-gfx90a -- scoped core COMPILES + GPU-validates (83 TUs)

A scoped libcudf core (83 TUs) compiles for gfx90a and links libcudf.so with a
real gfx90a code object. A focused GPU gtest passes 6/6 on MI250X (stable over
20 repeated runs): the 4 wave64 null-mask tests PLUS 2 new tuple-key radix-sort
tests (SortRadixFloatTupleKeyShim, SortedOrderRadixFloatTupleKeyShim). This
supersedes the prior "BLOCKED on cuCollections" verdict: cuCollections is now
ported (jeffdaily/cuCollections @ moat-port) and the foundational column core
(which the prior run could not compile at all) builds clean. The remaining work
to a FULL core is the rest of the type-erased dispatch link-closure (the
cuco-backed algorithms) + the JIT phase, not a fundamental blocker.

### This session cleared the 3 documented "next walls" (rocPRIM tuple, array, ballot)

- **rocPRIM radix sort over cuda::std::tuple keys -- SOLVED with a reusable
  shim, NO rocPRIM modification.** cudf's float fast-sort
  (sort/sort_radix.cu SortKeys, sort/sorted_order_radix.cu SortPairs) packs
  {int32 nan-bias, float} into a struct and sorts with a "decomposer" returning
  `cuda::std::tuple<size_type&, F&>`. hipCUB forwards the decomposer to rocPRIM,
  whose radix codec for a CUSTOM key only recognizes `rocprim::tuple`
  (`rocprim/type_traits.hpp` codec<Key,Desc,false>: `is_tuple_of_references` /
  `for_each_in_tuple` / `rocprim::tuple_size` / the internal
  `extract_digit_from_key_impl(const rocprim::tuple<Args...>&)` ALL pattern-match
  rocprim::tuple; `type_traits_functions.hpp` `is_tuple_of_references` primary
  even `static_assert(sizeof(T)==0,"only implemented for rocprim::tuple")`). Fix:
  `cpp/include/cudf/detail/utilities/hip/radix_tuple_key.cuh` -- a decomposer
  ADAPTOR that calls the user decomposer then re-`rocprim::tie`s the SAME lvalue
  references into a `rocprim::tuple<Refs...>` (the references point into the live
  key, safe across the wrapper). rocPRIM's existing, validated codec then drives
  the sort. Applied at the 2 float-path decomposer sites under `#if USE_HIP`
  (CUDA byte-for-byte unchanged). The header is library-agnostic (knows only
  cuda::std::tuple + rocprim::tuple, not cudf) -- LIFTABLE VERBATIM into
  cuCollections retrieve_all and raft, which hit the identical gap. GPU-proven:
  the standalone shim sorts 100000 floats correctly via
  hipcub::DeviceRadixSort::SortKeys, and the in-tree GPU gtest sorts FLOAT32
  columns asc+desc (sizes 1..100000) == host std::sort. So the rocPRIM tuple-key
  wall is NOT infeasible -- the shim clears it without touching rocPRIM, and the
  per-project packed-key fallback is NOT needed.
  - sorted_order_radix.cu ALSO needed a second fix: its zip-iterator transform
    functor `float_to_pair_and_seq` returned `cuda::std::pair<float_pair,size_type>`,
    and rocThrust's zip-iterator reference (a thrust::tuple) does not assign from
    a cuda::std::pair (CCCL thrust does; rocThrust does not). Changed to return
    `thrust::tuple` on HIP (USE_HIP-guarded).
- **cuda::std::array<unsigned,100> -- NOT a real wall (the prior finding was
  stale).** strings/case.cu uses `__constant__ cuda::std::array<uint32_t,100>`
  with an aggregate initializer; the type IS fully supported by the current
  vendored ROCm/libhipcxx (a standalone __constant__ + thrust::binary_search
  probe compiles). case.cu simply never `#include`d `<cuda/std/array>` (it relied
  on CCCL pulling it transitively, which libhipcxx does not). Fix: add the
  explicit include (correct on CUDA too -- same fault class as the cuda/std/bit
  missing-transitive-include).
- **wave64 ballot in strings/copying/concatenate.cu -- converted** to the
  tile-relative `ballot_32` helper (warp_primitives.cuh), identical arch-unified
  treatment as the in-scope valid_if sites. `fused_concatenate_string_offset_kernel`
  dropped the carried `active_mask` dance (tiled_partition<32>::ballot() ballots
  only active lanes). COMPILES; this specific kernel is not yet exercised by the
  GPU test (covered by the identical, GPU-validated valid_if pattern).

### Validation gotcha burned this session (cost ~hours): OOB in the TEST, not the port

The first version of the sort gtest crashed nondeterministically (~70% SIGSEGV/
SIGABRT) -- but ONLY in the gtest binary, never in any standalone libcudf-linked
repro (sort_radix/sorted_order_radix called hundreds of times = clean). Root
cause was a HEAP BUFFER OVERFLOW in the TEST's own input helper: `mixed_floats(n)`
wrote `h[10]/h[20]/h[100]/h[200]` unconditionally, which is OOB of the
std::vector for the small sizes (n=1,33,64,65) in the size loop, smashing the
host heap -> later malloc/free or the ROCm-runtime atexit finalizer crashed.
The cudf sort + shim were CORRECT throughout (which is exactly why every
standalone repro that built its input array correctly was clean). LESSON: when a
GPU test crashes intermittently in libamdhip64/libhsa host alloc or
__cxa_finalize but a standalone repro of the same library calls is clean,
suspect the TEST harness (an OOB host write, a bad index) before the port; guard
every fixed index against the parametrized size. Do NOT paper over it with an
rmm pool or _exit() (both "fixed" the symptom by changing heap layout and masked
the real bug).

## MACRO-CASCADE VERDICT (jeff's question): the 1-line fix DID clear it

Ground truth confirmed: `CUDF_HOST_DEVICE` is DEFINED once
(`cpp/include/cudf/types.hpp:23`, gated `#ifdef __CUDACC__`) and USED at 331
sites across 43 files. Changing that ONE definition to
`#if defined(__CUDACC__) || defined(__HIPCC__)` restored `__host__ __device__`
at all use sites and **cleared the entire column_device_view.cu -> string_view
"call to __host__ from __device__" cascade (the 71-header wall) in one edit**.
Verified by compiling `cpp/src/column/column_device_view.cu` standalone as HIP:
before, the cascade; after the 1-line fix, only a SEPARATE, smaller issue
remained -- `type_dispatcher.hpp` `base_type_to_id` explicit specializations
(the `CUDF_TYPE_MAPPING` macro + the `char` specialization) are declared
WITHOUT `CUDF_HOST_DEVICE` while the primary now HAS it; clang (unlike nvcc)
rejects the host/device-attribute mismatch ("no function template matches").
Adding `CUDF_HOST_DEVICE` to those 2 specialization definitions cleared it, and
column_device_view.cu then compiled with ZERO errors. So: the macro cascade was
real and the 1-line root fix dissolved it; the prior "46 guard site cascade"
framing was an overcount (it conflated the single definition with its 331 uses
plus the unrelated nv-pragma guards). Do NOT blanket-`#define __CUDACC__` (the
prior dead-end: it wrongly activates CUDA template paths in type_dispatcher and
elsewhere).

### How the __CUDACC__ / __CUDA_ARCH__ sites were actually handled

- `CUDF_HOST_DEVICE` / `CUDF_KERNEL` definition gate (types.hpp:19): root fix
  `|| defined(__HIPCC__)`. This is the whole cascade.
- Real `__device__`-code `#ifdef __CUDACC__` guards (4 sites, genuine device
  functions): `|| defined(__HIPCC__)` so the device fn is declared on HIP --
  `utilities/bit.hpp` (set_bit), `column/column_device_view_base.cuh`
  (set_valid), `src/jit/span.cuh` (is_valid_nocheck/is_valid).
- NVCC-only `#pragma nv_exec_check_disable` guards (the OTHER ~11 `#ifdef
  __CUDACC__` sites in type_dispatcher.hpp / column_view.hpp / aggregation.hpp /
  device_scalar.hpp / quantiles_util.hpp): LEFT `__CUDACC__`-only. clang neither
  has nor needs that pragma; adding `|| __HIPCC__` would make clang choke on an
  unknown pragma. This per-site discrimination is exactly why the blanket define
  fails.
- `__CUDA_ARCH__` (30 sites, all presence/absence -- NO numeric `>= N` gates in
  cudf): the raft fault-class-1 fix -- define `__CUDA_ARCH__ 800` in the HIP
  DEVICE pass only (compat header, guarded by `__HIP_DEVICE_COMPILE__`). Every
  `#ifndef __CUDA_ARCH__` (host-only error-throw branch) and `#ifdef
  __CUDA_ARCH__` (device direct-memory branch) then selects correctly in each
  pass. No NVIDIA intrinsics are newly activated (cudf has no `__CUDA_ARCH__ >=
  N` comparisons), so no per-site `&& !defined(USE_HIP)` guards were needed
  (unlike raft, which had __dp4a/__nanosleep/PTX behind numeric gates).

## Architecture of the HIP build

- `cpp/CMakeLists.txt`: 6-line `option(USE_HIP)` + guard at the very top.
- `cpp/cmake/hip/cudf_hip.cmake`: standalone HIP build. find_package(hip),
  find_package(rmm) (carries vendored libhipcxx cuda::std/cuda::mr +
  rapids_logger + the rmm compat force-include + USE_HIP/__HIP_PLATFORM_AMD__/
  LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE defs), add_subdirectory the
  ported cuCollections with USE_HIP (gives cuco::cuco header-only INTERFACE; its
  BUILD_TESTS is force-OFF so cudf's BUILD_TESTS does not pull cuco's Catch2
  tests), NVTX3 from a vendored header dir, gtest from conda. Generates
  version_config.hpp + logger_macros.hpp (rapids-logger template; logger ns is
  `cudf::default_logger()`, NOT `cudf::detail::`). Marks the scoped sources
  LANGUAGE HIP, force-includes the cudf compat header, links rmm::rmm +
  cuco::cuco + hip::host.
- `cpp/cmake/hip/cudf_hip_sources.cmake`: the curated in-scope source list (the
  knob that grows the core). Files NOT listed are scoped out, not deleted.
- `cpp/cmake/hip/cudf_hip_tests.cmake`: builds the focused GPU gtest.
- `cpp/include/cudf/detail/hip/cuda_to_hip.h`: the cudf compat header
  (force-included on every HIP TU). Adds the 4 cuda* runtime symbols rmm's
  compat does not cover (cudaGetSymbolAddress, cudaPeekAtLastError,
  cudaStreamDefault, cudaOccupancyMaxActiveBlocksPerMultiprocessor) and the
  `__CUDA_ARCH__ 800` device-pass define. Most of cudf's cuda* runtime surface
  is already aliased by rmm's compat (force-included first via rmm::rmm).
- `cpp/hip_compat/`: forwarding shims on the HIP include path (BEFORE) --
  `cuda_runtime.h`/`cuda_runtime_api.h`/`cuda.h` -> the compat header; `cub/*`
  -> hipCUB (cub.cuh + the 12 specific cub headers cudf includes, each
  `namespace cub = hipcub;`); `cooperative_groups.h` + `cooperative_groups/
  reduce.h` -> HIP CG + a tile-relative cg::reduce shim (copied from the cuco
  port). On NVIDIA this dir is absent so the real toolkit headers win.
- `cpp/include/cudf/detail/utilities/hip/warp_primitives.cuh`: the wave64
  ballot helper (below). Included by `detail/utilities/cuda.cuh` under USE_HIP.

## Fault classes fixed (all arch-unified; NVIDIA path byte-for-byte unchanged)

1. **wave64 null-mask ballot** (the central correctness gate). cudf builds a
   32-bit validity word per warp via `__ballot_sync(0xFFFF'FFFFu, pred)` and has
   the lane-0 leader write it to `output[word_index(i)]`. On AMD this (a) does
   not COMPILE (HIP `__ballot_sync` static_asserts a 64-bit mask) and (b) is
   semantically wrong on a 64-lane wavefront (64 lanes straddle two 32-bit
   words). cudf's bitmask granularity is genuinely 32 bits/word, so the fix
   KEEPS the logical warp at 32 lanes: `ballot_32(pred)` uses
   `cg::tiled_partition<32>(block).ballot(pred)` -- tile-relative on any wave
   width (the cuCollections/gsplat lesson), so the two 32-lane tiles of a wave64
   wavefront each produce their own 32-bit word matching `word_index`. Routed at
   all in-scope sites: `detail/valid_if.cuh` (x2), `detail/copy_range.cuh`,
   `detail/copy_if_else.cuh`, `src/copying/concatenate.cu` (x2),
   `src/replace/replace.cu`, `src/replace/nulls.cu`, each `#if defined(USE_HIP)`
   guarded. `warp_size` stays 32 (do NOT bump to 64 -- it is the bitmask word
   width, not the hardware wave width). GPU-validated: see Test results.
2. **HIP cooperative_groups gaps**: `grid_group::block_rank()` does not exist in
   HIP CG -> replaced (USE_HIP) with the explicit row-major linear block index
   (`blockIdx.x + blockIdx.y*gridDim.x + ...`), identical to CUDA's value
   (null_mask.cu set_null_masks_kernel). `cg::reduce`/`cg::plus` absent -> the
   tile-relative shfl_xor shim (hip_compat/cooperative_groups/reduce.h).
3. **clang-vs-nvcc strictness** (clang stricter): (a) hash-functor explicit
   specializations must match the constexpr primary's constexpr-ness
   (murmurhash3_x86_32/x64_128, xxhash_32/64 -- added `constexpr`; the two
   unreachable list_view/struct_view specializations also need a
   `-Winvalid-constexpr` pragma + an unreachable `return`, HIP-only); (b)
   `typename` on dependent type names (`cast_ops.cu`
   `cuda::std::chrono::floor<typename TargetT::duration>`); (c) a nested type's
   forward-declaration access must match its definition (list_device_view.cuh
   pair_accessor/pair_rep_accessor -- made the definitions public to match the
   public forward decls); (d) thrust::iterator_facade hooks called from
   `__host__ __device__` helpers must themselves be host+device
   (`row_operators.cuh` strong_index_iterator increment/advance/equal/
   dereference/distance_to -> CUDF_HOST_DEVICE); (e) gather.cuh bounds_checker
   ctor was `__device__`-only but thrust constructs it host-side ->
   CUDF_HOST_DEVICE.
4. **missing transitive includes that only libhipcxx exposes**: `cuda/std/bit`
   for `cuda::std::popcount`/`countl_zero` (null_mask.cuh, null_mask.cu --
   CUDA's CCCL pulled it transitively, libhipcxx does not); assert.cuh for
   `CUDF_UNREACHABLE` (integer_utils.hpp, now reachable in the device pass since
   __CUDA_ARCH__ is defined there). All are correct on CUDA too (explicit deps).
5. **missing cub headers**: shimmed (block_load, device_histogram,
   device_merge_sort, device_radix_sort, device_segmented_sort, warp_reduce in
   addition to the initial set).

## What COMPILES (the 83-TU scoped core)

(+4 over the prior 79: sort/sort_radix.cu, sort/sorted_order_radix.cu,
strings/case.cu, strings/copying/concatenate.cu -- the 3 cleared walls above.)


column, table, scalar, bitmask, utilities (cuda/grid_1d/stream_pool/traits/
type_dispatcher/type_checks/host_memory/prefetch/logger/...), unary
(cast/math/nan/null ops), copying (concatenate/copy/copy_range/gather/get_element/
reverse/sample/scatter/shift/slice/split/segmented_shift/purge_nonempty_nulls),
filling (fill/sequence), replace (replace/nulls), search (contains_scalar/
search_ordered), sort (sort/sort_column/stable_sort/stable_sort_column/is_sorted/
rank), and the foundational nested-type column views + factories + a slice of
their detail ops (strings/lists/dictionary/structs column_view + factories,
strings copy_range/shift/fill/utilities/concatenate(combine)/replace_nulls/
find_replace, lists copying gather/concatenate, dictionary add_keys/search/
encode/replace/concatenate, structs concatenate). libcudf.so links (shared libs
tolerate the residual undefined dispatch symbols).

## Scoped OUT (guarded via the source list; files NOT deleted)

- **all of src/io** (89 files: cuIO csv/json/parquet/orc/avro/text + comp) --
  home of the unportable **nvCOMP** (NVIDIA proprietary binary) and **KvikIO**
  (GPUDirect Storage). Cleanly separable (0 nvCOMP includes outside io/).
- **jitify/NVRTC JIT** kernels (binaryop-JIT, transform-JIT, rolling-JIT). The
  jitify->hiprtc plan is a separate effort (jitify_hiprtc_plan.md, owned
  elsewhere). `binaryop/binaryop.cpp` pulls jitify (15 includes).
- **the broad algorithm modules** (groupby, join, reductions, stream_compaction
  distinct/unique, hashing top-level, lists/strings/dictionary/structs full
  algorithm sets, text, interop-to-arrow, transform, rolling, datetime, etc.) --
  not yet in the source list. These are the cuco-backed and heavy-template
  algorithms; bring-up is incremental from here.

## The next concrete walls (precise blocked_reason material)

Walls 1-3 from the prior session (rocPRIM tuple-key radix sort,
cuda::std::array, the strings-concatenate ballot) are CLEARED this session (see
"This session cleared the 3 walls" above). The reusable rocPRIM tuple-key shim
(radix_tuple_key.cuh) is the key result -- it dissolves the same gap for cuco
retrieve_all and raft too. What remains to a FULLY-linking standalone libcudf
and a broad GPU gtest suite:

1. **The remaining type-erased dispatch link-closure (the broad algorithm
   modules).** `column.cu`/`column_factories.cu` reference per-type detail ops
   that transitively pull strings/lists/dictionary/structs FULL algorithm sets,
   plus the cuco-backed algorithms (join/groupby/hashing/stream_compaction
   distinct/reductions). libcudf.so currently has ~209 undefined deferred-dispatch
   symbols (e.g. `dictionary::detail::match_dictionaries`,
   `experimental::row::lexicographic::preprocessed_table::create`,
   `lists::detail::build_lists_child_column_recursive`); the scoped core links
   the .so fine (shared libs tolerate them) and the GPU tests never call them,
   but a FULL link needs these modules brought up. cuco is now available, so the
   cuco-backed ones are reachable; bring-up is incremental (add modules to
   cudf_hip_sources.cmake, fix the per-TU clang/wave64/library issues, extend the
   GPU gtest). Expect more wave64 ballots in the broader kernels (json_path.cu,
   merge.cu, rolling.cuh, conditional_join_kernels.cuh) -- same ballot_32
   treatment.
2. then the **JIT** wall (binaryop/transform/rolling) -- the jitify->hipRTC plan
   is written (jitify_hiprtc_plan.md) and the hipRTC PoC is proven; it is a
   separate phase that depends on this core compiling (done).

## rmm-as-a-dependency: WORKED (refreshed)

The cudf-rmm dep MUST be the CURRENT jeffdaily/rmm @ moat-port (now d22baff),
NOT the older 1473ffc: the older install exported a malformed force-include path
(`<prefix>//rmm/...`, missing `include/`, because CMAKE_INSTALL_INCLUDEDIR was
empty at export) that broke every cudf HIP TU. d22baff uses
`$<INSTALL_PREFIX>/include/rmm/...`. Recipe (per projects/rmm/notes.md):

```
# rmm dep clone updated to fork HEAD, rebuilt+installed into _deps/cudf-rmm/install
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S _deps/cudf-rmm/src/cpp -B _deps/cudf-rmm/build -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/cudf-rmm/install
cmake --build _deps/cudf-rmm/build --target install -j16
```

## Vendored deps a build needs (stable locations)

- rmm install: `_deps/cudf-rmm/install` (jeffdaily/rmm @ moat-port, built above).
- cuCollections: `_deps/cudf-cuco/src` (`git clone --depth 1 -b moat-port
  https://github.com/jeffdaily/cuCollections`). Header-only; passed via
  `-DCUDF_CUCO_SOURCE_DIR`.
- libhipcxx: `agent_space/libhipcxx/include` (ROCm/libhipcxx amd-develop).
- rapids_logger: `agent_space/rapids_logger` (rapidsai/rapids-logger 0.2.0).
- NVTX3: `agent_space/nvtx3_include` (vendored from pytorch's third_party/NVTX;
  pure host C++, no CUDA device content). Passed via `-DCUDF_NVTX3_INCLUDE_DIR`.

## Build script (lead, gfx90a)

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/cudf/src/cpp -B projects/cudf/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/cudf-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DCUDF_CUCO_SOURCE_DIR=/var/lib/jenkins/moat/_deps/cudf-cuco/src \
  -DCUDF_NVTX3_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/nvtx3_include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=ON
cmake --build projects/cudf/build-hip -j 16   # CAP at -j16: C++20 templates are
                                              # memory-hungry; OOM risk otherwise
```

Arch flows from `CMAKE_HIP_ARCHITECTURES` (defaults to gfx90a only when unset),
so a follower (gfx1100/gfx1151) builds the scoped core with only
`-DCMAKE_HIP_ARCHITECTURES=<arch>` and no source/CMake edit.

## Validation (gfx90a / MI250X, ROCm 7.2.1; run on a FREE GCD)

```
# pick a free GCD from `rocm-smi` (4 GCDs 0-3 on this box); this session used 2
HIP_VISIBLE_DEVICES=2 ctest --test-dir projects/cudf/build-hip --output-on-failure
# (LD_LIBRARY_PATH must include $CONDA_PREFIX/lib for libspdlog and
#  _deps/cudf-rmm/install/lib for librmm/librapids_logger -- sourced via
#  agent_space/cudf_build_env.sh)
```

libcudf.so built with a gfx90a code object (roc-obj-ls:
`hipv4-amdgcn-amd-amdhsa--gfx90a`). **MOAT_CORE_SMOKE_TEST: 6/6 PASS** on real
gfx90a (stable over 20 repeated runs; use a free GCD -- rocm-smi showed 0,1,2,3,
this session used HIP_VISIBLE_DEVICES=2):
- `CountSetBitsWave64`: device `count_set_bits` over known masks == host count,
  sizes {1,31,32,33,63,64,65,127,128,129,1000,4096,100000} (crossing 32-bit-word
  AND 64-lane-wavefront boundaries). The wave64 ballot gate.
- `SegmentedCountSetBits`: segmented reduce over uneven segments == host.
- `CreateNullMask`: ALL_VALID -> n set bits, ALL_NULL -> 0, on device.
- `CountIsDeterministic`: 3x repeated device count bit-identical.
- `SortRadixFloatTupleKeyShim`: `detail::sort_radix` on no-null FLOAT32 columns
  (the decomposer path through the rocPRIM tuple-key shim), asc AND desc, device
  result == host std::sort, sizes 1..100000. The tuple-key shim gate.
- `SortedOrderRadixFloatTupleKeyShim`: `detail::sorted_order_radix` produces the
  sort permutation; gathering the input by it == host std::sort (the SortPairs
  decomposer path + the thrust::tuple zip-iterator functor).

The smoke test links libcudf.so (cudf is a SHARED lib) with
`--unresolved-symbols=ignore-in-shared-libs` + `-z lazy`: the .so carries the
deferred dispatch-closure symbols (above), the test never calls them, and this
linkage tolerates unresolved symbols FROM the .so while still erroring on any
unresolved symbol in the test's own objects (tighter than the old ignore-all). A
broader GPU suite (gather/groupby/join/etc.) needs the remaining link-closure
(wall 1) resolved first.

## Install as a dependency

NOT written yet. cudf is consumed by cugraph/cuml, which need the broad
algorithm core (groupby/join/etc.), not just the foundational TUs here. Defer
the `find_package(cudf)` export contract until a fully-linking core exists
(after the link-closure walls above). The build wiring (cmake/hip/cudf_hip.cmake)
already exports a `cudf` target shape to model it on.
