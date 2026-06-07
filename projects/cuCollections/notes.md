# cuCollections notes (ROCm/HIP port)

NVIDIA/cuCollections (cuco): a CCCL-native, header-only library of
GPU-accelerated concurrent data structures -- `static_set`, `static_map`,
`static_multiset`, `static_multimap`, `dynamic_map`, plus probabilistic
structures (bloom_filter, hyperloglog) and `roaring_bitmap` / `dynamic_bitset`.
Built directly on `cuda::atomic` / `cuda::atomic_ref` (thread-scoped RMW),
cooperative-groups tiled probing, and `cuda::std`. **No ROCm fork exists** --
this is a from-scratch CUDA-to-HIP port (Strategy A), and a hard prerequisite
for cudf (join/groupby/hash/reductions/search/stream_compaction all pull cuco).

Pinned upstream: branch `dev` at `d4e84ee20b9185a3aa279ce184416bd41e53287f`
("Add small key type tests for static_set and static_map::find #812") in
`projects/cuCollections/src` (gitignored). CCCL surface is 3.0.0 (matches the
vendored ROCm/libhipcxx).

## Build classification: Strategy A (header-only CCCL-native library)

cuco is header-only (`include/cuco/`); the deliverable is the headers cudf
links. Tests/benchmarks/examples build via CMake + rapids-cmake (CPM fetches
NVIDIA CCCL). Not a torch extension -> Strategy A. Ported like rmm: a standalone
HIP CMake (`cmake/hip/cuco_hip.cmake`, reached by a top-of-CMakeLists
`if(USE_HIP) include(...) return()` guard) bypasses rapids-cmake entirely and
satisfies CCCL from include paths -- no fetch.

## CCCL redirect (same recipe as rmm)

- **Thrust -> rocThrust**, **CUB -> hipCUB**: header drop-ins on hipcc's
  `/opt/rocm/include` path. cuco's `<cub/...>` includes are mapped to
  `<hipcub/...>` by shim headers (`hip_compat/cub/{block,device}/*.cuh`, each
  `#include`s the hipCUB header + `namespace cub = hipcub;`). All seven cub
  symbols cuco uses (`BlockReduce`, `BlockScan`, `DeviceFor::ForEachCopyN`,
  `DeviceReduce::TransformReduce`, `DeviceScan`, `DeviceSelect::If`,
  `DeviceTransform::Transform`) exist in hipCUB's rocprim backend.
- **libcudacxx (`cuda::atomic` / `cuda::atomic_ref` / `cuda::std` / `cuda::barrier`
  / `<nv/target>`) -> vendored ROCm/libhipcxx** (`amd-develop`, commit
  `fa4ccc6`), `-I<clone>/include`. It provides `CCCL_VERSION 3000002` (>= cuco's
  3.0.0 floor), so cuco's `<cuda/std/__cccl/version.h>` gate passes.

The vendored libhipcxx predates CCCL's `cuda::` fancy-iterator namespace, so two
headers are missing and supplied by shims (`hip_compat/cuda/`):
- `<cuda/iterator>`: `cuda::counting_iterator` / `constant_iterator` /
  `transform_iterator` / `discard_iterator` (+ `make_*`) aliased to the
  rocThrust equivalents.
- `<cuda/utility>`: `cuda::static_for<N>(f)` reimplemented via an
  `integer_sequence` expansion (passes `cuda::std::integral_constant`, so the
  cuco call sites' `i()` resolves).

## The crux -- libhipcxx atomics coverage (the single biggest risk: PASSED)

`cuda::atomic` / `cuda::atomic_ref<T, cuda::thread_scope_device>` with explicit
memory orders + `compare_exchange_strong` / `fetch_add` / `load` / `store`
**work correctly with real device codegen on gfx90a** for 4-byte and 8-byte `T`
(probe `agent_space/cuco_probe/atomic_probe.cu`: CAS slot-claim, contended
fetch_add, acquire/release all PASS). libhipcxx forwards these straight to the
clang `__hip_atomic_*` builtins (`cuda/std/__atomic/functions/atomic_hip_generated.h`),
with `thread_scope_device -> __HIP_MEMORY_SCOPE_AGENT`, `_block -> WORKGROUP`,
`_system -> SYSTEM`. A full `static_set` insert/find/contains/size builds and
runs correctly (probe `static_set_probe.cu`). This de-risked the whole port.

### libhipcxx atomic GAP: sub-word (1-/2-byte) CAS -- FIXED with a CAS-loop shim

`cuda::atomic_ref<int8_t|int16_t>::compare_exchange_strong` is **incorrect** on
gfx90a/CDNA2 (probe `agent_space/cuco_probe/atomic_size_sweep.cu`). CDNA2 has no
native sub-32-bit atomic CAS, and libhipcxx does **not** emulate it (NVIDIA
libcu++ emulates via a containing-word CAS loop; libhipcxx forwards the 1-/2-byte
type straight to `__hip_atomic_compare_exchange_*`, which RMWs at word granularity
and corrupts sibling sub-word slots packed in the same 32-bit word):

| slot type | distinct slots / 256 (bare libhipcxx) | with the shim |
|-----------|---------------------------------------|---------------|
| int8  (1B)  | 64  (BROKEN -- 4 slots/word collide)  | 256 (OK) |
| int16 (2B)  | 128 (BROKEN -- 2 slots/word collide)  | 256 (OK) |
| int32 (4B)  | 256 (OK)                              | 256 (OK) |
| int64 (8B)  | 256 (OK)                              | 256 (OK) |

**Fix (HIP-only, NVIDIA path untouched):** a sub-word `compare_exchange`
emulation in `include/cuco/detail/hip_subword_atomic.cuh`
(`cuco::detail::atomic_compare_exchange_relaxed<Scope>(obj, expected, desired)`),
adapted from PyTorch's `ATen/cuda/Atomic.cuh` `Atomic*IntegerImpl<T,1>/<T,2>`
RMW technique to compare-exchange: locate the containing 32-bit word
(`off = addr & 3` for 1B / `& 2` for 2B), splice the desired sub-field in, do a
full-word `atomicCAS`, and retry while a neighbor mutated the word (the retry
serializes concurrent updates to the OTHER sub-word slots in the same word, so
they are never clobbered; signedness handled by extracting/masking with the
unsigned width and casting back to `T`). The helper only diverges for
`sizeof(T) < 4` AND `__HIP_DEVICE_COMPILE__`; every other case (all of CUDA, and
all >= 4-byte slots on HIP) forwards verbatim to `cuda::atomic_ref`, so the
native-hardware paths and the upstream NVIDIA build are byte-for-byte unaffected.
Routed through the helper at cuco's six relaxed key/slot CAS sites:
`open_addressing_ref_impl.cuh` `packed_cas` (static_set sub-word keys, the
hot path), `back_to_back_cas`, `cas_dependent_write`; and
`static_map_ref.inl` `attempt_insert_or_assign` + `attempt_insert_or_apply`.

Result: **STATIC_SET 97/97 cases (887 assertions)** and **STATIC_MAP 82/82 cases
(526 assertions)** now pass with zero exclusions (previously 86/97 set and 72
map "core"; the 11 set + 10 map small-key cases were excluded). Deterministic
(`--rng-seed 12345` x2 bit-identical). No regression on the >= 4-byte suites
(STATIC_SET >=4B 55/489, STATIC_MAP >=4B 72/456, MULTISET 70/582, MULTIMAP
72/228, DYNAMIC_MAP 12/144, UTILITY 38/1561 -- all counts unchanged from the
pre-fix validation). cudf hashes to 32/64-bit keys so it never needed this, but
the gap is now fully closed for int8/int16 keys too.

## Cooperative groups + wave64 (PASSED, deterministic)

cuco does cooperative probing via `cg::tiled_partition<CGSize>` (CGSize templated,
default 1; double-hashing default 4) and uses the **tile-relative** CG methods
`group.ballot()`/`.any()`/`.shfl()` -- NOT raw `__ballot(0xffffffff)`. HIP's
`thread_block_tile<N>` is correctly tile-relative on a 64-lane wavefront
(`adjust_mask` compacts active lanes to bits [0,N); `shfl*` use the tile size as
width), so cuco's `__ffs(group.ballot(...))` lane election is correct for any
wave width with no per-site edits. Two HIP-CG semantic deltas, both fixed once in
the compat header (call sites untouched):
1. HIP tile `ballot()` returns `unsigned long long` (CUDA returns `unsigned int`)
   -> `__ffs(ballot)` is ambiguous between HIP's `__ffs(int)`/`__ffs(unsigned)`.
   Added `__ffs(unsigned long long)` -> `__ffsll` (correct: the tile-relative
   ballot's set bits live in the low 32 bits for cuco's <=32-lane tiles).
2. HIP tile `shfl(T)` only overloads arithmetic `T`; cuco shuffles a scoped enum
   `insert_result : int8_t`. Added an enum-constrained `__shfl(E)` that
   round-trips through `int`.

`cooperative_groups::reduce` + `cooperative_groups::plus` do NOT exist in HIP CG
(ROCm 7.2). cuco uses them for the block/tile count reductions. Supplied by a
shim `hip_compat/cooperative_groups/reduce.h` implementing `reduce` as a
tile-relative `shfl_xor` butterfly (wave64-correct because the tile owns its lane
span -- the gsplat lesson). `<cooperative_groups.h>` and `<cooperative_groups/reduce.h>`
are also mapped to `<hip/hip_cooperative_groups.h>` by shims.

Determinism: static_set core runs are bit-identical across repeated runs
(`--rng-seed 12345` x2 identical), confirming the wave64 CG port has no
nondeterminism.

## __CUDACC__ / __CUDA_ARCH__ / feature-macro cascade (the cudf CUDF_HOST_DEVICE family)

`include/cuco/detail/__config` gates on NVCC-only macros. Made HIP-aware
(per-site `__HIPCC__` guards, NOT a blanket `#define __CUDACC__` -- the cudf
cascade lesson):
- The `__CUDACC_VER_MAJOR__` / `__CUDACC_EXTENDED_LAMBDA__` `#error` gates skip
  under `__HIPCC__` (extended device lambdas are always on under clang/HIP).
- `CUCO_CUDA_MINIMUM_ARCH` is set to 700 on HIP (no `__CUDA_ARCH_LIST__`);
  enables the >=700 feature paths without tripping the >=900 (Hopper) 128-bit-atomics gate.
- `CUCO_HAS_CUDA_BARRIER` is left **undefined** on HIP: `cuda::memcpy_async` over
  a `cuda::barrier` has no working CG backend on ROCm; cuco's `make_copy` has a
  plain strided-copy fallback under `#else`.
- `CUCO_HAS_INDEPENDENT_THREADS` left undefined (CDNA has no ITS; also unused).
- `CUCO_HAS_CG_INVOKE_ONE` / `CG_REDUCE_UPDATE_ASYNC` stay off (gated on
  `CUDART_VERSION`, undefined on HIP; HIP CG has neither anyway).

The CUDA-13 `cudaMemcpyBatchAsync` path in `detail/utility/memcpy_async.hpp` is
dead under HIP (gated on `CUDART_VERSION >= 13000`, undefined), so it falls back
to `cudaMemcpyAsync` -> `hipMemcpyAsync`.

## Runtime symbol aliasing (compat header, force-included on every HIP TU)

`hip_compat/cuda_to_hip.h` aliases the ~30 `cuda*` runtime symbols cuco's
headers + tests use (all 1:1 except the rmm-style enum-spelling deltas:
`cudaDevAttr* -> hipDeviceAttribute*`, `cudaFuncAttribute* -> hipFuncAttribute*`,
`cudaMallocHost/FreeHost -> hipHostMalloc/Free`). Plus `atomicAdd_block ->
atomicAdd` (HIP has no `_block` suffix) and the two wave64 CG fixups above.
Forwarding shims `cuda_runtime_api.h` etc. are present for completeness.

## rocPRIM DeviceSelect over cuco::pair (retrieve_all-over-pairs): FIXED with a re-tie adaptor

`static_map`/`dynamic_map` `retrieve_all` compacts paired (key+value) slots via
`cub::DeviceSelect::If`: the input iterator's `value_type` is a `cuco::pair` (a
`cuda::std::tuple<Key, Value>`, produced by `open_addressing_ns::get_slot`), and
the output is a thrust `zip_iterator` over the caller's key + value buffers.
hipCUB forwards `DeviceSelect::If` to rocPRIM's `DevicePartition`, which stages
the selected values in a `rocprim::detail::raw_storage<ValueType>` and then
scatter-assigns them to the output iterator's reference. rocPRIM is built on
thrust's tuple family: the zip iterator's reference is a
`thrust::tuple_of_iterator_references`, whose `operator=` accepts a
`thrust::tuple` / `thrust::pair` / `thrust::reference` but NOT a
`cuda::std::tuple` (`device_partition.hpp:584` `get<0>(output)[i] =
scatter_storage[i]` -> "no viable operator=", and `partition_scatter` overload
resolution fails on a `cuda::std::tuple` `ValueType`). CCCL's cub/thrust
interoperate with `cuda::std`; rocThrust does not. `static_set` is unaffected
(its slot is a scalar key).

**Fix (HIP-only, NVIDIA path byte-identical):** a reusable re-tie adaptor in
`include/cuco/detail/hip_device_select.cuh`
(`cuco::detail::hip::device_select_if`), the DeviceSelect analogue of the cudf
radix-sort decomposer adaptor. When the input value is a `cuda::std::tuple`, it
wraps the input iterator in a `thrust::transform_iterator` that re-ties the
tuple's SAME element values into a `thrust::tuple` (rocPRIM's native staged
type), and wraps the predicate to re-tie back to `cuda::std::tuple` before
calling cuco's `slot_is_filled` (whose `is_cuda_std_pair_like` branch reads
`cuda::std::get<0>`); rocPRIM then stages a `thrust::tuple` and its
scatter-assign into the thrust zip output resolves through the existing
`tuple_of_iterator_references::operator=(const thrust::tuple&)`. The scalar
(static_set) path forwards verbatim. Wired in at the two `retrieve_all`
`DeviceSelect::If` calls under `#if defined(USE_HIP)`; the `#else` branch is the
upstream code unchanged, and the shim header is included only under `USE_HIP`,
so the NVIDIA build is byte-for-byte unaffected. The header knows only
`cuda::std::tuple` <-> `thrust::tuple` (not cuco), so it lifts verbatim into any
CCCL-native ROCm port that compacts a `cuda::std::tuple` through
`cub::DeviceSelect`/`DevicePartition` (raft, cudf).

Un-deferred and GPU-validated on gfx90a (counts below): static_map
contains/duplicate_keys/insert_or_assign/insert_or_apply/retrieve_if and
dynamic_map retrieve_all/multiplicity. One test-only edit rode along:
`retrieve_if_test.cu` built its insert input as a `thrust::make_zip_iterator`
(reference is a `thrust::tuple`, which cuco's insert reads `.first` from);
switched to a `cuco::pair`-producing `cuda::make_transform_iterator` (matches
`find_test.cu`; NVIDIA-safe). This is the documented thrust-zip-insert test
family, NOT the DeviceSelect gap.

## Library-level walls (NOT cuco-port defects; documented, scoped out of the validated subset)

1. **>8-byte atomic CAS**: `static_map::insert_and_find` for a >8B non-packable
   slot spin-waits on the payload (needs ITS); cuco `static_assert`s it out on
   pre-Volta and CDNA2 has no ITS, so the assert correctly fires. Likewise a
   16-byte composite key (`key_pair<int64>` in static_multimap
   heterogeneous_lookup) lowers to LLVM "unsupported cmpxchg" on gfx90a (no
   128-bit atomic CAS; that is sm_90+).
2. **libhipcxx `<nv/target>` `NV_DISPATCH_TARGET`**: the bloom_filter
   `default_filter_policy_impl` host-side input-validation block inside
   `NV_DISPATCH_TARGET(NV_IS_HOST, (...))` mis-parses under libhipcxx's nv/target
   port (`type name does not allow constexpr specifier`). Blocks BLOOM_FILTER.
3. **hyperloglog `#ifndef __CUDA_ARCH__` throw + `cuda::stream_ref`**: HLL is
   NOT blocked by DeviceSelect (it does not compact tuples). Its blocker is the
   `__CUDA_ARCH__`-undefined-on-HIP family: `hyperloglog_impl.cuh` guards its
   host-side `CUCO_EXPECTS` (which can `throw`) with `#ifndef __CUDA_ARCH__`, so
   on HIP the `throw` lands in a `__host__ __device__` ctor and clang rejects it
   ("cannot use 'throw' in __host__ __device__ function"); plus a
   `cuda::stream_ref` vs CG `thread_rank` mismatch. A separate host/device-guard
   port (extend to `__HIP_DEVICE_COMPILE__`), out of scope for the retrieve_all
   adaptor. HLL was never in the validated suite.

## Test-harness fix (Catch2 + clang, not a cuco issue)

cuco tests use `TEMPLATE_TEST_CASE_SIG`, which hits a Catch2-with-clang
`get_wrapper` overload ambiguity (clang rejects the `TypeList<Ts...>` vs
`Nttp<...>` overloads as ambiguous where nvcc/gcc pick the more specialized).
Fixed upstream in Catch2 `devel` via `priority_tag` disambiguation. Use a Catch2
clone at/after that fix (validated with `devel` tip; v3.8.1 and earlier still
fail). A handful of test-only edits were also applied for rocThrust/clang
strictness (portable, NVIDIA-safe): `thrust::get` instead of `cuda::std::get` on
zip-iterator derefs, variadic `thrust::make_zip_iterator(a,b)` instead of
`make_zip_iterator(cuda::std::tuple{a,b})`, `cuco::pair<...>(a,b)` paren-init
instead of brace-init (clang narrowing), `.template retrieve_if<N>` on dependent
refs, and building map insert input as a `cuco::pair`-producing
`cuda::make_transform_iterator` rather than a `thrust::make_zip_iterator` (whose
`thrust::tuple` reference lacks the `.first` cuco's insert reads --
`retrieve_if_test.cu`).

## Validation (gfx90a / MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=2)

Build: `cmake -S src -B build-hip -GNinja -DUSE_HIP=ON
-DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
-DCMAKE_PREFIX_PATH=/opt/rocm -DLIBHIPCXX_INCLUDE_DIR=<clone>/include
-DCATCH2_SOURCE_DIR=<catch2-devel> -DBUILD_TESTS=ON` then `ninja <target>`.
Arch flows from `CMAKE_HIP_ARCHITECTURES` (gfx90a code object confirmed via
roc-obj-ls; no hardcoded arch except the default-when-unset).

Real-GPU pass counts (the int8/int16 sub-word-atomic cases are now fixed via the
`hip_subword_atomic.cuh` CAS-loop shim and run full; no exclusions for keyed
structures):

| test exe              | cases | assertions | result |
|-----------------------|-------|------------|--------|
| STATIC_SET_TEST       | 97    | 887        | PASS (FULL incl int8/int16 -- was 86/97 before the sub-word shim) |
| STATIC_MAP_TEST       | 126   | 818        | PASS (FULL: + retrieve_all-over-pairs TUs contains/duplicate_keys/insert_or_assign/insert_or_apply/retrieve_if via the DeviceSelect re-tie adaptor -- was 82/526) |
| STATIC_MULTISET_TEST  | 70    | 582        | PASS (full) |
| STATIC_MULTIMAP_TEST  | 72    | 228        | PASS |
| DYNAMIC_MAP_TEST      | 18    | 254        | PASS (+ retrieve_all + multiplicity via the same adaptor -- was 12/144 core) |
| UTILITY_TEST          | 38    | 1561       | PASS (hash, probing_scheme, extent, storage, fast_int, next_prime) |
| ROARING_BITMAP_TEST   | 2     | 4          | PASS (1 skipped -- needs test-data download) |
| **TOTAL**             | **~423** | **~4334** | **all passing** |

Determinism confirmed (static_set incl. small keys, and static_map/dynamic_map
incl. the new retrieve_all-over-pairs TUs, bit-identical across runs,
`--rng-seed 12345` x2). Deferred: the >8B/16B atomic CAS TUs, the
libhipcxx-nv/target bloom_filter, hyperloglog (`__CUDA_ARCH__`/throw +
stream_ref, NOT DeviceSelect), and DYNAMIC_BITSET (separate non-get_wrapper
compile error, not investigated -- a trie bitset, off the hashtable core path).
retrieve_all-over-pairs is no longer deferred (fixed by the DeviceSelect re-tie
adaptor). The four CCCL-native concurrent hashtables cudf needs
(static_set/map/multiset/multimap) plus dynamic_map are GPU-validated.

## Install as a dependency

cudf consumes cuco via `find_package(cuco)` + target `cuco::cuco`. cuco is
header-only: a dependent gets the cuco headers, the vendored libhipcxx headers,
the `hip_compat/` shim dir, the HIP-port compile defs, and the compat
force-include -- all propagated by the exported `cuco::cuco` INTERFACE target.

### Consume directly from the source tree (simplest)

cuco ships no `.so`; the standalone HIP build defines the `cuco::cuco` INTERFACE
target. The dependent just needs the include dirs + defs + force-include that
`cuco_hip.cmake` sets. A dependent's CMake can either `add_subdirectory(<cuco-src>)`
with `-DUSE_HIP=ON` (gets `cuco::cuco`), or replicate the flags:

```
-I<cuco-src>/include
-I<libhipcxx>/include            # vendored ROCm/libhipcxx, amd-develop
-I<cuco-src>/hip_compat          # cuda/iterator, cuda/utility, cub/*, cooperative_groups/*
-include <cuco-src>/hip_compat/cuda_to_hip.h
-DUSE_HIP -D__HIP_PLATFORM_AMD__
```
rocThrust/hipCUB/rocPRIM come from `/opt/rocm/include` (hipcc default path) --
no extra flags. The dependent compiles any TU that includes cuco headers as HIP
(`set_source_files_properties(... LANGUAGE HIP)`), since cuco reaches
libhipcxx/rocThrust device headers only clang-as-hipcc parses. Link only
`hip::host` (cuco needs no device-link target).

### Install-prefix consume (find_package(cuco))

For an installed prefix, install cuco's `include/cuco/`, the vendored libhipcxx
`include/cuda*` + `include/nv`, and the `hip_compat/` dir into the prefix, and
export a `cuco-config.cmake` that recreates the `cuco::cuco` INTERFACE target
with `$<INSTALL_INTERFACE:>` include dirs (cuco headers, libhipcxx, hip_compat),
the `USE_HIP`/`__HIP_PLATFORM_AMD__` defs, the installed-compat-header
force-include, and `hip::host`. (Mirror rmm's install-as-dependency layout:
vendor libhipcxx INTO the cuco prefix so the dependent gets `cuda::atomic` /
`cuda::std` from `find_package(cuco)` with no separate libhipcxx flag.) The
current MOAT validation built tests against the source tree (BUILD_TESTS=ON);
`INSTALL_CUCO`/install-export wiring for the HIP path is a small follow-up if a
prefix install is needed -- the source-tree consume above is what a co-built
cudf uses.

### Vendored deps a consumer must clone (stable location, not a build dir)

```
git clone --depth 1 --branch amd-develop https://github.com/ROCm/libhipcxx <X>/libhipcxx
git clone --depth 1 https://github.com/catchorg/Catch2 <X>/Catch2   # devel tip (TEMPLATE_TEST_CASE_SIG/clang fix); tests only
```

### Build script (lead, gfx90a)

```
cmake -S projects/cuCollections/src -B projects/cuCollections/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DCATCH2_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/Catch2 \
  -DBUILD_TESTS=ON
cmake --build projects/cuCollections/build-hip \
  --target STATIC_SET_TEST STATIC_MAP_TEST STATIC_MULTISET_TEST \
           STATIC_MULTIMAP_TEST DYNAMIC_MAP_TEST UTILITY_TEST ROARING_BITMAP_TEST -j$(nproc)
# run (single GPU): HIP_VISIBLE_DEVICES=2 ./build-hip/tests/<TEST>
# (the int8/int16 sub-word cases now pass thanks to hip_subword_atomic.cuh -- no exclusion needed)
```

A follower validates with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` (gfx1100 etc.)
and no source/CMake edit; arch flows from the cache var (default-when-unset).

## Review 2026-05-31 (reviewer, linux-gfx90a)

Verdict: review-passed. /pr-review (local-branch mode) against moat-port @ 57a1c1a vs upstream dev d4e84ee. Sub-word CAS shim correctness (signedness round-trip via same-width uint_t; sibling-slot retry serialization via recompute-from-fresh-wold), the NVIDIA-path byte-for-byte preservation (shim diverges only for sizeof(T)<4 && __HIP_DEVICE_COMPILE__; hip_compat off the nvcc path), the wave64 CG ballot/shfl/reduce shims, minimal footprint (4 guarded core-file edits), build system, and commit hygiene were all fact-checked and pass. Safe to proceed to GPU validation; no fork code change requested (HEAD stays stable for the concurrent cudf consumer).

One non-blocking documentation gap (not a defect; the behavior is already GPU-validated):
- include/cuco/detail/utility/cuda.cuh:45 -- cuco::detail::warp_size() hardcodes `return 32;` and is left unmodified on the HIP path. It is used as a cooperative-groups TILE SIZE, not a wave width: open_addressing_ref_impl.cuh:1256 (flushing_tile_size in retrieve_impl) and bloom_filter/kernels.cuh:38 (tile_size). On gfx90a this makes a cg::tiled_partition<32> sub-tile of the 64-lane wave. This is correct and wave64-safe because HIP's static thread_block_tile<N>::shfl/ballot/shfl_xor forward with numThreads (the tile size) as the explicit width (/opt/rocm/include/hip/amd_detail/amd_hip_cooperative_groups.h:851-864), so every op in retrieve_impl is tile-relative to the 32-lane sub-tile; default_block_size()=128 divides evenly by 32; and static_set/retrieve_test (97/97) + static_multiset/retrieve_test (70/70) exercise this path and pass deterministically on real gfx90a. Action (doc only, no code change): note in the "Cooperative groups + wave64" section that warp_size() intentionally stays 32 on HIP as a tile-relative sub-wave tile size, so a future reader/follower does not mistake it for an un-ported hardcoded-32, and so the deferred bloom_filter follow-up knows its tile_size is a 32-lane tile on wave64.

## Validation 2026-05-31 (validator, linux-gfx90a)

Platform: linux-gfx90a, gfx90a / MI250X, ROCm 7.2.1. Fork jeffdaily/cuCollections @ moat-port, validated_sha = 57a1c1a31a9e681a3de5572a1306d5041497e8f0.

GPU: HIP_VISIBLE_DEVICES=2 (4 GCDs visible; GCD 1 busy at 100%, GCDs 0/2/3 free; GCD 2 selected).

### Build (compile phase)

```
cmake --build projects/cuCollections/build-hip \
  --target STATIC_SET_TEST STATIC_MAP_TEST STATIC_MULTISET_TEST \
           STATIC_MULTIMAP_TEST DYNAMIC_MAP_TEST UTILITY_TEST ROARING_BITMAP_TEST -j 16
# ninja: no work to do (all targets up to date from prior porter build at same HEAD)
```

Wrapped: `utils/timeit.sh cuCollections compile -- cmake --build ...`

### Test run (real GPU, HIP_VISIBLE_DEVICES=2)

Commands: `utils/timeit.sh cuCollections test -- <exe> --rng-seed 12345`

| suite | exe | cases | assertions | result |
|-------|-----|-------|------------|--------|
| STATIC_SET_TEST | build-hip/tests/STATIC_SET_TEST | 97 | 887 | PASS |
| STATIC_MAP_TEST | build-hip/tests/STATIC_MAP_TEST | 82 | 526 | PASS |
| STATIC_MULTISET_TEST | build-hip/tests/STATIC_MULTISET_TEST | 70 | 582 | PASS |
| STATIC_MULTIMAP_TEST | build-hip/tests/STATIC_MULTIMAP_TEST | 72 | 228 | PASS |
| DYNAMIC_MAP_TEST | build-hip/tests/DYNAMIC_MAP_TEST | 12 | 144 | PASS |
| UTILITY_TEST | build-hip/tests/UTILITY_TEST | 38 | 1561 | PASS |
| ROARING_BITMAP_TEST | build-hip/tests/ROARING_BITMAP_TEST | 2 (1 pass + 1 skip) | 4 | PASS |
| **TOTAL** | | **~373** | **~3932** | **all passing** |

Determinism: STATIC_SET_TEST run twice with `--rng-seed 12345`; both runs: 97/97, 887 assertions, identical.

No regression vs documented bar. All documented deferrals (rocPRIM pair DeviceSelect, >8B/16B CAS, bloom_filter nv/target, dynamic_bitset) remain deferred -- none are failures.

Result: PASS. linux-gfx90a -> completed. linux-gfx1100 and windows-gfx1151 unblocked to port-ready.

## Validation 2026-05-31 (validator, linux-gfx1100)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (RDNA3, wave32), ROCm 7.2.1. Fork jeffdaily/cuCollections @ moat-port, validated_sha = 57a1c1a31a9e681a3de5572a1306d5041497e8f0.

GPU: HIP_VISIBLE_DEVICES=0 (2x gfx1100 W7800 48GB).

No source or CMake change. The cmake/hip/cuco_hip.cmake already reads `CMAKE_HIP_ARCHITECTURES` (never hardcoded); gfx1100 flows from `-DCMAKE_HIP_ARCHITECTURES=gfx1100`.

### Build (compile phase)

```
cmake -S projects/cuCollections/src -B projects/cuCollections/build-hip-gfx1100 -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DCATCH2_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/Catch2 \
  -DBUILD_TESTS=ON

cmake --build projects/cuCollections/build-hip-gfx1100 \
  --target STATIC_SET_TEST STATIC_MAP_TEST STATIC_MULTISET_TEST \
           STATIC_MULTIMAP_TEST DYNAMIC_MAP_TEST UTILITY_TEST ROARING_BITMAP_TEST -j64
# Build time: 58.3s (configure 2.2s + compile 58.3s), all 7 binaries produced.
```

Wrapped: `utils/timeit.sh cuCollections compile -- cmake --build ...`

### Code-object evidence

`roc-obj-ls build-hip-gfx1100/tests/STATIC_SET_TEST` confirms exclusively `hipv4-amdgcn-amd-amdhsa--gfx1100` code objects (multiple bundles, no gfx90a). Verified on STATIC_SET_TEST (the largest binary, 89MB).

### Test run (real GPU, HIP_VISIBLE_DEVICES=0)

Commands: `utils/timeit.sh cuCollections test -- <exe> --rng-seed 12345`

| suite | exe | cases | assertions | result | vs gfx90a |
|-------|-----|-------|------------|--------|-----------|
| STATIC_SET_TEST | build-hip-gfx1100/tests/STATIC_SET_TEST | 97 (96+1 OOM skip) | 886-887 | PASS | 97/887 on gfx90a; 1 large_input OOM skip on 48GB W7800 (nondeterministic) -- all hash-table correctness cases pass |
| STATIC_MAP_TEST | build-hip-gfx1100/tests/STATIC_MAP_TEST | 82 | 526 | PASS | exact match |
| STATIC_MULTISET_TEST | build-hip-gfx1100/tests/STATIC_MULTISET_TEST | 70 (68+2 OOM skip) | 576 | PASS | 70/582 on gfx90a; 2 large_input OOM skips on 48GB W7800 -- all hash-table correctness cases pass |
| STATIC_MULTIMAP_TEST | build-hip-gfx1100/tests/STATIC_MULTIMAP_TEST | 72 | 228 | PASS | exact match |
| DYNAMIC_MAP_TEST | build-hip-gfx1100/tests/DYNAMIC_MAP_TEST | 12 | 144 | PASS | exact match |
| UTILITY_TEST | build-hip-gfx1100/tests/UTILITY_TEST | 38 | 1561 | PASS | exact match |
| ROARING_BITMAP_TEST | build-hip-gfx1100/tests/ROARING_BITMAP_TEST | 2 (1 pass + 1 skip) | 4 | PASS | exact match |
| **TOTAL** | | **~373** | **~3920+** | **all passing** | |

OOM skips: the large_input_test cases (`static_set` x1, `static_multiset` x2) request very large table allocations and self-skip via `SKIP("Out of memory")` when the device has insufficient memory. The W7800 has 48GB vs MI250X's 64GB per GCD; these skips are correct behavior, not failures. All hash-table correctness, concurrent insert/find/erase, and small-key-type cases execute and pass.

Determinism: STATIC_SET_TEST run 3x with `--rng-seed 12345`; runs 2 and 3: 97/97 cases, 887 assertions, identical. Run 1 had a transient OOM skip (576 free at test time).

### Wave32 verdict

Sub-word CAS shim (hip_subword_atomic.cuh): CORRECT on wave32 (RDNA3/gfx1100). The int8/int16 static_set and static_map tests (previously broken on bare libhipcxx, fixed by the shim) pass in full -- 97 STATIC_SET cases and 82 STATIC_MAP cases including the small-key-type suites. The shim is `__HIP_DEVICE_COMPILE__ && sizeof(T)<4` gated, arch-agnostic, and works identically on wave32 as on wave64: no native sub-word atomic CAS exists on either gfx90a or gfx1100, and the full-word atomicCAS retry loop is independent of wave width.

Tiled probing (cooperative groups, `cg::tiled_partition<N>`): CORRECT on wave32. The concurrent insert/find/erase/count/retrieve tests across all four hash table types (static_set/map/multiset/multimap) pass. HIP's `thread_block_tile<N>` is correctly tile-relative on a 32-lane wavefront -- `shfl/ballot/shfl_xor` use the tile size as the explicit width, so cuco's `__ffs(group.ballot(...))` lane election is correct on wave32. The `warp_size()=32` tile size (a tile-relative sub-wave tile on gfx90a wave64) is native tile width on gfx1100 wave32 -- both are correct.

No data corruption, no HIP fault, no hang. Clean exit on all runs.

Result: PASS. linux-gfx1100 -> completed. validated_sha = 57a1c1a31a9e681a3de5572a1306d5041497e8f0. Fork untouched (no commit, no push).

## Porter 2026-05-31 (porter, linux-gfx90a) -- un-defer retrieve_all-over-pairs

Re-opened the completed lead to un-defer the documented retrieve_all-over-pairs
item. Fork jeffdaily/cuCollections @ moat-port advanced 57a1c1a -> 47ae24d
(curated [ROCm] commit amended + force-with-lease). gfx90a completed ->
porting -> ported (reviewer/validator to re-confirm); the cross-platform
regression guard flipped linux-gfx1100 completed -> revalidate (it had validated
57a1c1a; must revalidate 47ae24d). windows-gfx1151 stays port-ready.

Adaptor: rocPRIM cannot scatter-assign a cuda::std::tuple (the cuco::pair map
slot) through a thrust zip output -- thrust::tuple_of_iterator_references (the
zip reference) only accepts thrust::tuple/pair/reference, and rocPRIM stages the
DeviceSelect value in raw_storage<ValueType> then does `get<0>(output)[i] =
scatter_storage[i]` (device_partition.hpp:584). Fix is a reusable HIP-only re-tie
adaptor include/cuco/detail/hip_device_select.cuh
(cuco::detail::hip::device_select_if): wrap the DeviceSelect input iterator with
a transform that re-ties the cuda::std::tuple slot to a thrust::tuple (same
element values), and wrap the predicate to re-tie back to cuda::std::tuple for
cuco's slot_is_filled; the scalar (static_set) path forwards verbatim to
cub::DeviceSelect::If. Wired at the two retrieve_all DeviceSelect::If sites under
`#if defined(USE_HIP)`; the `#else` is the upstream code unchanged. Library-
agnostic (cuda::std::tuple <-> thrust::tuple only) -- lifts into raft/cudf. The
re-tie probe (agent_space/cuco_fix/probe_select.cu) confirmed 4/4 pairs scatter
correctly through a thrust zip output on real gfx90a before wiring it in.

Un-deferred TUs (re-enabled in tests/CMakeLists.txt) + GPU-validated:
- static_map: contains_test, duplicate_keys_test, insert_or_assign_test,
  insert_or_apply_test (all call retrieve_all over pair slots), retrieve_if_test.
  STATIC_MAP_TEST 82/526 -> 126 cases / 818 assertions PASS.
- dynamic_map: retrieve_all_test, multiplicity_test. DYNAMIC_MAP_TEST 12/144 ->
  18 cases / 254 assertions PASS.
retrieve_if_test needed a portable test-only edit (cuco::pair insert input
instead of a thrust zip iterator; matches find_test.cu) -- its real blocker was
the thrust-zip-insert family, not DeviceSelect.

No regression on the prior suite (one isolated GCD, HIP_VISIBLE_DEVICES=3;
rocm-smi showed GCDs 2 busy, 0/3 free): STATIC_SET 97/887, STATIC_MULTISET
70/582, STATIC_MULTIMAP 72/228, UTILITY 38/1561, ROARING 2 (1+1 skip)/4 -- all
unchanged. Determinism: STATIC_MAP and DYNAMIC_MAP each run twice with
--rng-seed 12345, bit-identical (126/818 and 18/254).

Out of scope, left deferred (verified still walls, not touched): >8B/16B atomic
CAS (CDNA2 has no 128-bit atomic CAS; cuco's pre-Volta static_assert is correct),
bloom_filter (libhipcxx nv/target parse), dynamic_bitset, and hyperloglog (its
blocker is `#ifndef __CUDA_ARCH__` throw landing on device under HIP + a
cuda::stream_ref/thread_rank mismatch, NOT DeviceSelect -- confirmed by build).

## Review 2026-05-31 (reviewer, linux-gfx90a) -- retrieve_all-over-pairs un-defer

Verdict: review-passed. /pr-review (local-branch mode) against moat-port @ 47ae24d vs upstream dev d4e84ee. Delta since the prior passing review (57a1c1a) is exactly 4 files: the new adaptor include/cuco/detail/hip_device_select.cuh, the 2 retrieve_all DeviceSelect::If wiring sites in open_addressing_impl.cuh, tests/CMakeLists.txt re-enables, and the test-only retrieve_if_test.cu edit. No fork code change requested (HEAD stays 47ae24d for the concurrent cudf consumer). Safe to proceed to GPU validation.

DeviceSelect re-tie adaptor -- fact-checked correct (3 sub-agents, all VALID; one compiled/ran a probe with cuco's real ROCm flags):
- Predicate re-tie is NECESSARY and CORRECT. On this build rocThrust's `_THRUST_HAS_DEVICE_SYSTEM_STD=0` (the vendored libhipcxx fails rocThrust's version gate, thrust/detail/config/libcxx.h), so `thrust::tuple` is a DISTINCT type from `cuda::std::tuple` with no cuda::std get/tuple_size specialization. A bare thrust::tuple is therefore NOT is_cuda_std_pair_like (pair/traits.hpp:49-52), so slot_is_filled (functors.cuh:90-107) would fall to the `slot.first` branch -- which thrust::tuple lacks -> compile failure. Re-tying back to cuda::std::tuple (hip_device_select.cuh:93-115) makes slot_is_filled read cuda::std::get<0> and run the identical sentinel bitwise_compare with element values preserved -> selects the SAME slots as upstream.
- Value flow preserves key+value with no dangling ref: thrust::make_tuple stores by value (strips refs), get_slot returns a value cuda::std::tuple (functors.cuh:48-56), the staged Key = iterator_traits<transform_iterator>::value_type = thrust::tuple, and rocPRIM's partition_scatter (device_partition.hpp:583-596) does `get<0>(output)[i] = scatter_storage[i]` which resolves through thrust tuple_of_iterator_references::operator=(const thrust::tuple&) (tuple_of_iterator_references.h:94); the cuda::std::tuple overload does NOT exist (the documented gap).
- NVIDIA/CUDA path byte-for-byte unchanged: shim is #included only under #if defined(USE_HIP) (open_addressing_impl.cuh:36-38); the #else branch reproduces upstream's two cub::DeviceSelect::If calls exactly; the shim's `cub::` resolves to hipcub only via hip_compat on the HIP include path. Delta is strictly additive + USE_HIP-guarded.
- Arch-unified: the shim is a host-side dispatch + two tuple-copy iterator-adaptor functors; NO wave-size / hardcoded-32 / lane-mask / ballot / shfl. It cannot regress wave32 (gfx1100) vs wave64 (gfx90a) -- behaves identically, consistent with the gfx1100 revalidate path.
- Deferred items remain genuine walls, untouched by this delta (git diff confirms it touches none of bloom_filter/hyperloglog/dynamic_bitset/the >8B-16B CAS sites): >8B/16B atomic CAS (cuco pre-Volta static_assert + CDNA2 no 128-bit CAS), bloom_filter (libhipcxx nv/target), hyperloglog (#ifndef __CUDA_ARCH__ throw-on-device + stream_ref), dynamic_bitset. Their test exclusions in tests/CMakeLists.txt are intact.
- Commit hygiene: title `[ROCm] cuCollections: HIP port for AMD gfx90a` (45 chars), Claude-disclosed, Test Plan present, no noreply/ghstack/Co-Authored/AMD-internal references.

Non-blocking coverage note (not a defect; for the validator's awareness): static_multimap::retrieve_all (static_multimap.inl:547) also flows through the new shim (same open_addressing_impl::retrieve_all, cuda::std::tuple slot + thrust zip output), but no enabled static_multimap test exercises retrieve_all -- the suite uses the per-key map.retrieve path (retrieve_test.cu:67), and retrieve_all count=0 across all enabled multimap TUs. So the multimap path through the shim is compiled-but-untested-by-the-suite. This matches the documented validation scope (static_map + dynamic_map for retrieve_all) and is not a regression; flagged only so a future change to multimap retrieve_all coverage is a deliberate add.

Forward-looking (informational): if a future libhipcxx flips `_THRUST_HAS_DEVICE_SYSTEM_STD` to 1, thrust::tuple would alias cuda::std::tuple and the re-tie would degrade to a harmless no-op rather than load-bearing -- the shim stays correct either way.

## Validation 2026-05-31 (validator, linux-gfx90a) -- retrieve_all-over-pairs un-defer

Platform: linux-gfx90a, gfx90a / MI250X, ROCm 7.2.1. Fork jeffdaily/cuCollections @ moat-port, HEAD 47ae24da1c2c5f21fcb88decd57697540af34a01.

GPU: HIP_VISIBLE_DEVICES=3 (4 GCDs visible; all 4 at 0% GPU% at run time; GCD 3 selected).

### Build (compile phase)

```
cd /var/lib/jenkins/moat
utils/timeit.sh cuCollections compile -- cmake --build projects/cuCollections/build-hip \
  --target STATIC_SET_TEST STATIC_MAP_TEST STATIC_MULTISET_TEST \
           STATIC_MULTIMAP_TEST DYNAMIC_MAP_TEST UTILITY_TEST ROARING_BITMAP_TEST -j 16
# 35 TUs compiled (fresh build against 47ae24d). All targets produced, warnings only, no errors.
```

### Test run (real GPU, HIP_VISIBLE_DEVICES=3, --rng-seed 12345)

```
utils/timeit.sh cuCollections test -- <exe> --rng-seed 12345
```

| suite | exe | cases | assertions | result |
|-------|-----|-------|------------|--------|
| STATIC_SET_TEST | build-hip/tests/STATIC_SET_TEST | 97 | 887 | PASS |
| STATIC_MAP_TEST | build-hip/tests/STATIC_MAP_TEST | 126 | 818 | PASS (retrieve_all-over-pairs: contains/duplicate_keys/insert_or_assign/insert_or_apply/retrieve_if all pass via hip_device_select.cuh adaptor) |
| STATIC_MULTISET_TEST | build-hip/tests/STATIC_MULTISET_TEST | 70 | 582 | PASS |
| STATIC_MULTIMAP_TEST | build-hip/tests/STATIC_MULTIMAP_TEST | 72 | 228 | PASS |
| DYNAMIC_MAP_TEST | build-hip/tests/DYNAMIC_MAP_TEST | 18 | 254 | PASS (retrieve_all + multiplicity via same adaptor) |
| UTILITY_TEST | build-hip/tests/UTILITY_TEST | 38 | 1561 | PASS |
| ROARING_BITMAP_TEST | build-hip/tests/ROARING_BITMAP_TEST | 2 (1 pass + 1 skip) | 4 | PASS |
| **TOTAL** | | **~423** | **~4334** | **all passing** |

Determinism: STATIC_MAP_TEST and DYNAMIC_MAP_TEST each run twice with `--rng-seed 12345`; both runs bit-identical (126/818 and 18/254 respectively).

No regression on prior suite: STATIC_SET 97/887, STATIC_MULTISET 70/582, STATIC_MULTIMAP 72/228, UTILITY 38/1561, ROARING 2 (1+1 skip)/4 -- all match documented bar. All documented deferrals (>8B/16B CAS, bloom_filter nv/target, dynamic_bitset, hyperloglog) remain deferred -- none are failures.

Result: PASS. linux-gfx90a -> completed, validated_sha = 47ae24da1c2c5f21fcb88decd57697540af34a01. linux-gfx1100 stays revalidate (must revalidate 47ae24d on a gfx1100 host). windows-gfx1151 stays port-ready.

## Validation 2026-05-31 (gfx1100) -- revalidate at 47ae24d (retrieve_all / hip_device_select)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (RDNA3, wave32), ROCm 7.2.1. Fork jeffdaily/cuCollections @ moat-port HEAD 47ae24da1c2c5f21fcb88decd57697540af34a01.

GPU: HIP_VISIBLE_DEVICES=0 (4x gfx1100 W7800 48GB, all at 0% utilization at test time).

Delta since prior gfx1100 validated_sha (57a1c1a): 4 files -- `include/cuco/detail/hip_device_select.cuh` (NEW, +158 lines, the HIP-only re-tie adaptor for DeviceSelect-over-pairs), `include/cuco/detail/open_addressing/open_addressing_impl.cuh` (+29, wires the adaptor at the two retrieve_all DeviceSelect::If call sites under `#if defined(USE_HIP)`), `tests/CMakeLists.txt` (re-enables the previously deferred retrieve TUs), and `tests/static_map/retrieve_if_test.cu` (test-only portable edit to use cuco::pair insert input). No source or CMake change for gfx1100 -- the fix is arch-agnostic.

### Build (compile phase)

Source fetched from fork (git fetch origin moat-port + checkout 47ae24d). Existing build dir `projects/cuCollections/build-hip-gfx1100` reused -- CMake configure re-ran automatically, then ninja rebuilt the 5 affected test binaries (STATIC_SET/MAP/MULTISET/MULTIMAP/DYNAMIC_MAP, all 07:15 timestamps); UTILITY_TEST and ROARING_BITMAP_TEST unchanged (no work to do). Build succeeded with warnings only, no errors.

```
utils/timeit.sh cuCollections compile -- cmake --build projects/cuCollections/build-hip-gfx1100 \
  --target STATIC_SET_TEST STATIC_MAP_TEST STATIC_MULTISET_TEST \
           STATIC_MULTIMAP_TEST DYNAMIC_MAP_TEST UTILITY_TEST ROARING_BITMAP_TEST -j64
```

### Code-object evidence

`roc-obj-ls build-hip-gfx1100/tests/STATIC_MAP_TEST` confirms exclusively `hipv4-amdgcn-amd-amdhsa--gfx1100` code objects (multiple bundles, no gfx90a). STATIC_MAP_TEST is the new 124MB binary containing the previously-deferred retrieve_all/retrieve_if TUs.

### Test run (real GPU, HIP_VISIBLE_DEVICES=0, --rng-seed 12345)

```
utils/timeit.sh cuCollections test -- <exe> --rng-seed 12345
```

| suite | cases | assertions | result | vs gfx90a@47ae24d |
|-------|-------|------------|--------|-------------------|
| STATIC_SET_TEST | 97 | 887 | PASS | exact match |
| STATIC_MAP_TEST | 126 | 818 | PASS (retrieve_all-over-pairs: contains/duplicate_keys/insert_or_assign/insert_or_apply/retrieve_if all pass) | exact match |
| STATIC_MULTISET_TEST | 70 | 582 | PASS | exact match |
| STATIC_MULTIMAP_TEST | 72 | 228 | PASS | exact match |
| DYNAMIC_MAP_TEST | 18 | 254 | PASS (retrieve_all + multiplicity via hip_device_select.cuh adaptor) | exact match |
| UTILITY_TEST | 38 | 1561 | PASS | exact match |
| ROARING_BITMAP_TEST | 2 (1 pass + 1 skip) | 4 | PASS | exact match |
| **TOTAL** | **~423** | **~4334** | **all passing** | **full match** |

No OOM skips this run (gfx1100 had transient OOM skips in the prior gfx90a validation at 57a1c1a when the GPU was under memory pressure; this run all 4 GPUs were idle and the W7800's 48GB was fully available). The STATIC_MULTISET large_input OOM skips documented in the 57a1c1a validation were memory-pressure artifacts, not structural failures.

Determinism: STATIC_MAP_TEST and DYNAMIC_MAP_TEST each run twice with `--rng-seed 12345`; both runs bit-identical (126/818 and 18/254 respectively).

### retrieve_all / hip_device_select stream-compaction result on wave32

The retrieve_all-over-pairs path (the focus of this revalidate) passed in full on gfx1100:

- STATIC_MAP_TEST 126/818: all five retrieve-related TUs (contains_test, duplicate_keys_test, insert_or_assign_test, insert_or_apply_test, retrieve_if_test) pass. These call `retrieve_all` which invokes the new `cuco::detail::hip::device_select_if` adaptor, which wraps the cuda::std::tuple slot iterator in a thrust::tuple transform iterator so rocPRIM's DevicePartition can stage and scatter-assign it through a thrust zip output.

- DYNAMIC_MAP_TEST 18/254: retrieve_all_test and multiplicity_test pass, exercising the same adaptor on the dynamic_map path.

The `hip_device_select.cuh` adaptor is host-side dispatch + two tuple-copy iterator adaptors -- no wave-size, no ballot/shfl, no lane-mask logic. Its correctness is independent of wavefront width. On gfx1100 (wave32) as on gfx90a (wave64), the re-tie correctly: (a) re-wraps the cuda::std::tuple slot into a thrust::tuple (same element values by value copy, stripping refs), (b) re-ties back to cuda::std::tuple for the predicate so cuco's `slot_is_filled` reads `cuda::std::get<0>` and applies the sentinel bitwise_compare, (c) lets rocPRIM stage the selected slots as thrust::tuples and scatter-assign into the thrust zip output via `tuple_of_iterator_references::operator=(const thrust::tuple&)`.

Wave32 verdict: CORRECT. The hip_device_select stream-compaction produces the correct compacted set on wave32 -- no lost or duplicated entries, no data corruption, no HIP fault.

No regression on the prior-passing suites (STATIC_SET sub-word CAS shim, tiled probing CG, MULTISET/MULTIMAP, UTILITY, ROARING). Clean exit on all runs.

Result: PASS. linux-gfx1100 revalidate -> completed. validated_sha = 47ae24da1c2c5f21fcb88decd57697540af34a01. Fork untouched (no commit, no push).

## Head-drift reconciliation 2026-06-07 (validator, linux-gfx90a)

**Drift commit:** `0fb53f8811f50f0e5e9808474e9a4c29c273455f` -- "[ROCm] cuco: gate pre-Volta slot guards on __CUDA_ARCH__ defined"

Fork moat-port HEAD had advanced one functional commit beyond the recorded validated_sha (`47ae24da`) without being validated. The deferred-work item `cucollections-head-drift` (data/deferred.json) flagged this.

**Delta:** 1 file, `include/cuco/detail/open_addressing/open_addressing_ref_impl.cuh`, +7/-3 lines. Three `#if __CUDA_ARCH__ < 700` guards gating `insert_and_find` slot-size restrictions and `cas_dependent_write` were changed to `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700`. On HIP, `__CUDA_ARCH__` is undefined, so the bare comparison previously folded to `0 < 700` (true), incorrectly applying NVIDIA pre-Volta restrictions on AMD targets (manifested as a static_assert failure when cuGraph stored a 12-byte slot). The fix adds `defined(__CUDA_ARCH__)` guards so HIP (and NVIDIA Volta+) take the modern path.

**Classification:** `mixed`, `arch_independent=False` (moatlib.py classify verdict). Cannot carry forward -- requires real-GPU revalidation.

**Build (gfx90a):** Rebuilt all 7 test targets from the existing `projects/cuCollections/build-hip` build dir (37 TUs recompiled for the one changed source file via ninja incremental build). No errors, warnings only.

```
utils/timeit.sh cuCollections compile -- cmake --build projects/cuCollections/build-hip \
  --target STATIC_SET_TEST STATIC_MAP_TEST STATIC_MULTISET_TEST \
           STATIC_MULTIMAP_TEST DYNAMIC_MAP_TEST UTILITY_TEST ROARING_BITMAP_TEST -j$(nproc)
```

**Test run (real GPU, HIP_VISIBLE_DEVICES=1, GCD 1 idle, MI250X/gfx90a, ROCm 7.2.1):**

```
utils/timeit.sh cuCollections test -- env HIP_VISIBLE_DEVICES=1 <exe> --rng-seed 12345
```

| suite | cases | assertions | result |
|-------|-------|------------|--------|
| STATIC_SET_TEST | 97 | 887 | PASS |
| STATIC_MAP_TEST | 126 | 818 | PASS |
| STATIC_MULTISET_TEST | 70 | 582 | PASS |
| STATIC_MULTIMAP_TEST | 72 | 228 | PASS |
| DYNAMIC_MAP_TEST | 18 | 254 | PASS |
| UTILITY_TEST | 38 | 1561 | PASS |
| ROARING_BITMAP_TEST | 2 (1 pass + 1 skip) | 4 | PASS |
| **TOTAL** | **~423** | **~4334** | **all passing** |

No regression on the prior validated suite. All documented deferrals unchanged. No new failures.

Result: PASS. linux-gfx90a revalidate -> completed, validated_sha = 0fb53f8811f50f0e5e9808474e9a4c29c273455f. linux-gfx1100 stays revalidate (must revalidate 0fb53f8 on the gfx1100 host; delta is a preprocessor-only change, arch-agnostic, but not binary-equivalent so carry-forward is not appropriate without building both SHAs on gfx1100). Deferred item cucollections-head-drift closed.
