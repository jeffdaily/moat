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

## Library-level walls (NOT cuco-port defects; documented, scoped out of the validated subset)

1. **rocPRIM `DeviceSelect`/`DeviceTransform`/partition over `cuco::pair`**:
   `static_map`/`dynamic_map` `retrieve_all` compacts paired (key+value) slots
   via `cub::DeviceSelect::If` over a `cuco::pair` (a `cuda::std::tuple`).
   rocPRIM's `partition_scatter` cannot assign `cuda::std::tuple` through its zip
   output (`no viable operator=` / `no type named value_type in
   iterator_traits<cuda::std::tuple<...>>`). `static_set` is unaffected (scalar
   slot -> retrieve_all works). Blocks the map TUs that call `retrieve_all`
   (contains_test, duplicate_keys, insert_or_assign, insert_or_apply,
   retrieve_if, dynamic_map retrieve_all/multiplicity) and hyperloglog
   (DeviceTransform over tuple iterators).
2. **>8-byte atomic CAS**: `static_map::insert_and_find` for a >8B non-packable
   slot spin-waits on the payload (needs ITS); cuco `static_assert`s it out on
   pre-Volta and CDNA2 has no ITS, so the assert correctly fires. Likewise a
   16-byte composite key (`key_pair<int64>` in static_multimap
   heterogeneous_lookup) lowers to LLVM "unsupported cmpxchg" on gfx90a (no
   128-bit atomic CAS; that is sm_90+).
3. **libhipcxx `<nv/target>` `NV_DISPATCH_TARGET`**: the bloom_filter
   `default_filter_policy_impl` host-side input-validation block inside
   `NV_DISPATCH_TARGET(NV_IS_HOST, (...))` mis-parses under libhipcxx's nv/target
   port (`type name does not allow constexpr specifier`). Blocks BLOOM_FILTER.

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
instead of brace-init (clang narrowing), and `.template retrieve_if<N>` on
dependent refs.

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
| STATIC_MAP_TEST       | 82    | 526        | PASS (FULL incl int8/int16 -- was 72 core before the sub-word shim) |
| STATIC_MULTISET_TEST  | 70    | 582        | PASS (full) |
| STATIC_MULTIMAP_TEST  | 72    | 228        | PASS |
| DYNAMIC_MAP_TEST      | 12    | 144        | PASS (core: insert/find/erase) |
| UTILITY_TEST          | 38    | 1561       | PASS (hash, probing_scheme, extent, storage, fast_int, next_prime) |
| ROARING_BITMAP_TEST   | 2     | 4          | PASS (1 skipped -- needs test-data download) |
| **TOTAL**             | **~373** | **~3932** | **all passing** |

Determinism confirmed (static_set incl. small keys bit-identical across runs,
`--rng-seed 12345` x2). Deferred:
the rocPRIM-pair-DeviceSelect TUs (retrieve_all-over-pairs), the >8B/16B atomic
CAS TUs, the libhipcxx-nv/target bloom_filter, and DYNAMIC_BITSET (separate
non-get_wrapper compile error, not investigated -- a trie bitset, off the
hashtable core path). The four CCCL-native concurrent hashtables cudf needs
(static_set/map/multiset/multimap) are GPU-validated.

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
