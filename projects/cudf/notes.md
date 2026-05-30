# cudf notes (ROCm/HIP port)

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated
RAPIDS). `depends_on: [rmm]` (rmm is COMPLETED). cudf is itself a dependency of
cugraph and cuml, so a future successful port needs a `## Install as a
dependency` section -- NOT written yet because the port is BLOCKED (see below).

Pinned upstream: tag `v25.08.00` (in `projects/cudf/src`, gitignored), matching
the rmm v25.08.00 line.

## State: BLOCKED on linux-gfx90a

Reason (one line): libcudf core cannot reach a GPU-validatable HIP build in a
focused run -- it hard-depends on **cuCollections** (a CCCL-native concurrent
hash-table library with NO ROCm fork) for nearly every non-trivial algorithm,
on **jitify/NVRTC** for binaryop/transform/rolling, and the 404-file C++20
monolithic core triggers a header-tree-wide `__CUDACC__`/`CUDF_HOST_DEVICE`
macro cascade that fails to compile at the very FIRST/foundational TU. Arrow is
NOT the blocker (see below).

## External-dependency assessment (the feasibility gate)

cudf's `cpp/CMakeLists.txt` (deps block lines 247-330) pulls, in order:

| Dependency        | How fetched                          | HIP disposition |
|-------------------|--------------------------------------|-----------------|
| CUDAToolkit       | `rapids_find_package REQUIRED`       | -> HIP runtime (the core CUDA->HIP work) |
| ZLIB, Threads     | system / conda                       | OK (host libs)  |
| rapids_logger     | rapids-cpm                           | OK (proven portable by rmm: PORTING_GUIDE.md:209) |
| jitify            | get_jitify.cmake                     | **NVRTC JIT, no HIP drop-in** (binaryop/transform/rolling; 12 files) |
| NVTX              | get_nvtx.cmake                       | stub-able / rocTX; pervasive (226 files include it) |
| **nvCOMP**        | get_nvcomp.cmake, **proprietary binary** | **UNPORTABLE** -- NVIDIA-only `.so`; GPU compression. Confined to `src/io/` (13 files), so disabling cuIO removes it cleanly |
| CCCL              | get_cccl.cmake (rapids-cpm)          | redirect to rocThrust/hipCUB/libhipcxx (rmm pattern, PORTING_GUIDE.md:207) |
| rmm               | get_rmm.cmake                        | DONE (consumed via find_package, works -- see below) |
| flatbuffers       | get_flatbuffers.cmake                | OK (host lib)   |
| dlpack            | get_dlpack.cmake                     | OK (header)     |
| **cuCollections** | get_cucollections.cmake (rapids-cpm) | **NO ROCm fork; CCCL-native concurrent hashtable**. Used by join/groupby/hash/stream_compaction/search/reductions (23 core files). A major sub-port. |
| gtest             | get_gtest.cmake                      | OK (conda)      |
| KvikIO            | get_kvikio.cmake                     | GPUDirect Storage (cuFile); confined to `src/io/` (4 files) |
| **nanoarrow**     | get_nanoarrow.cmake                  | OK -- small vendored C lib (apache/arrow-nanoarrow), NOT Apache Arrow C++ |
| thread_pool, zstd | get_*.cmake                          | OK (host libs)  |

### Arrow is NOT the blocker (key finding)

The widely-assumed "libcudf needs Apache Arrow C++" wall does **not** apply at
v25.08. `cpp/cmake/thirdparty/get_arrow.cmake` exists but is **dead** -- the
main `cpp/CMakeLists.txt` does NOT `include()` it (verified: no
`get_arrow`/`find_package(Arrow)`/`rapids_find_package(Arrow)` anywhere in
cpp/CMakeLists.txt). The interop layer (to_arrow/from_arrow) is built on
**nanoarrow** -- a tiny standalone C library vendored via CPM and namespaced
`cudf` -- which builds with plain CMake and has no CUDA content, so it is
satisfiable for a HIP build. cudf migrated off heavyweight libarrow to
nanoarrow before this release. So Arrow does not gate the port; the gate is
cuCollections + jitify + the CCCL/macro cascade + the 404-file C++20 scale.

### The two genuinely unportable / unported deps

1. **nvCOMP** -- NVIDIA proprietary binary (`rapids_cpm_nvcomp(... USE_PROPRIETARY_BINARY)`),
   no AMD build exists. GPU (de)compression for Parquet/ORC/text. ENTIRELY
   inside `src/io/` (0 includes outside io/), so it is cleanly removable by
   disabling cuIO -- it is NOT a core blocker, it is a cuIO blocker.
2. **cuCollections** -- `github.com/NVIDIA/cuCollections`, header-only but
   built ENTIRELY on libcudacxx (`cuda::atomic` thread-scoped RMW,
   `cuda::std::atomic_ref`, cooperative-group probing, `cuda::std` everywhere).
   **No `ROCm/cuCollections` fork exists** (checked: repo not found). cudf uses
   `cuco::static_set` (19), `static_map` (8), `static_multimap` (4), with
   `cuco::op`, `*_ref` types, `thread_scope_block/device/thread`,
   `insert_and_find`, `linear_probing`, `extent`, `pair`. This is used by
   join (1), groupby (7), hash, stream_compaction (2), search (1),
   reductions (1), text (3) -- i.e. essentially every interesting algorithm.
   Porting cuco to HIP is a substantial project in its own right (comparable to
   a separate MOAT target), NOT something to inline here. This is the dominant
   blocker for a meaningful core.

## rmm-as-a-dependency recipe: WORKED (no gaps)

The `projects/rmm/notes.md` "## Install as a dependency" recipe worked exactly
as written and validates the MOAT dependency mechanism end-to-end:

```
git clone --depth 1 -b moat-port https://github.com/jeffdaily/rmm _deps/cudf-rmm/src
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S _deps/cudf-rmm/src/cpp -B _deps/cudf-rmm/build -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/cudf-rmm/install
cmake --build _deps/cudf-rmm/build --target install -j$(nproc)
```

Result: clean configure (2.5s) + build/install (exit 0). `rmm::rmm` exported in
`_deps/cudf-rmm/install/lib/cmake/rmm/`; vendored libhipcxx (`include/cuda/`)
and rapids_logger bundled into the prefix as documented; `librmm.so` +
`librapids_logger.so` present. A cudf HIP build would consume it with
`find_package(rmm)` + `-DCMAKE_PREFIX_PATH=/var/lib/jenkins/moat/_deps/cudf-rmm/install;/opt/rocm;$CONDA_PREFIX`.
No gaps in the rmm dependency contract. The clone went into a CONSUMER-PRIVATE
prefix (`_deps/cudf-rmm/`), NOT shared `_deps/rmm/`, because a concurrent RAPIDS
port (raft) is using `_deps/raft-rmm/` -- the per-consumer prefix convention
works.

## Empirical wall (probe evidence, gfx90a, ROCm 7.2.1)

Probed the SINGLE SMALLEST, most foundational core TU --
`cpp/src/column/column_device_view.cu` (176 lines; `column_device_view` is the
base type every cudf operation builds on). Compiled standalone as HIP against
the rmm install + libhipcxx + rocThrust on the include path
(`agent_space/cudf_probe/`):

1. First wall: `CUDF_HOST_DEVICE` (cudf's own `__host__ __device__` macro,
   `cpp/include/cudf/types.hpp:19`) is gated on `#ifdef __CUDACC__`. hipcc
   defines `__HIPCC__`, NOT `__CUDACC__`, so the macro expands to EMPTY ->
   every host/device member (e.g. all `string_view` ctors) becomes host-only ->
   "call to __host__ function from __device__ function" pulled in transitively
   via column_device_view -> string_view. Reaches 71 headers. Same fault family
   as PORTING_GUIDE.md:157 (define __CUDACC__ when __HIPCC__).
2. Applying the blanket `#define __CUDACC__ 1` workaround (prelude) clears that
   but CASCADES: `type_dispatcher.hpp` `base_type_to_id` template
   specializations then take a CUDA-specific path and fail ("no function
   template matches", ~20 errors) -- the MPPI-Generic lesson
   (PORTING_GUIDE.md:158) that defining __CUDACC__/__CUDA_ARCH__ for HIP has
   cascading effects requiring per-site guards, not a blanket define.

There are **46 `__CUDACC__`/`__CUDA_ARCH__` guard sites across 22 headers**, each
needing the HIP-aware `|| __HIP_DEVICE_COMPILE__` / `|| __HIPCC__` treatment by
hand. And that is just layer 1 -- below it sit the cuda:: libcudacxx reach
(270 files), cooperative_groups (31 files; HIP CG lacks cg::reduce /
labeled_partition / memcpy_async per PORTING_GUIDE.md:185,203), and then
cuCollections, before any algorithm compiles.

Conclusion: not even the trivial column core compiles within a focused run;
the dominant blocker (cuCollections, no ROCm port) makes the interesting core
(join/groupby/hash/reductions) unreachable regardless. This is the documented
"blocked on X" good outcome.

## Scale (for the record)

- `cpp/src`: **493 source files** in the single monolithic `add_library(cudf ...)`
  (cpp/CMakeLists.txt:339-830) -- 404 `.cu` + 89 `.cpp`.
- Of those, `src/io/` (cuIO: csv/json/parquet/orc/avro/text + comp) is **89
  files** and the home of nvCOMP+KvikIO -- the natural module to disable first.
- C++20, `CXX_EXTENSIONS ON`, heavy template metaprogramming, NVTX in 226 files.

## If resumed later (path to unblock)

Prerequisite ordering for a real cudf HIP port:
1. Port **cuCollections** to ROCm/HIP first (its own MOAT target): static_set/
   static_map/static_multimap + refs on libhipcxx `cuda::atomic`/`cuda::std`.
   This is the long pole. Without it only a tiny non-hashing subset is buildable.
2. Replace **jitify** (NVRTC) usage with hiprtc, or build those kernels AOT, or
   defer binaryop-JIT/transform-JIT/rolling-JIT.
3. Apply the rmm standalone-HIP bypass (`option(USE_HIP) ... include(cmake/hip/cudf_hip.cmake) return()`)
   so the rapids-cmake/CPM CCCL fetch is skipped; satisfy CCCL from /opt/rocm
   (rocThrust/hipCUB) + the rmm-bundled libhipcxx; consume rmm via find_package.
4. Make `CUDF_HOST_DEVICE` / `CUDF_KERNEL` (types.hpp) and all 46
   `__CUDACC__`/`__CUDA_ARCH__` guard sites HIP-aware
   (`__CUDACC__||__HIPCC__`, `__CUDA_ARCH__||__HIP_DEVICE_COMPILE__`).
5. Disable cuIO (drop the 89 `src/io/` files + nvCOMP + KvikIO + zstd-comp) and
   scope to column/table/copying/sort/reductions/unary/etc. for first GPU
   validation; nanoarrow interop is fine to keep (no Arrow C++ needed).
