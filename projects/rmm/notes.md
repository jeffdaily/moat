# rmm notes (ROCm/HIP port)

Upstreaming unlikely (NVIDIA-affiliated RAPIDS); kept for the MOAT automation
exercise and -- critically -- as the FOUNDATIONAL RAPIDS base that raft, cudf,
cuvs, cugraph, and cuml all build on. The `## Install as a dependency` section
below is the contract the rest of RAPIDS consumes.

Pinned upstream: tag `v25.08.00` (in `projects/rmm/src`, gitignored). Chosen as
a recent stable release whose CCCL surface matches the vendored ROCm/libhipcxx
(`cuda::std` + `cuda::mr`), and whose memory/arena path uses only the CUDA
runtime stream-ordered allocator -- no CUDA driver-API VMM (`cuMem*`) calls,
which simplifies the port substantially.

## Existing AMD support

None. `grep -riE 'hip|rocm|amdgpu|gfx9'` over the whole tree returns nothing; no
ROCm branch/fork upstream. From-scratch CUDA-to-HIP, Strategy A.

## Build classification: Strategy A (header-heavy pure-CMake library)

rmm is a header-only C++ library (device memory resources + containers under
`cpp/include/rmm/`) plus a tiny `librmm.so` compiled from 6 HOST-C++ `.cpp`
files (`cpp/src/*.cpp`) -- none are `.cu`; rmm's library ships no device kernels
of its own. (The `.cu` files exist only in `cpp/tests` and `cpp/benchmarks`.)
Not a torch extension -> Strategy A. The headers are the deliverable RAPIDS
links against.

## CUDA surface and HIP mapping

Entirely the CUDA RUNTIME API (`cuda*`); zero driver-API (`cu*`) symbols. The
compat header `cpp/include/rmm/detail/hip/cuda_to_hip.h` aliases every symbol;
all are 1:1 except enum-spelling deltas:
- Device-attribute enums: CUDA `cudaDevAttr*` -> HIP `hipDeviceAttribute*`
  (e.g. `cudaDevAttrMemoryPoolsSupported` -> `hipDeviceAttributeMemoryPoolsSupported`).
- Host alloc: `cudaMallocHost`/`cudaHostAlloc`/`cudaFreeHost` ->
  `hipHostMalloc`/`hipHostFree`; `cudaHostAllocDefault` -> `hipHostMallocDefault`.
- Alloc error code: `cudaErrorMemoryAllocation` -> `hipErrorOutOfMemory`.
- Stream-ordered pool (`hipMemPoolProps` fields `allocType/handleTypes/
  location.{type,id}/maxSize` match CUDA exactly), prefetch/advise, pointer
  attributes (`hipPointerAttribute_t.type`), handle/location/memory-type enums:
  all present in HIP, plain `cuda*`->`hip*`.

CUDA-12-only symbols HIP lacks -- already dead/guarded in rmm, so no block:
- `cudaMemFabricHandle` / `cudaMemHandleTypeFabric`: rmm's `fabric` enumerator
  is the literal `0x8`, not the CUDA symbol; the runtime reports it unsupported
  and `AsyncMRFabricTest.FabricHandlesSupport` GTEST_SKIPs (confirmed on gfx90a).
- `cudaMemPoolCreateUsageHwDecompress` (CUDA 12.8+): all uses sit behind
  `#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080`; `CUDA_VERSION` is
  undefined under HIP so the code is dead. `HWDECOMPRESS_TEST` is not built.

## CCCL redirect (the main obstacle) -- how it was solved

rmm needs all three CCCL components and rmm's stock build FETCHES NVIDIA CCCL
via rapids-cmake/CPM at configure time (`cpp/CMakeLists.txt` bootstraps
`rapidsai/rapids-cmake`, then `rapids_cpm_init()` + `get_cccl`, gated on
`CUDAToolkit REQUIRED` + `enable_language(CUDA)`).

**Solution: bypass the rapids-cmake/CPM bootstrap entirely with a standalone HIP
CMake build, and satisfy CCCL from include paths -- no fetch.** This works
because on ROCm the CCCL trio is already available header-only:
- **Thrust -> rocThrust**: drop-in at `/opt/rocm/include/thrust` (hipcc's
  default include path). One namespace delta: rmm's `exec_policy.hpp` uses
  `thrust::cuda::par[_nosync]` and `thrust::cuda_cub::execute_on_stream
  [_nosync]_base`; rocThrust spells these `thrust::hip::par[_nosync]` and
  `thrust::hip_rocprim::execute_on_stream[_nosync]_base`, and the CUDA backend
  header `<thrust/system/cuda/execution_policy.h>` hard-fails on ROCm
  (`cub/detail/detect_cuda_runtime.cuh` not found). Fixed in `exec_policy.hpp`
  with a `USE_HIP`-guarded include swap + two backend namespace ALIASES
  (`thrust_backend`/`thrust_backend_cub`) so the policy bodies stay identical
  and the CUDA path is byte-for-byte unchanged. (`thrust_allocator_adaptor.hpp`
  uses only backend-agnostic Thrust -- no change.)
- **CUB -> hipCUB**: drop-in at `/opt/rocm/include/hipcub` (used only
  transitively via Thrust).
- **libcudacxx / `cuda::std` AND `cuda::mr` -> ROCm/libhipcxx**: VENDORED (not
  in the ROCm release). rmm uses `cuda::mr` heavily (resource_ref,
  device_accessible, forward_property, has_property), `<cuda/memory_resource>`,
  `<cuda/stream_ref>`, and `cuda::std::*`. The libhipcxx clone at
  `agent_space/libhipcxx` (branch `amd-develop`, the commit validated in
  findings/libhipcxx/NOTES.md) provides ALL of these. Two requirements to
  activate it: add `-I<clone>/include`, AND define
  `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE` (rmm's headers `#error`
  without it, and it is what exposes `cuda::mr`).

The standalone HIP build lives in `cpp/cmake/hip/rmm_hip.cmake` (+
`rmm_hip_tests.cmake`), reached by a 6-line `if(USE_HIP) include(...) return()`
guard at the top of `cpp/CMakeLists.txt`. It generates the two headers
rapids-cmake would (`version_config.hpp`, `logger_macros.hpp` via
rapids-logger's own `create_logger_macros.cmake` template), compiles
`rapids_logger` from its `src/logger.cpp` (its public header hides spdlog; only
that TU needs spdlog), marks the 6 rmm `.cpp` `LANGUAGE HIP`, force-includes
the compat header, links `hip::host` (NOT `hip::device` -- avoids leaking
`--offload-arch` to host consumers, the cupoch lesson), and exports `rmm::rmm`
+ a minimal `rmm-config.cmake`.

Forwarding shim headers in `cpp/hip_compat/` (HIP include path only) named
`cuda_runtime_api.h`, `cuda_runtime.h`, `driver_types.h`, `cuda.h` -- each just
`#include`s the compat header -- so rmm's own `#include <cuda_runtime_api.h>`
(and a test's `<driver_types.h>`) resolve without editing every source. On
NVIDIA this dir is absent so the real toolkit headers win.

## Fault classes

- Warp size: N/A (rmm has no `__shfl/__ballot/warpSize`; it is allocator/
  container code). The real work was the memory-API enum spellings + the CCCL
  redirect + the Thrust namespace delta.
- One HIP-vs-CUDA runtime ROBUSTNESS diff (not a defect): `hipStreamSynchronize`
  on a destroyed stream handle returns `hipSuccess` on gfx90a (HIP tolerates the
  freed handle), whereas CUDA errors and trips rmm's `assert()`. This makes the
  `CudaStreamDeathTest.TestSyncNoThrow` death test (which expects the process to
  abort) not hold on HIP. `synchronize_no_throw()`'s own contract (never throw)
  is correct on HIP. Guarded that one death test out under USE_HIP in
  `cpp/tests/cuda_stream_tests.cpp` with an explanatory comment. Standalone
  probe confirming the behavior: `agent_space/stream_destroy_probe.cpp`.

## Build (lead, gfx90a)

Dependencies (installed via conda into the `py_3.12` env): `gtest`/`gmock`
(1.17), `spdlog`+`fmt`. Vendored clones in `agent_space/`: `libhipcxx`
(`amd-develop`) and `rapids_logger` (`release/0.2.0`).

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/rmm/src/cpp -B <build> -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/conda/envs/py_3.12" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DCMAKE_INSTALL_PREFIX=<prefix> -DBUILD_TESTS=ON
cmake --build <build> -j$(nproc)
```

Arch is taken from `CMAKE_HIP_ARCHITECTURES` (defaults to gfx90a only when
unset), so a follower validates with only `-DCMAKE_HIP_ARCHITECTURES=<arch>`
and no source/CMake edit.

## Validation (gfx90a, HIP_VISIBLE_DEVICES=2)

Full gtest suite built (27 of the stock 28 test executables; HWDECOMPRESS
deferred) and run on real hardware via `ctest` (serial -- single GPU):

```
cd <build> && HIP_VISIBLE_DEVICES=2 ctest --output-on-failure
```

Result: **100% tests passed, 0 failed out of 27** (~158 s); **658 individual
gtest cases** total. Coverage spans every memory resource and container:
- Resources: cuda, managed/system, **cuda_async (stream-ordered pool,
  `hipMallocAsync`/`hipMemPool_t`)**, pool (suballocator), **arena (42 cases)**,
  aligned, binning, fixed_size (via pool/binning), callback, plus the adaptors
  (limiting, logging/statistics, tracking, failure_callback, prefetch,
  thread_safe, aligned, stream-allocator, polymorphic/thrust allocator).
- Containers: device_buffer (46), device_uvector (100), device_scalar (56),
  device_check / multidevice.
- Streams/events, prefetch, error macros, logger.
- Real GPU allocation/free confirmed (DEVICE_BUFFER_TEST 46/46; POOL_MR 2.2 s of
  live alloc; ARENA 2.4 s). The only SKIP is `AsyncMRFabricTest` (HIP has no
  fabric IPC handle -- expected, documented above).

Deferred (documented, off the validated path): `HWDECOMPRESS_TEST` (CUDA-12.8
hw decompression, no HIP equivalent); the per-thread-default-stream (PTDS)
duplicate test variants (same tests, different stream mode -- not new resource
coverage); the Python/Cython package (scikit-build, heavy RAPIDS wheel
machinery -- the C++ library + headers are what RAPIDS links). The CUDA
driver-API VMM arena path does not exist in v25.08.00 (runtime pool only), so
nothing VMM was deferred.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: linux-gfx1100 (2x AMD Radeon Pro W7800 48GB, gfx1100 RDNA3, wave32); ROCm 7.2.1; HIP clang 22.0.0. Follower validation against head_sha 1473ffc5bab2b81efd7d849db55e13a62b08822f. Fork untouched (no commits, no CI workflow added).

### Build command

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/rmm/src/cpp -B projects/rmm/build-gfx1100 -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/conda/envs/py_3.12" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/projects/rmm/install-gfx1100 \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/rmm/build-gfx1100 -j$(nproc)
```

Configure: 2.1 s; build: 16.8 s (65 targets, warnings only -- no errors). Conda deps installed fresh: gtest/gmock 1.17, spdlog 1.17, fmt 12.1. Vendored libhipcxx cloned at fa4ccc6 (amd-develop), rapids_logger at 4c72b59 (release/0.2.0).

### gfx1100 code-object evidence

```
roc-obj-ls projects/rmm/build-gfx1100/gtests/DEVICE_BUFFER_TEST
```
Output:
```
1  host-x86_64-unknown-linux-gnu-       ...#offset=282624&size=0
1  hipv4-amdgcn-amd-amdhsa--gfx1100    ...#offset=282624&size=3035920
```
No gfx90a code object present; all 27 test binaries target gfx1100 exclusively. `CMAKE_HIP_ARCHITECTURES` flows correctly from the CMake flag through `HIP_ARCHITECTURES` target property (no hardcoded arch in the HIP cmake files).

### Test results

```
HIP_VISIBLE_DEVICES=0 ctest --test-dir projects/rmm/build-gfx1100 --output-on-failure
```
Result: **100% tests passed, 0 failed out of 27** (68.9 s wall time).

Individual gtest case counts (658 total -- matches gfx90a exactly):

| Test binary          | Passed | Skipped | Failed |
|----------------------|--------|---------|--------|
| ADAPTOR_TEST         | 32     | 0       | 0      |
| ALIGNED_TEST         | 9      | 0       | 0      |
| ARENA_MR_TEST        | 42     | 0       | 0      |
| BINNING_MR_TEST      | 2      | 0       | 0      |
| CALLBACK_MR_TEST     | 2      | 0       | 0      |
| CONTAINER_MULTIDEVICE_TEST | 12 | 0     | 0      |
| CUDA_ASYNC_MR_TEST   | 3      | 1       | 0      |
| CUDA_STREAM_TEST     | 12     | 0       | 0      |
| DEVICE_BUFFER_TEST   | 46     | 0       | 0      |
| DEVICE_MR_REF_TEST   | 204    | 5       | 0      |
| DEVICE_SCALAR_TEST   | 56     | 0       | 0      |
| DEVICE_UVECTOR_TEST  | 100    | 0       | 0      |
| ERROR_MACROS_TEST    | 11     | 0       | 0      |
| FAILURE_CALLBACK_TEST| 2      | 0       | 0      |
| HOST_MR_REF_TEST     | 23     | 0       | 0      |
| LIMITING_TEST        | 5      | 0       | 0      |
| LOGGER_TEST          | 7      | 0       | 0      |
| PINNED_POOL_MR_TEST  | 6      | 0       | 0      |
| POLYMORPHIC_ALLOCATOR_TEST | 10 | 0    | 0      |
| POOL_MR_TEST         | 10     | 0       | 0      |
| PREFETCH_ADAPTOR_TEST| 5      | 0       | 0      |
| PREFETCH_TEST        | 6      | 0       | 0      |
| STATISTICS_TEST      | 7      | 0       | 0      |
| STREAM_ADAPTOR_TEST  | 8      | 0       | 0      |
| SYSTEM_MR_TEST       | 1      | 4       | 0      |
| THRUST_ALLOCATOR_TEST| 12     | 6       | 0      |
| TRACKING_TEST        | 9      | 0       | 0      |
| **TOTAL**            | **642**| **16**  | **0**  |

### Skip set vs gfx90a

Identical to gfx90a -- no difference:
- `AsyncMRFabricTest.FabricHandlesSupport` (CUDA_ASYNC_MR_TEST, 1 skip): HIP has no fabric IPC handle; GTEST_SKIP at runtime.
- `ResourceTests/mr_ref_test.*/System` (DEVICE_MR_REF_TEST, 5 skips): system/managed resource not supported on this device config.
- `SystemMRTest.*` (SYSTEM_MR_TEST, 4 skips): HeadroomMR + FirstTouch variants -- system memory resource tests that skip when conditions not met.
- `ThrustAllocatorTests/allocator_test.multi_device/*` (THRUST_ALLOCATOR_TEST, 6 skips): multi-device variants skip when only 1 device is visible.
- `CudaStreamDeathTest.TestSyncNoThrow`: guarded out at the source level under USE_HIP (not listed by gtest_list_tests) -- same as gfx90a.

### Determinism

Allocator tests (POOL_MR_TEST, ARENA_MR_TEST, DEVICE_BUFFER_TEST) produced identical pass counts across multiple ctest invocations; no flakiness observed.

### Verdict

PASS. gfx1100 validation matches gfx90a identically (27/27 ctest, 642 passed + 16 skipped + 0 failed out of 658 gtest cases). Real GPU allocation/stream-ordered ops confirmed on AMD Radeon Pro W7800 (gfx1100). State: port-ready -> completed (validated_sha = 1473ffc5bab2b81efd7d849db55e13a62b08822f).

## Install as a dependency

This is the contract raft/cudf/cuvs/cugraph/cuml consume. rmm is header-heavy:
a dependent gets the rmm headers, the vendored libhipcxx headers, rapids_logger,
and links `librmm.so` + `librapids_logger.so`.

### 1. Build + install rmm (ROCm/gfx90a)

Prereqs once per host: `conda install -n <env> -c conda-forge gtest gmock
spdlog fmt`; clone the two vendored deps into a stable location (NOT a build
dir):

```
git clone --depth 1 --branch amd-develop https://github.com/ROCm/libhipcxx <X>/libhipcxx
git clone --depth 1 --branch release/0.2.0 https://github.com/rapidsai/rapids-logger <X>/rapids_logger
```

Configure + build + install (BUILD_TESTS=OFF for a dependency build):

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S <rmm>/cpp -B <rmm>/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=<X>/libhipcxx/include \
  -DRAPIDS_LOGGER_SOURCE_DIR=<X>/rapids_logger \
  -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/rmm/install
cmake --build <rmm>/build-hip --target install -j$(nproc)
```

(For the MOAT deps workflow: clone `jeffdaily/rmm @ moat-port` into
`_deps/rmm/src`, apply the working-tree port if delivering via fork, and install
to `_deps/rmm/install` per the above.)

### 2. Install-prefix layout

```
<prefix>/include/rmm/...                 rmm headers (+ generated version_config.hpp, logger_macros.hpp)
<prefix>/include/rmm/detail/hip/cuda_to_hip.h   the compat header (force-included into HIP consumers)
<prefix>/include/rmm/hip_compat/...      cuda_runtime_api.h etc. forwarding shims
<prefix>/include/cuda/...                VENDORED ROCm/libhipcxx (cuda::std / cuda::mr / <cuda/memory_resource>)
<prefix>/include/rapids_logger/...       rapids_logger headers
<prefix>/lib/librmm.so, librapids_logger.so
<prefix>/lib/cmake/rmm/rmm-config.cmake  + rmm-targets.cmake (+ version)
```

The libhipcxx headers are vendored INTO the rmm install prefix, so a dependent
gets `cuda::std` / `cuda::mr` from `find_package(rmm)` with no separate
libhipcxx flag.

### 3. What a dependent (raft/cudf) sets to consume it

```
find_package(rmm REQUIRED)            # resolves via CMAKE_PREFIX_PATH below
target_link_libraries(<tgt> PRIVATE rmm::rmm)
```

Configure the dependent with:
```
-DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/rmm/install;/opt/rocm;$CONDA_PREFIX"
```

The exported `rmm::rmm` target propagates, automatically:
- include dirs: rmm headers, the `hip_compat` shim dir, and the vendored
  libhipcxx `cuda/` headers (so `cuda::std`/`cuda::mr` resolve);
- compile defs: `USE_HIP`, `__HIP_PLATFORM_AMD__`,
  `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE`;
- a HIP-language force-include of the installed `cuda_to_hip.h` (so the
  dependent's `.cu`/HIP TUs that pull rmm headers get the `cuda*`->`hip*`
  aliasing automatically);
- link: `hip::host`, `rapids_logger::rapids_logger`, `${CMAKE_DL_LIBS}`.

The dependent must itself `enable_language(HIP)` and compile any TU that
includes rmm headers as HIP (`set_source_files_properties(... LANGUAGE HIP)`),
because rmm headers reach rocThrust/hipCUB/libhipcxx device headers that only
clang-as-hipcc parses. rocThrust/hipCUB/rocPRIM come from `/opt/rocm/include`
(hipcc's default path) -- no extra flags. spdlog must be findable
(`find_dependency(spdlog)` in rmm-config) -- keep the conda env's
`CMAKE_PREFIX_PATH` entry. At runtime put `<prefix>/lib` and `$CONDA_PREFIX/lib`
(libspdlog) on `LD_LIBRARY_PATH`.

### 4. Verified

A standalone consumer (`agent_space/rmm_consumer/`) doing exactly the above --
`find_package(rmm)` + `rmm::rmm`, a `pool_memory_resource<cuda_memory_resource>`
feeding a `rmm::device_uvector<int>`, set/get element on a GPU stream --
compiles against the install tree (only `CMAKE_PREFIX_PATH` set) and runs
correctly on gfx90a:
`consumer: device_uvector[0] = 42 (expect 42) via find_package(rmm)+rmm::rmm`.
