# rmm (RAPIDS Memory Manager) -- ROCm/HIP port plan

## Existing AMD support assessment

rmm v25.08.00 (cloned at `projects/rmm/src`, checked out tag `v25.08.00`) has
**zero** AMD/HIP/ROCm references anywhere in the tree (`grep -riE
'hip|rocm|amdgpu|gfx9'` over `*.cpp/*.hpp/*.cu/*.cmake/CMakeLists.txt` returns
nothing). No ROCm branch/fork upstream. This is a from-scratch CUDA-to-HIP port.

## Build classification: Strategy A (pure CMake, header-heavy library)

- Pure CMake project (`cpp/CMakeLists.txt`), no `find_package(Torch)`, no
  `torch.utils.cpp_extension`. -> Strategy A.
- The CORE DELIVERABLE is the **header-only C++ library** under
  `cpp/include/rmm/` (device memory resources + containers). A small compiled
  `librmm.so` is built from only 6 host-C++ `.cpp` files (`cpp/src/*.cpp`) --
  NONE are `.cu`; rmm's library has no device kernels of its own. The only
  `.cu` files in the tree are in `cpp/tests` and `cpp/benchmarks` (they
  instantiate device containers / thrust algorithms).
- This is the RAPIDS base library: raft, cudf, cuvs, cugraph, cuml all link
  `rmm::rmm`, so the `## Install as a dependency` recipe in notes.md is as
  important as the port.

## CUDA surface (from grep of cpp/{include,src,tests})

Entirely the CUDA **runtime** API (`cuda*`); **no driver API (`cu*`) symbols at
all** in this version. The arena/pool path uses the runtime stream-ordered
allocator (`cudaMallocAsync`/`cudaMemPool_t`), NOT the `cuMemCreate/cuMemMap`
VMM driver path -- so the VMM-arena concern in the brief does not arise here.

Symbol groups and HIP mapping (all 1:1 unless noted):
- Core memory: `cudaMalloc/Free`, `cudaMemcpy[Async]`, `cudaMemset[Async]`,
  `cudaMallocManaged`, `cudaMemGetInfo`, `cudaHostAlloc`/`cudaMallocHost`/
  `cudaFreeHost` -> `hip*` (host alloc: `hipHostMalloc`/`hipHostFree`).
- Stream/event: `cudaStream*`, `cudaEvent*`, `cudaStreamPerThread`,
  `cudaStreamLegacy` -> `hip*` (all present).
- Stream-ordered pool: `cudaMallocAsync`, `cudaFreeAsync`,
  `cudaMallocFromPoolAsync`, `cudaMemPool_t`, `cudaMemPoolCreate/Destroy/
  SetAttribute`, `cudaMemPoolProps`, `cudaDeviceGetDefaultMemPool` -> `hip*`
  (all present; `hipMemPoolProps` fields `allocType/handleTypes/location.{type,
  id}/maxSize` match CUDA exactly).
- Managed/prefetch: `cudaMemPrefetchAsync`, `cudaMemAdvise`,
  `cudaMemRangeGetAttribute`, `cudaMemAdviseSetPreferredLocation`,
  `cudaMemRangeAttributeLastPrefetchLocation` -> `hip*` (all present).
- Device attribute enums: spelling differs -- CUDA `cudaDevAttr*` -> HIP
  `hipDeviceAttribute*` (e.g. `cudaDevAttrMemoryPoolsSupported` ->
  `hipDeviceAttributeMemoryPoolsSupported`). Handle alias-by-alias.
- Handle-type / location / memory-type enums: `cudaMemHandleType{None,
  PosixFileDescriptor,Win32,Win32Kmt}`, `cudaMemAllocationTypePinned`,
  `cudaMemLocationTypeDevice`, `cudaMemoryType*` -> `hip*` (all present).
- Pointer attrs: `cudaPointerGetAttributes` / `cudaPointerAttributes` ->
  `hipPointerGetAttributes` / `hipPointerAttribute_t`; `.type` field name
  matches.

CUDA-12.x-only symbols that HIP lacks -- already dead/guarded in rmm, so they
do NOT block:
- `cudaMemFabricHandle` / `cudaMemHandleTypeFabric`: rmm's `fabric` enumerator
  is the literal `0x8` (not the CUDA symbol); the runtime reports it
  unsupported and `AsyncMRFabricTest` `GTEST_SKIP`s.
- `cudaMemPoolCreateUsageHwDecompress` (CUDA 12.8+): all uses are behind
  `#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080`; `CUDA_VERSION` is
  undefined under HIP, so the code is dead.

## CCCL (the main obstacle) -- ROCm trio + redirect

rmm depends on all three CCCL components:
- **Thrust** -> rocThrust: header drop-in at `/opt/rocm/include/thrust`. One
  namespace delta to guard: rmm's `exec_policy.hpp` /
  `thrust_allocator_adaptor.hpp` use `thrust::cuda::par[_nosync]` and
  `thrust::cuda_cub::execute_on_stream[_nosync]_base`; rocThrust spells these
  `thrust::hip::par[_nosync]` and `thrust::hip_rocprim::execute_on_stream
  [_nosync]_base`. USE_HIP-guard those two spellings.
- **CUB** -> hipCUB: header drop-in at `/opt/rocm/include/hipcub` (rmm uses CUB
  only transitively via Thrust; little/no direct `cub::`).
- **libcudacxx / `cuda::std` AND `cuda::mr`** -> ROCm/libhipcxx. rmm uses
  `cuda::mr` heavily (67 hits: `resource_ref`, `device_accessible`,
  `forward_property`, `has_property`), `<cuda/memory_resource>`,
  `<cuda/stream_ref>`, and `cuda::std::{span,enable_if_t,void_t,...}`.
  **Verified** the vendored `agent_space/libhipcxx` (branch `amd-develop`,
  commit `fa4ccc6`, the libhipcxx NOTES-validated commit) provides ALL of these
  (`include/cuda/memory_resource`, `include/cuda/stream_ref`,
  `include/cuda/__memory_resource/{resource_ref,properties,get_property}.h`,
  `include/cuda/std/*`).

The hard part is rmm's **rapids-cmake/CPM machinery that FETCHES NVIDIA CCCL**
from GitHub at configure time (`cpp/CMakeLists.txt` bootstraps
`rapidsai/rapids-cmake` via `cmake/RAPIDS.cmake`, then `rapids_cpm_init()` +
`get_cccl`/`get_nvtx`, and gates everything on `CUDAToolkit REQUIRED` +
`enable_language(CUDA)`). Fighting that entire bootstrap to redirect CCCL
inside CPM is brittle and network-coupled.

**Chosen redirect strategy: bypass the rapids-cmake bootstrap with a small
standalone HIP CMakeLists.** Because rocThrust/hipCUB/rocPRIM already live on
hipcc's default include path (`/opt/rocm/include`) and libhipcxx is vendored
(`-I.../libhipcxx/include`), **CCCL is satisfied by include paths alone -- no
CPM fetch is needed at all.** rmm's library is 6 host-C++ files; its real
artifact is the headers. A dedicated `cpp/CMakeLists.hip.cmake` (driven by a
top-level `-DUSE_HIP=ON`) builds `librmm.so` from the 6 `.cpp` under hipcc,
installs the `rmm::rmm` INTERFACE/!shared target + headers + an
`rmm-config.cmake`, and force-includes the compat header. The stock
`cpp/CMakeLists.txt` (NVIDIA path) is left byte-for-byte unchanged. This keeps
the diff minimal, the CUDA build identical, and removes the network/rapids-cmake
entanglement that is the documented main obstacle. (The generalizable recipe --
"for RAPIDS libs whose CCCL deps are header-only on ROCm, satisfy CCCL via the
ROCm include path + vendored libhipcxx and skip the CPM fetch" -- goes to
PORTING_GUIDE.md.)

## Port mechanics (minimal footprint, CUDA path unchanged)

1. One compat header `cpp/include/rmm/detail/hip/cuda_to_hip.h`: on
   `USE_HIP||__HIP_PLATFORM_AMD__` include `<cstring>/<cstdlib>` then
   `<hip/hip_runtime.h>` and `#define` every `cuda*` symbol from the surface
   above to its `hip*` equivalent; else no-op. Force-include it on the HIP
   target (`-include .../cuda_to_hip.h`).
2. Forwarding shim headers (HIP-only include dir) named exactly
   `cuda_runtime_api.h` and `cuda_runtime.h` (each `#include`s the compat
   header) so rmm's `#include <cuda_runtime_api.h>` resolves without editing
   every source. `<cuda/...>` and `<thrust/...>` resolve from libhipcxx /
   /opt/rocm directly.
3. USE_HIP-guard the two Thrust namespace deltas in `exec_policy.hpp` and
   `thrust_allocator_adaptor.hpp` (`cuda`->`hip`, `cuda_cub`->`hip_rocprim`).
4. CMake: a `USE_HIP` path that `enable_language(HIP)`, sets
   `CMAKE_HIP_ARCHITECTURES` from the cache var (default `gfx90a` only when
   unset -- never hardcode), marks the 6 `.cpp` `LANGUAGE HIP`, links
   `hip::host`, and exports `rmm::rmm`.

## Fault classes

- Warp size: **N/A** -- rmm has no warp-level primitives (no `__shfl/__ballot/
  warpSize`); it is allocator/container code.
- The real work is the memory-API enum spellings + the CCCL redirect + the two
  Thrust namespace deltas. No texture/atomics/OOB classes apply.

## Build + validate (gfx90a, HIP_VISIBLE_DEVICES=2)

- Build `librmm.so` + the gtest suite under hipcc; the tests exercise every
  resource (cuda, managed, pool, arena, aligned, binning, fixed_size, callback,
  the adaptors) and the containers (device_buffer/uvector/scalar) +
  stream-ordered alloc. Run on real gfx90a; confirm allocation/free works and
  the suite passes. Note any resource deferred.
- Optionally build the Python (scikit-build/Cython) package + smoke.

## Scope

Target the CORE C++ memory-resource framework + containers + their gtests
passing on gfx90a, and the `## Install as a dependency` recipe working. Defer
(documented) anything genuinely CUDA-12-only (fabric IPC, hwdecompress) and the
Python layer if it proves heavy. Set linux-gfx90a `ported` with the scope note.
