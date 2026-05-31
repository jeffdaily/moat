# Plan: heavydb (HeavyAI / OmniSciDB GPU SQL database)

## Project
- Name: heavydb
- Upstream: https://github.com/heavyai/heavydb
- Default branch: master
- Clone head at planning: da23f70ea060f308bb9536ec8515a4ef1d07a4bd (2025-06-17)
- Kind: GPU-accelerated columnar SQL analytics database SERVER (not a library). ~111 MB tree, ~459 files in QueryEngine.

## Disposition: SKIP -- cant-port (recommended)

HeavyDB is not a tractable mechanical CUDA->HIP MOAT port. Its GPU execution is built on a
runtime LLVM NVVM/NVPTX query-codegen pipeline that is structurally bound to the NVIDIA LLVM
target and the CUDA driver JIT, plus a CUDA-driver-API-coupled runtime (CudaMgr, memory, sort,
hash-join) and a cooperative-groups GPU hash-table dependency (warpcore). A real AMD backend
is an LLVM-AMDGPU-codegen + HIP-module-runtime re-engineering effort that a well-resourced
vendor (Intel) already attempted for their own architecture (SPIR-V + Level Zero) and ABANDONED
upstream -- every one of those backend PRs was closed unmerged. This is upstream-scale work,
not a Strategy-A compat-header or Strategy-B hipify port, and it cannot be GPU-validated within
the MOAT pipeline. Recorded command:
`python3 utils/triage.py skip heavyai/heavydb --reason cant-port --note "<see below>"`.

## Existing AMD support

None. There is no HIP/ROCm/AMDGPU code path anywhere in the tree (the "hip" grep hits are
substring false positives: ship/chip/relationship). What exists is the residue of an
abandoned Intel GPU effort:

- `L0Mgr/` -- a Level Zero (Intel oneAPI) DEVICE+MEMORY manager wrapper, gated on `ENABLE_L0`
  (default OFF; the default build compiles `L0MgrNoL0.cpp`, a no-op stub). It is built
  (`add_subdirectory(L0Mgr)`, CMakeLists.txt:1104) but is NOT referenced anywhere outside
  `L0Mgr/`: grep of the whole tree for `L0Mgr`/`HAVE_L0`/`l0_mgr` in .cpp/.h outside that dir
  is EMPTY. It is dead/vestigial -- never wired into DataMgr or the executor.
- There is NO SPIR-V (or any non-NVPTX) codegen path: grep for `spirv`/`spir-v`/SPIR-V codegen
  in QueryEngine is empty.
- Upstream history (GitHub API) shows the Intel backend as a series of CLOSED, UNMERGED PRs
  against master: #688 "Standalone L0 mgr component" (head l0-mgr), #683 "Spirv translator
  build script" (spirv-translator-build), #689 "Spirv build test" (spirv-build-test),
  #690 "L0 kernel" (l0-kernel), #691 "Introduce L0 target" (l0-target), #671 "Aggregations
  support for level zero path" (gpu_support), #697 "Add level zero manager to data manager"
  (l0-data-mgr). All `merged=false`. Branches `alexb_intel/*` corroborate the effort.
  PR #669 (Intel oneDAL K-Means UDTF) is the only Intel item still open and is a CPU ML path,
  not a GPU backend.

Decision: this is an ABANDONED non-NVIDIA backend, but it is abandoned at vendor/upstream
scale and only the inert memory-manager stub landed. Finishing it is not a mechanical port; see
the fault classes. A ROCm/HIP port would still add value in principle (no HIP path exists), but
the cost is upstream-engineering-scale and not GPU-validatable here -> skip cant-port.

## Build classification: pure CMake (not a pytorch extension)

- Top-level `CMakeLists.txt:254-256`: `option(ENABLE_CUDA "Enable CUDA support" ON)` then
  `enable_language(CUDA)`; `find_package(CUDA ...)`, `CMAKE_CUDA_ARCHITECTURES` (lines 273-344).
- No `find_package(Torch)` for the GPU path (Torch is an optional CPU ML-inference dep only,
  `HAVE_CUDA_TORCH`, CMakeLists.txt:1083-1087), no `torch.utils.cpp_extension`, no CUDAExtension.
- By the PORTING_GUIDE rule this is Strategy A territory. BUT Strategy A (a `cuda_to_hip.h`
  compat header + `.cu` marked LANGUAGE HIP) does NOT address this project's core, because the
  GPU code is mostly EMITTED AT RUNTIME by an LLVM backend, not compiled from `.cu` at build time.

## Port strategy: N/A (no viable mechanical strategy)

Neither Strategy A nor Strategy B ports the dominant surface:

- Strategy A would retarget only the 8 hand-written `.cu` files (5.6k LOC: cuda_mapd_rt,
  GpuInitGroups, InPlaceSortImpl, ResultSetSortImpl, TopKSort, HashJoinRuntimeGpu,
  AggModeHashTableGpu, ProfileUtils). That is the SMALL part. The LARGE part is the runtime
  query codegen, which a compat header cannot touch.
- Strategy B does not apply (no torch extension machinery).

The real work is a new LLVM-AMDGPU codegen backend + a HIP runtime-compile/module-load path +
porting the CUDA-driver-coupled runtime. That is a re-architecture, not a translation.

## CUDA surface inventory

1. RUNTIME LLVM NVVM/NVPTX QUERY CODEGEN (the heart; QueryEngine/NativeCodegen.cpp, 3579 LOC):
   - `initializeNVPTXBackend` (line 1623): `llvm::TargetRegistry::lookupTarget("nvptx64")`
     then `target->createTargetMachine("nvptx64-nvidia-cuda", ...)` (line 1638).
   - Module is stamped triple `nvptx64-nvidia-cuda` and an NVPTX datalayout (lines 1347-1352);
     kernels are marked via `nvvm.annotations` "kernel" metadata (line 1375-1383).
   - `legalize_nvvm_ir` (line 1026) strips NVPTX-incompatible IR; NVVMReflect pass +
     `nvptx-f32ftz`/`nvvm-reflect-ftz` attrs (lines 1289-1304).
   - libdevice: `check_module_requires_libdevice` keys on the `__nv_*` prefix (lines 333-338);
     `linkModuleWithLibdevice` links `<cuda>/nvvm/libdevice/libdevice.10.bc` (line 1196).
   - `generatePTX` (line 1592): `nvptx_target_machine->addPassesToEmitFile(..., CGFT_AssemblyFile)`
     emits PTX text.
   - LLVM target dependence is at link granularity: CMakeLists.txt:751
     `llvm_map_components_to_libnames(llvm_libs ${LLVM_TARGETS_TO_BUILD} ...)`.

2. PTX->CUBIN JIT + MODULE LOAD/LAUNCH (QueryEngine/NvidiaKernel.cpp, DeviceKernel.cpp):
   - `ptx_to_cubin` (NvidiaKernel.cpp:132): CUDA driver JIT linker
     `cuLinkCreate`/`cuLinkAddFile`(x2 fatbins)/`cuLinkAddData`(PTX)/`cuLinkComplete` (lines 141-188).
   - One of the linked fatbins is `cuda_mapd_rt.fatbin` (the device runtime, below).
   - `cuModuleLoadData`/`cuModuleGetFunction` (line 218); launch via `cuLaunchKernel`
     (DeviceKernel.cpp), `cuOccupancyMaxPotentialBlockSize`, `cuFuncSetAttribute`.
   - `create_device_kernel` factory returns ONLY `NvidiaKernel` under `#ifdef HAVE_CUDA`, else
     nullptr (DeviceKernel.cpp). The "DeviceKernel" virtual interface is CUDA-only in practice;
     its factory even takes a `CUstream`.

3. DEVICE RUNTIME LIBRARY `cuda_mapd_rt.cu` (1427 LOC) + `GpuInitGroups.cu` etc.:
   - Compiled by nvcc with `-fatbin -rdc=true` (QueryEngine/CMakeLists.txt:471-497) into
     `cuda_mapd_rt.fatbin`, then JIT-linked against each query's PTX. This is the set of
     `extern "C" __device__` runtime functions the codegen calls (get_thread_index,
     pos_start_impl, agg_*, hashing, etc.). Warp size is parameterized
     (`thread_warp_idx(warp_sz)`), and warp intrinsics are nearly absent (only a `/32`,`%32`
     bit-array word index at cuda_mapd_rt.cu:1305) -- so the device runtime ITSELF is the most
     portable piece, but it is downstream of the unportable codegen.

4. HAND-WRITTEN CUDA KERNELS using Thrust/CUB (would map to rocThrust/hipCUB):
   - ResultSetSortImpl.cu, InPlaceSortImpl.cu, TopKSort.cu, SortUtils.cuh,
     JoinHashTable/Runtime/HashJoinRuntimeGpu.cu, DataMgr thrust allocators.

5. warpcore (ThirdParty/warpcore) -- header-only NVIDIA cooperative-groups GPU hash table,
   used by AggModeHashTableGpu (the SQL MODE aggregate). Cooperative groups + `__shfl`/FULL_MASK.
   Requires `--extended-lambda` (CMakeLists.txt:260). Would need a HIP port of warpcore or a
   CK/native rewrite.

6. CUDA driver/runtime memory + management (CudaMgr/CudaMgr.cpp, 87 cu*/cuda* sites;
   DataMgr/Allocators/CudaAllocator, ThrustAllocator; GpuMemUtils, QueryMemoryInitializer,
   GpuInterrupt, InPlaceSort, StringDictionaryTranslationMgr, TreeModelPredictionMgr,
   WindowContext). ~68 distinct cu*/CU* driver symbols across ~70 files outside ThirdParty.
   `CUstream`/`CUdeviceptr`/`CUcontext` types are threaded through executor headers
   (Execute.h, QueryEngine.h, ResultSet.h, QueryMemoryInitializer.h).

## Fault classes (why this is upstream-scale, with citations)

FC-1 (blocking) -- LLVM target retarget. The codegen hard-targets `nvptx64-nvidia-cuda`
(NativeCodegen.cpp:1352, 1638). An AMD backend must build a SECOND codegen path to
`amdgcn-amd-amdhsa`: new TargetMachine, AMDGPU datalayout, AMDGPU kernel calling convention +
`amdgpu_kernel`/`!amdgpu.*` metadata in place of `nvvm.annotations` (1375), AMDGPU address-space
handling, removal/replacement of the NVVMReflect/`nvptx-f32ftz` passes (1289-1304), and a swap of
libdevice (`__nv_*` + libdevice.10.bc, 334-1196) for ROCm device-libs (ocml/ockl + the
oclc_* control bitcodes), including the `__ockl_*`/`__ocml_*` math ABI. This is new compiler
engineering inside HeavyDB, not symbol aliasing.

FC-2 (blocking) -- LLVM build itself. `llvm_map_components_to_libnames(... ${LLVM_TARGETS_TO_BUILD})`
(CMakeLists.txt:751) means HeavyDB links whatever targets the host LLVM was built with. The
codegen needs the AMDGPU target registered IN THE SAME LLVM that the rest of HeavyDB links.
On this host the ROCm LLVM (22.0.0git) reports targets "AMDGPU X86" -- NO NVPTX, so HeavyDB's
existing path will not even link/run against it; and a stock LLVM with NVPTX usually lacks the
full ROCm AMDGPU lowering HeavyDB would now need. Reconciling the LLVM build (one LLVM with both,
matching HeavyDB's pinned LLVM version, matching ROCm's device-libs ABI) is a project-level
toolchain problem.

FC-3 (blocking) -- runtime PTX->cubin JIT has no drop-in HIP analogue. `ptx_to_cubin` uses the
CUDA driver JIT linker to link generated PTX against precompiled fatbins
(NvidiaKernel.cpp:141-188). The HIP analogue is different in shape: emit an AMDGPU LLVM module ->
either feed it to hipRTC, or run the AMDGPU backend + lld to a relocatable code object and
link/bundle against a pre-built device-runtime code object, then `hipModuleLoadData`/
`hipModuleGetFunction`/`hipModuleLaunchKernel`. The link-against-a-prebuilt-runtime-fatbin step
(`cuLinkAddFile` of cuda_mapd_rt.fatbin) must be re-expressed in AMDGPU code-object terms (comgr /
lld `--whole-archive` of a runtime code object). New runtime plumbing, not a rename.

FC-4 (large) -- CUDA-driver-API runtime port. ~68 driver symbols across ~70 files; `CUstream`/
`CUdeviceptr`/`CUcontext` are in executor headers, so a compat header is not isolated to `.cu`
TUs -- it touches host C++ broadly. CudaMgr (87 sites) maps to HIP (hipCtx/hipModule/hipStream),
but the abstraction (DeviceKernel/CudaMgrNoCuda) is not generic; `create_device_kernel` is
`#ifdef HAVE_CUDA` -> NvidiaKernel only. Real multi-backend support means a third manager +
factory wiring (exactly the L0 work that was abandoned).

FC-5 (large) -- warpcore cooperative-groups hash table. Per PORTING_GUIDE, HIP CG (ROCm 7.2.1)
lacks cg::reduce/labeled_partition; warpcore is heavier (full CG hash table, FULL_MASK shuffles).
Either port warpcore to HIP or rewrite AggMode against a portable hash table. Plus the standard
wave64 warp-size class (PORTING_GUIDE) once any device code runs on CDNA.

FC-6 (medium) -- Thrust/CUB sort + hash-join kernels (FC-4 files) map to rocThrust/hipCUB
(mostly drop-in per prior ports), but they sit behind the same CudaMgr/driver coupling.

FC-7 (validation) -- HeavyDB is a database SERVER. GPU correctness is observed only by running
SQL through the full executor (ExecuteTest in GPU device-type mode). There is no isolated GPU
kernel unit test that can pass before FC-1..FC-4 are ALL solved -- i.e. no incremental
correctness-first slice exists. Either the whole codegen+runtime path works for a query, or no
GPU query runs at all.

## Why there is no tractable scoped gfx90a slice

A MOAT port wants a correctness-first first slice that GPU-validates. HeavyDB has none:
- The hand-written `.cu` runtime (the only Strategy-A-shaped surface) is NEVER executed on its
  own; it is JIT-linked against runtime-generated PTX. With no AMDGPU codegen (FC-1) and no
  AMD JIT path (FC-3), the device runtime cannot be exercised by any query.
- The GPU code is GENERATED, so there is no static kernel to translate-and-test.
- The smallest validatable unit is "a SQL query runs on GPU", which requires the entire
  codegen+JIT+runtime+memory stack on AMD simultaneously.
This is the structural NVVM/NVPTX-codegen coupling the task asked to assess: it makes the port
upstream-scale, and it is precisely where Intel's own effort died (FC analysis above).

## Dependency footprint (secondary, but reinforces "server, not library")

Heavy: LLVM+Clang (pinned), Thrift, Arrow/Parquet, GDAL/PROJ/TIFF/GEOS/LibKML, Boost (10+
components), RdKafka, LibArchive, TBB, BLOSC, Snappy, CURL, gperftools, Calcite (Java/Maven),
optional oneDAL/MLPACK/Armadillo, optional Torch. Provisioned via scripts/mapd-deps-*.sh and a
conda recipe (scripts/conda/meta.yaml). Obtainable, but this is a multi-hour-from-source server
build, not a library bringup -- compounding the cost of an already-untractable GPU re-architecture.

## Inter-project deps

None set. HeavyDB does not consume any other MOAT project, and being skipped, nothing should be
made to depend on it. No `set-deps` action.

## Test plan (documented for completeness; NOT runnable without the port)

If the port existed, validation would be: build with the AMD GPU backend enabled, start a
HeavyDB instance on the gfx90a host, and run the gtest suite (Tests/, ~200 executables) with the
central ExecuteTest in `ExecutorDeviceType::GPU` mode (gated by `QR::get()->gpusPresent()`,
ExecuteTest.cpp:224; SKIP_NO_GPU at 648), asserting GPU results equal CPU results for the full
query matrix; plus CudaMgrTest -> the AMD-manager equivalent. Non-GPU regression set: the CPU
device-type half of ExecuteTest and the many storage/catalog/parser tests. None of this is
reachable until FC-1..FC-4 are solved, so it cannot serve as a MOAT validation gate now.

## Recommendation

SKIP, reason cant-port. Precise reason: HeavyDB executes GPU queries via a runtime LLVM
NVVM->NVPTX codegen pipeline (NativeCodegen.cpp lookupTarget("nvptx64") / triple
"nvptx64-nvidia-cuda" / nvvm.annotations / libdevice) whose output is JIT-linked to cubin by the
CUDA driver (NvidiaKernel.cpp cuLink*) and launched via the CUDA driver API; there is no static
GPU kernel to translate and no isolated GPU unit test, so no correctness-first slice exists. A
ROCm port requires a new LLVM-AMDGPU codegen backend, an AMD device-libs (ocml/ockl) link path, a
hipRTC/comgr runtime-compile + hipModule path, and a port of the CUDA-driver-coupled runtime
(~70 files) and the warpcore CG hash table -- upstream-engineering scale. The analogous
non-NVIDIA backend (Intel SPIR-V + Level Zero, PRs #671/#683/#688-691/#697) was attempted by
Intel and CLOSED UNMERGED; only an inert L0 memory-manager stub remains in master. Not a
tractable mechanical MOAT port and not GPU-validatable within the pipeline.

## Open questions (for jeff)

- Confirm the skip. If an AMD LLVM-AMDGPU query-codegen backend for HeavyDB is ever desired, it
  is a standalone multi-week+ engineering project (new compiler target path + runtime), better
  tracked outside the per-repo MOAT mechanical-port pipeline.
