# PORTING_GUIDE

Living best practices for porting CUDA projects to ROCm/HIP. The planner reads this before every project. Any agent that learns a generalizable lesson appends it here (see Changelog) in the same run. Project-specific quirks go in `projects/<name>/notes.md` instead.

ROCm is AMD's GPU compute stack; HIP is its CUDA-like C++ runtime and kernel language. Most CUDA C++ has a 1:1 HIP spelling (`cudaMalloc` -> `hipMalloc`), so a port is usually mechanical plus a handful of real semantic differences listed under Fault classes.

## Before porting: assess existing AMD support

A "port" is not always a fresh CUDA-to-HIP conversion. Check first:
- Mature ROCm/HIP support already upstream -> skip it (disposition already-supported); no work to do.
- AMD supported only via OpenCL, Vulkan, or SYCL, with no HIP path -> a ROCm/HIP port of the CUDA code is still valuable.
- An abandoned or incomplete ROCm/HIP port (stale branch, unmerged PR, old fork) -> finish it rather than starting over.
- A ROCm/HIP port exists but is below the best practices here -> improve it (minimal footprint, warp_size abstraction, and so on).
The planner makes this call per project and records it in plan.md. "Already supports AMD via OpenCL/Vulkan" alone does not mean skip.

Performance-critical kernels (attention, GEMM, quantization) are often tuned to NVIDIA-specific features (CUTLASS/CuTe, Hopper sm90 wgmma/MMA, warp specialization). A straight CUDA-to-HIP translation will compile and run but can leave large performance on the table versus an AMD-native implementation (rocWMMA, Composable Kernel, MFMA intrinsics). For these the planner decides between a mechanical port (correctness first) and an AMD-native rewrite of the hot kernels, and says which in plan.md.

## Build classification

Decide which strategy applies before touching code.

- Pure CMake project (not tied to pytorch): a standalone CMake build with `.cu` sources and CUDA libraries. Use Strategy A.
- pytorch extension: builds via setup.py using `torch.utils.cpp_extension` (`CUDAExtension` / `BuildExtension`), or a CMake build that finds Torch and uses its extension machinery. Use Strategy B.

How to tell: look for `find_package(Torch)`, `torch.utils.cpp_extension`, `CUDAExtension`, or a torch dependency in setup.py / pyproject.toml. If present, it is a pytorch extension. Otherwise treat it as a pure CMake (or Makefile) project.

## Strategy A: pure CMake, colmap model (preferred, minimal footprint)

Goal: only `.cu`/`.hip` translation units see the HIP toolchain; host C++ is untouched; the diff stays small.

1. Add one CUDA-to-HIP compat header (e.g. `src/.../cuda_to_hip.h`). On ROCm it aliases the CUDA spellings the project uses to their HIP equivalents and includes the HIP runtime; on NVIDIA it is a no-op include of the CUDA runtime. Everywhere else keep the plain CUDA spelling (`cudaXxx`, `curand*`, `cublas*`). This header is the only file that knows about HIP.

       #pragma once
       #if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
       #include <hip/hip_runtime.h>
       #define cudaMalloc        hipMalloc
       #define cudaFree          hipFree
       #define cudaMemcpy        hipMemcpy
       #define cudaStream_t      hipStream_t
       #define cudaError_t       hipError_t
       #define cudaSuccess       hipSuccess
       // ... only the symbols the project actually uses
       #else
       #include <cuda_runtime.h>
       #endif

   Use hipify's mapping tables as the authoritative cuda->hip name source when adding aliases: `torch/utils/hipify/cuda_to_hip_mappings.py` in a pytorch checkout lists 3000+ symbol mappings.

2. In CMake, gate the language on a HIP option instead of renaming files:

       option(USE_HIP "Build with HIP for AMD GPUs" OFF)
       if(USE_HIP)
         enable_language(HIP)
         # Never hardcode the lead arch as a literal here: a literal "gfx90a"
         # overrides -DCMAKE_HIP_ARCHITECTURES, so every follower (gfx1100,
         # gfx1151) is forced to edit this file to build -- churning the curated
         # commit's head_sha and forcing already-passed platforms to revalidate.
         if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
           set(CMAKE_HIP_ARCHITECTURES "gfx90a")  # default the lead arch only when unset
         endif()
         set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
         set_target_properties(<tgt> PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
       else()
         enable_language(CUDA)
       endif()

   Marking the existing `.cu` files `LANGUAGE HIP` keeps the diff minimal and the NVIDIA build intact. Configure with `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a` (add `-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` if CMake does not find it). Because the target reads `${CMAKE_HIP_ARCHITECTURES}`, the same lead-port commit builds for any AMD target with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` (gfx90a, gfx1100, gfx1151) and no source change, so a follower validation needs no commit. Also pass every target arch you can test at planning time so the lead bringup is right the first time.

3. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep such guards rare. Dispatch sites that accept either backend use `#if defined(USE_CUDA) || defined(USE_HIP)`.

This is how colmap was ported (PR 4420 plus follow-ups): one compat header, `.cu` marked `LANGUAGE HIP`, a few guarded fixes. PyTorch validated that this isolates HIP: on an MI250 build only the HIP translation units receive `-x hip`; host files are untouched.

## Strategy B: pytorch extension

Torch hipifies extension sources at build time. Do not add a compat header and do not hand-rename symbols.

- Building a `CUDAExtension` on a ROCm torch automatically runs `torch.utils.hipify` on the extension's `.cu`/`.cuh` sources and links the HIP runtime (`amdhip64`, `c10_hip`, `torch_hip`). See `torch/utils/cpp_extension.py`.
- Keep sources in CUDA spelling; hipify translates them. Fix only what hipify cannot (warp size, see below) in source, guarded by `USE_ROCM`.
- Build against a ROCm torch. If the tree was hipified once and is stale after edits, re-run the project's hipify step before rebuilding (a known incremental-build gotcha: edits to `.cu` can recompile the stale hipified mirror unless you re-hipify first).
- For projects shipping their own `.cu` plus a setup.py, the change is often just: build against a ROCm torch and fix the fault classes below.

## Fault classes (AMD is strict where CUDA is lenient)

These are the real semantic differences. Most porting bugs are here, not in symbol names.

- Warp size. NVIDIA warp = 32 lanes always. AMD wavefront = 64 on CDNA (gfx90a, gfx94x) and 32 on RDNA (gfx10xx, gfx11xx, including gfx1100 and gfx1151). Never hardcode 32.
  - Host code (kernel launch, host-side shared-mem sizing): query at runtime, `hipGetDeviceProperties(&prop, dev); prop.warpSize`. PyTorch exposes `at::cuda::warp_size()`.
  - Device code: use a per-arch constant. PyTorch's `C10_WARP_SIZE` is 32 on CUDA, and on HIP is 64 for `__GFX9__` else 32 (`torch/headeronly/macros/Macros.h`). Replicate it as:

        #if defined(__HIP_PLATFORM_AMD__)
        #if defined(__GFX9__)
        static constexpr int kWarpSize = 64;   // CDNA: gfx90a, gfx94x
        #else
        static constexpr int kWarpSize = 32;   // RDNA: gfx10xx, gfx11xx
        #endif
        #else
        static constexpr int kWarpSize = 32;   // CUDA
        #endif

    `__GFX9__` is defined only during device compilation; for host-only code use the runtime query.
  - Static shared-memory arrays sized by warp count must use a compile-time upper bound (PyTorch's `C10_WARP_SIZE_UPPER_BOUND`) or size to 64, since the runtime value is not a constant expression.
  - Lane masks: `__shfl*`, `__ballot`, `__activemask`. On a 64-wide wavefront a `uint32` mask is wrong; use 64-bit masks (`unsigned long long`) where the API takes one.

- Rule-of-five on resource handles. CUDA tolerates a default-constructed or double-destroyed texture/stream/event handle; AMD faults. Give RAII wrappers explicit default init (`handle = 0`), move-only semantics, and a guarded destructor. (colmap CuTexObj bug.)

- Out-of-bounds reads. CUDA often tolerates a read one element past an allocation; AMD faults. Kernels that read index +/- 1 or +/- width at edges (stencils, neighbor gathers) must clamp indices. (colmap ComputeDOG bug.)

- Texture pitch alignment. AMD requires 256-byte row pitch for pitched 2D texture binds; widths that work on CUDA can fail. If a kernel only point-samples, a linear (`tex1Dfetch`-style) bind avoids pitch entirely. (colmap BindTexture2D bug.)

- Texture hardware linear filtering. CUDA accepts a `cudaFilterModeLinear` + `cudaReadModeElementType` texture over a float array; HIP/ROCm rejects it at creation (`hipCreateTextureObject` -> "operation not supported") -- AMD hardware does not support linear filtering on element-read float textures. Fix: on HIP create the texture `cudaFilterModePoint` and do manual bilinear interpolation in software (point-sample the 4 neighbors and lerp), matching CUDA's unnormalized -0.5 texel-center convention: a coordinate c samples texel `floor(c-0.5)` and `floor(c-0.5)+1` with weight `(c-0.5)-floor(c-0.5)`. Put the interpolation behind the project's texture-fetch helper so callers are unchanged. (popsift sift_octave linear textures.)

- Layered-image arrays are broken across kernel launches (use a non-layered 3D array instead). HIP runtime bug, confirmed on gfx90a/CDNA2 (ROCm 7.2.1): for a `cudaArrayLayered | cudaArraySurfaceLoadStore` float array written one layer at a time via `surf2DLayeredwrite`, a read in a LATER kernel launch returns ONE single (last-written) layer's data for EVERY layer index -- `tex2DLayered`, `surf2DLayeredread`, AND even host `hipMemcpy3D` all collapse the layer dimension. `hipDeviceSynchronize` between writes and recreating the texture/surface do NOT help. (An earlier "stale texture but surface is fresh" reading was an artifact of a repro that only ever touched ONE layer, so the last-written layer happened to equal the layer read; reading surf2DLayeredread does NOT actually fix it.) A NON-layered 3D array -- `hipMalloc3DArray` WITHOUT `cudaArrayLayered`, accessed with `surf3Dwrite`/`surf3Dread`/`tex3D` and the level as a real z coordinate -- IS fully per-slice coherent across launches (so is a tall 2D array, W x H*L, slice k at y+k*H). FIX: on HIP drop `cudaArrayLayered` and make the array 3D; map every `surf2DLayeredwrite(v,s,x,y,layer)` -> `surf3Dwrite(v,s,x,y,layer)` (1:1; do it once in the compat header's write wrapper so call sites are untouched) and route every layered read through the project's fetch helper to `surf3Dread`/`tex3D`. With a 3D array `tex3D` is also coherent, so textures can be kept. CUDA keeps the real layered array, byte-for-byte. Standalone repros: agent_space/popsift_run/{multilayer_check*,array3d_check,tall2d_check}.cpp. (popsift: the whole Gaussian pyramid + DoG were layered arrays; without this the DoG was all-zero / 0 features.)

- Library swaps. cuBLAS -> hipBLAS, cuFFT -> hipFFT, cuRAND -> hipRAND, cuSPARSE -> hipSPARSE, cuDNN -> MIOpen, Thrust/CUB -> rocThrust/hipCUB. APIs are mostly 1:1; watch handle types and a few signature differences (for example hipBLAS v2 enums).

## Validation policy

- A port is validated only when the project's real test suite builds and passes on a real AMD GPU for the target arch, with no regression in non-GPU tests.
- A follower validation (gfx1100/gfx1151) must not introduce a NEW fork commit for anything non-essential (CI YAML, formatting, comments). Advancing head_sha forces every already-passed platform back to `revalidate` for a zero-GPU-effect change -- pure churn. If a follower needs no code change, leave the curated commit untouched; amend in only a genuinely necessary build/source fix (e.g. configurable arch).
- A CPU-only docker build (image `rocm/dev-ubuntu-24.04:7.2.4-complete`) proves the code compiles and links under ROCm. It cannot observe any Fault-class bug above, since no GPU runs, so it is never a validation gate. Do NOT wire it into the fork's GitHub Actions: a yml change bumps the fork HEAD sha and forces every platform to revalidate (churn), and the run just fails and emails. Disable Actions on the fork instead; run a CPU-only docker build locally if you want a manual compile check.
- gfx90a is CDNA (wave64); gfx1100/gfx1151 are RDNA (wave32). A change that passes on gfx90a can still fail on RDNA via the warp-size class, which is why followers re-validate on their own hardware.

## Per-arch notes

- gfx90a: MI200-class CDNA2, wavefront 64. Lead platform.
- gfx1100: RDNA3 (Radeon), wavefront 32. Watch warp-size assumptions and RDNA occupancy.
- gfx1151: RDNA3.5 APU, wavefront 32. The Windows HIP SDK is less complete than Linux ROCm; some libraries may be unavailable. Best-effort, after Linux is proven.

## Changelog

Append `YYYY-MM-DD -- lesson -- source project` when you learn a generalizable lesson. Seed entries below come from prior ports and are not yet re-derived inside MOAT.

- seed -- warp-size abstraction: runtime `warpSize` on host, per-arch constant in device code, compile-time upper bound for static arrays -- pytorch, FBGEMM
- seed -- colmap model: `enable_language(HIP)` + a single `cuda_to_hip.h`, `.cu` marked `LANGUAGE HIP` -- colmap
- seed -- rule-of-five on texture handles; clamp out-of-bounds neighbor reads; 256-byte texture pitch -- colmap
- 2026-05-30 -- before dismissing the 256B-pitch fault class, confirm the ACTUAL texture resource type: a cudaResourceTypePitch2D bind is subject to it (even if the pitch happens to satisfy it); do not assume the texture is cudaArray-backed -- CudaSift
- 2026-05-30 -- review (code/strategy/analysis) and validate (real GPU run) are separate MOAT stages; the reviewer does not block on a missing GPU run, the validator provides it -- CudaSift
- 2026-05-30 -- perf-critical kernels (attention/GEMM/quant, often CUTLASS/CuTe/Hopper-tuned): a straight HIP translation may underperform an AMD-native (rocWMMA/CK/MFMA) approach; the planner decides port-vs-rewrite -- per jeff
- 2026-05-30 -- HIP rejects cudaFilterModeLinear + element-read float textures; replace the hardware linear fetch with manual bilinear interpolation (point fetch 4 neighbors + lerp, -0.5 texel-center convention) -- popsift
- 2026-05-30 -- kernels that pack two 32-thread rows into one block break on wave64 (one 64-lane wavefront): __ballot/__popc/__any/warp-bitonic-sort must operate per 32-lane half; treat the wavefront as two independent 32-lane groups so each row matches a 32-lane NVIDIA warp -- popsift (confirmed: ballot_group/any_group helpers + width-32 shuffles restore run-to-run determinism and remove the -nan descriptors)
- 2026-05-30 -- HIP runtime bug (confirmed, gfx90a/CDNA2, ROCm 7.2.1): a cudaArrayLayered+SurfaceLoadStore array written layer-by-layer collapses its layer dimension on any later-launch read (tex2DLayered, surf2DLayeredread, AND host hipMemcpy3D all return the last-written layer for every index); sync and recreating the texture/surface do not help. surf2DLayeredread does NOT fix it (the earlier "surface is fresh" claim was a one-layer-repro artifact). A non-layered 3D array (drop cudaArrayLayered; surf3Dwrite/surf3Dread/tex3D, level as z) IS coherent. Fix: 3D array + map surf2DLayeredwrite->surf3Dwrite in the compat header (call sites untouched) and reads->surf3Dread/tex3D via the project's fetch helper. Repros agent_space/popsift_run/{multilayer_check*,array3d_check,tall2d_check}.cpp; bug report findings/popsift-texsurf-coherency -- popsift
- 2026-05-30 -- wave64 leader-election trap: a kernel that packs two 32-thread rows into one 64-lane wavefront (block (32,HEIGHT)) and does per-row leader election with `lane=threadIdx.x&31; if(lane==0) atomicAdd(...)` FIRES THE ATOMIC ON EVERY SET BALLOT BIT, not just one leader, inflating a global counter ~32-64x (popsift extrema_count: real 575 extrema -> count 36691; downstream walked uninitialized index slots -> a few keypoints emitted ~9000x and run-to-run-nondeterministic totals). When all rows feed the SAME counter, just do ONE full-wavefront 64-bit ballot: atomicAdd `__popcll(ballot)` by wavefront-lane-0, broadcast `__shfl(.,0,64)`, exclusive prefix `__popcll(ballot & ((1<<wflane)-1))`, wflane=threadIdx.x+(threadIdx.y&1)*32. Symptom that fingerprints it: stable primary count but wildly non-deterministic secondary count + massive duplicate outputs -- popsift
- 2026-05-30 -- RootSift/normalization NaN on AMD: sqrt(bin/sum) NaNs when the descriptor accumulation's round-toward-+inf intrinsics (mapped to round-to-nearest in the compat header) push a bin slightly negative, or an all-flat window gives sum==0 (sqrt(neg), 0/0). Default popsift norm is RootSift, not L2 -- check which path actually runs. Fix: clamp `sqrt(fmaxf(bin/sum,0))` and guard sum<=0 / non-finite frsqrt to 0 -- popsift
- 2026-05-30 -- inside a .cu compiled as HIP, host-side memcpy/memset can resolve to HIP's __device__ overloads once <hip/hip_runtime.h> is in scope (host helper fails to compile); include <cstring>/<cstdlib> BEFORE hip_runtime in the compat header so the libc host decls win -- gpuRIR
- 2026-05-30 -- IPO/LTO + the HIP toolchain breaks pybind11 modules: INTERPROCEDURAL_OPTIMIZATION leaves the .so as slim LTO bitcode with no PyInit_* (ImportError, tiny .so), because the HIP link step does not finalize LTO; disable IPO for the HIP build (it is usually optional) -- gpuRIR
- 2026-05-30 -- the 1D-texture linear-filter rejection is the same fault class as 2D (popsift): a cudaFilterModeLinear + cudaReadModeElementType float texture is rejected at create time on AMD regardless of dims; create it cudaFilterModePoint and lerp in software (point-fetch 2 neighbors, -0.5 texel-center convention) -- gpuRIR
- 2026-05-30 -- RIR/audio validation: the direct path is the FIRST significant arrival (round(dist/c*Fs)), NOT the global max; constructive early reflections can exceed it in a reverberant room, so a global-argmax peak check gives a false negative -- gpuRIR
- 2026-05-30 -- bake a configurable HIP arch into the LEAD port: set HIP_ARCHITECTURES from ${CMAKE_HIP_ARCHITECTURES} (default the lead arch only when unset), never a literal "gfx90a". A hardcoded arch overrides the cache var, so every follower validation is forced to edit CMake -- churning the curated commit's head_sha and forcing already-passed platforms to revalidate. Design the lead bringup for all target arches up front to avoid this churn -- CudaSift, Gpufit (per jeff)
- 2026-05-30 -- int atomicMin/atomicMax (and unsigned) are SILENTLY DROPPED on coarse-grained memory on gfx90a/CDNA2 (default hipMallocManaged): the RMW no-ops, the value never updates (atomicAdd and atomicCAS are unaffected). Emulate int min/max with an atomicCAS loop on HIP; use the UNSIGNED comparison when the sentinel is (uint32_t)-1. Micro-test: same op PASSES on hipMalloc device memory, FAILS on hipMallocManaged -- cudaKDTree (spatial builder bounds box stayed empty -> degenerate tree -> OOB)
- 2026-05-30 -- hipCUB DeviceRadixSort::SortKeys with a NONZERO begin_bit (e.g. bits [32,64) of a 64-bit {hi:lo} key) does not sort correctly on ROCm (gfx90a): output is not ordered by the selected bits. Sort the FULL key width instead (begin_bit 0); the high bits still dominate. Direct test: full-64-bit PASS, [32,64) BROKEN (9974/10000 misordered) -- cudaKDTree
- 2026-05-30 -- __both__ / __host__ __device__ macros keyed on `#if defined(__CUDA_ARCH__)` break under HIP: __CUDA_ARCH__ is NOT defined during HIP device compilation (HIP uses __HIP_DEVICE_COMPILE__), so __both__ collapses to host-only and you get "reference to __host__ function in __host__ __device__ function". Make __both__ unconditionally __host__ __device__ on HIP, and add one CUKD_DEVICE_CODE-style macro (= __HIP_DEVICE_COMPILE__ on HIP, __CUDA_ARCH__ on CUDA) for the device-intrinsic-vs-host-fallback `#ifdef`s -- cudaKDTree
- 2026-05-30 -- clang (hence HIP) enforces two-phase name lookup where nvcc/MSVC do not: an unqualified call to an inherited member of a DEPENDENT base (e.g. base method from a class template parameterized base) needs `this->` or explicit qualification ("explicit qualification required to use member ... from dependent base class"). Also: a function-template explicit specialization must match the primary template's __host__/__device__ attributes (a __device__-only specialization of a __both__ primary -> "no function template matches") -- cudaKDTree
- 2026-05-30 -- rocThrust is a true drop-in for Thrust: same `thrust::` API and the same `<thrust/...>` header paths under /opt/rocm/include, so CUDA sources using thrust::sort/device_vector/zip_iterator compile unchanged on HIP with no source swap (only the compat-header CUB alias `#define cub hipcub` is needed for cub::) -- cudaKDTree
- 2026-05-30 -- CUKD_CUDA_CALL(X)-style macros that paste `cuda##X` hide the real symbol names from a grep for `cuda[A-Z]`; expand the macro usages (e.g. CUKD_CUDA_CALL(MemsetAsync) -> cudaMemsetAsync) when enumerating the symbol surface for the compat header, or the build will fail on a few missed symbols -- cudaKDTree
- 2026-05-30 -- a project may already ship an upstream HIP path (makefile.hip + a gpu_macro.cuh-style compat header + USE_HIP guards); do NOT auto-skip as already-supported. These HIP makefiles bitrot against the ROCm in use, so the MOAT value is to actually build+GPU-validate them and fix the rot. Most common rot: the HIP build inherits the CUDA `-std=c++14`, but rocThrust/rocPRIM/hipCUB hard-`#error` "rocPRIM requires at least C++17" (rocprim/config.hpp; headers use std::variant/std::is_same_v). CUDA Thrust accepts C++14 so upstream never caught it. Fix: bump the HIP build to -std=c++17 (only the TUs including thrust/cub fail) -- GPUMD
- 2026-05-30 -- chaotic GPU MD (and any Lyapunov-unstable sim) cannot be bit-validated: run-to-run on the SAME GPU diverges because neighbor-list cell binning uses `ind = atomicAdd(&cell_count[cell],1); contents[base+ind]=atom` (neighbor.cu:80) and atomicAdd ordering across threads is non-deterministic on every GPU (CUDA included), so force-summation order, hence ULP-level forces, differ each run and chaos amplifies them. Upstream `diff -q thermo.out thermo1.out` regression tests therefore can't pass across vendors/runs. Validate by PHYSICS instead: total-energy conservation in NVE (rel drift ~1e-6), temperature at target, and PE/spectra agreement with the reference within tolerance (PE ~1e-5, DOS ~5e-2). Each run independently conserving energy (|dEtot| small while KE/PE decorrelate) is the proof the forces are correct -- GPUMD
- 2026-05-30 -- code on the cuSPARSE GENERIC api (cusparseCreateCsr/SpMV/SpSV/CreateDnVec) ports most cleanly through hipSPARSE, NOT the lower-level rocSPARSE: hipSPARSE mirrors cuSPARSE name-for-name (cusparse->hipsparse, CUSPARSE_->HIPSPARSE_, cudaDataType/CUDA_R_*->hipDataType/HIP_R_* from <hip/library_types.h>), so the file ports line-for-line. Decisive for SpSV: hipSPARSE keeps cuSPARSE's PERSISTENT hipsparseSpSVDescr_t (createDescr/analysis/solve), whereas rocSPARSE's generic SpSV is stage-based with NO persistent descriptor, which would force a structural rewrite. Pick hipSPARSE for generic-api cuSPARSE code; reserve rocSPARSE for new AMD-native work -- amgcl
- 2026-05-30 -- hipSPARSE descriptor TYPE ALIASING gotcha: hipsparseMatDescr_t AND hipsparseDnVecDescr_t are BOTH `typedef void*` (hipsparse-types.h, hipsparse-generic-types.h), while in cuSPARSE they are distinct opaque struct pointers. A single deleter/visitor that overloads operator() on both cuSPARSE types fails to compile under hipSPARSE ("class member cannot be redeclared"). Fix: keep one void* overload and give the other descriptor its own deleter; std::shared_ptr<remove_pointer<...>::type> is shared_ptr<void> for both, so the distinct deleter callable at .reset() is what disambiguates, not the type -- amgcl
- 2026-05-30 -- there is no generic-dispatch hipsparseXcsrilu02_bufferSize/_analysis/hipsparseXcsrilu02 (nor the cuSPARSE X-forms); only type-suffixed hipsparse{S,D,C,Z}csrilu02* exist. cuSPARSE code typically hides this behind static `cusparseXcsrilu02_*` D/S overload wrappers -- port those wrappers verbatim to hipsparse* names. BUT the type-agnostic hipsparseXcsrilu02_zeroPivot DOES exist (matches cuSPARSE). Same pattern for csrsv2 in the pre-generic path -- amgcl
- 2026-05-30 -- for header-only template libs with mutually-exclusive backends (e.g. amgcl's SOLVER_BACKEND_*), mirror the CUDA backend header into a sibling HIP header (backend/cuda.hpp -> backend/hip.hpp) rather than #ifdef-ing one file: keeps the NVIDIA path byte-identical (zero regression risk) and the diff purely additive. The Thrust vector-op _impl specializations are identical between the two (rocThrust device_vector == thrust device_vector), so they only coexist safely because the backends are never compiled into the same TU; do not include both backend headers in one unit -- amgcl
- 2026-05-30 -- AMG/sparse-solver GPU validation that actually proves correctness: solve a 7-point 3D Poisson system (SPD, every AMG lib ships a generator) with the new HIP backend AND the builtin CPU backend, same matrix; assert (a) CONVERGES (final rel resid <= solver tol), (b) iteration count + final residual MATCH the CPU backend (same algorithm -> should be near-identical), (c) ||x_hip - x_cpu||inf/||x_cpu||inf ~ machine eps (~1e-15), (d) run-to-run bitwise determinism (achievable here because the deterministic SpMV alg rowsplit/CSR_ALG1 + Thrust reductions give a fixed reduction order). amgcl hip vs builtin matched to 1.4e-15 with 0 run-to-run diff at n=64 (262144 unknowns) -- amgcl
- 2026-05-30 -- hipEvent_t timing calls (hipEventCreate/Record/Synchronize/ElapsedTime) are marked nodiscard in the HIP headers whereas the cudaEvent* equivalents are not, so a CUDA timing helper that ignores the return code warns under HIP; wrap them in the project's existing error-check macro instead of casting to void -- amgcl
- 2026-05-30 -- HIP `__shfl_sync`/`__any_sync`/`__all_sync`/`__ballot_sync` REQUIRE a 64-bit mask (ROCm 7.x static_asserts `sizeof(MaskT)==8`, "The mask must be a 64-bit integer"), regardless of the active wave width -- so the literal `0xffffffff` that every CUDA warp-sync uses fails to COMPILE on HIP even before any wave64-correctness question. Define a compat full-warp mask keyed on USE_HIP (0xffffffffffffffffULL), NOT on warp width; CUDA keeps the 32-bit literal. The error surfaces during hipcc's HOST compile pass of the .cu (clang compiles the TU for host first), pointing at the macro -- AutoDock-GPU
- 2026-05-30 -- wave64 with atomicAdd-combined warp reductions: when per-warp partials are combined block-wide via `atomicAdd` into a shared accumulator (not assembled positionally), port the reduction NATIVELY to 64 lanes (warpmask=63/warpbits=6, add the stride-32 shuffle/exchange step) rather than treating the wave as two 32-lane halves -- the atomicAdd-combine makes warp granularity flexible, the true 64-lane warp is fewer steps, and it matches a `cData.warpmask`/`warpbits`-parameterized design. (Contrast popsift, where rows are packed positionally 2-per-block and you MUST keep 32-lane half-warps.) Pick the axis by how partials recombine, not reflexively -- AutoDock-GPU
- 2026-05-30 -- a project parameterized for warp size (device reads cData.warpmask/warpbits) can still hardcode them on the HOST (performdocking.cpp set them to 31/5) and in a stray device literal (kernel4 `blockDim.x/32`). Grep BOTH sides: device for the `& warpmask`/`>> warpbits` reads AND the host for where they are assigned, plus bare `/32`,`*32`,`>>5`,`<<5`,`&31`. Set them from props.warpSize at runtime so the same source yields 31/5 on CUDA and 63/6 on gfx90a with zero NVIDIA behavior change -- AutoDock-GPU
- 2026-05-30 -- Strategy A on a project whose HOST .cpp is full of `#ifdef USE_CUDA` GPU-driver blocks (alloc/memcpy/launch) that HIP genuinely shares: define USE_CUDA FOR the HIP build (one line) so all those blocks compile unchanged, and gate only the NVIDIA-only includes (cuda.h/curand.h) to `defined(USE_CUDA) && !defined(USE_HIP)`. The compat header retargets the cuda* runtime symbols to hip*. Far smaller footprint than converting 30 guards to `defined(USE_CUDA)||defined(USE_HIP)` -- AutoDock-GPU
- 2026-05-30 -- a hipcc-compiled kernel .o linked into a host executable by g++ needs `-fPIE` (or -fPIC): modern g++ links PIE by default, and a non-PIC device object trips `relocation R_X86_64_32 against .bss can not be used when making a PIE object`. Add -fPIE to the hipcc kernel-compile in the Makefile -- AutoDock-GPU
- 2026-05-30 -- ship-test cross-check is not always available: AutoDock-GPU's pre-existing OpenCL backend (the path that "already supports AMD") FAILS to compile its kernels under the ROCm 7.2 OpenCL runtime (upstream arg-type bug in gradient_inter_z), so it could not serve as a same-machine reference -- validate the HIP port directly against the published physics instead (1stp streptavidin-biotin: best binding ~-8.3 kcal/mol, best-pose reference RMSD <0.5A, all runs in one cluster). A broken OpenCL path also reinforces why the native HIP backend is worth adding -- AutoDock-GPU
- 2026-05-30 -- HIP vector types ship arithmetic operators that CUDA's plain structs lack: a project that hand-rolls e.g. `operator+=(float3&,float3)` (CUDA's float3 has no operators) makes EVERY `+=` ambiguous on HIP, because HIP's float3 is `HIP_vector_type<float,3>` with a member `operator+=` (clang: "use of overloaded operator '+=' is ambiguous"). Fix: `#if !defined(USE_HIP)`-guard only the colliding overload (HIP's built-in has identical componentwise semantics). Check each overload individually -- HIP provides vector-vector `*`/`/` as friends but NOT the float3*scalar forms, so `operator*(float3,float)` etc. do not collide and must stay -- Fast-Poisson-Image-Editing
- 2026-05-30 -- pybind11 + HIP LTO (refines the gpuRIR lesson): pybind11_add_module injects `-flto=auto -fno-fat-lto-objects` into the compile AND link flags directly, NOT via CMake's IPO property, so `set_target_properties(tgt INTERPROCEDURAL_OPTIMIZATION OFF)` does NOT remove it and the module still links without an exported PyInit_* under HIP (ImportError "dynamic module does not define module export function"). The working fix is `pybind11_add_module(tgt NO_EXTRAS ...)`, which drops pybind11's LTO flags entirely -- Fast-Poisson-Image-Editing
