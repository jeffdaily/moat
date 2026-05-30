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
         set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
         set_target_properties(<tgt> PROPERTIES HIP_ARCHITECTURES "gfx90a")
       else()
         enable_language(CUDA)
       endif()

   Marking the existing `.cu` files `LANGUAGE HIP` keeps the diff minimal and the NVIDIA build intact. Configure with `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a` (add `-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` if CMake does not find it).

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

- Texture/surface coherency. Observed on gfx90a (CDNA2): a texture read returned stale or zero data after a `surf2DLayeredwrite` to that same layered `cudaMalloc3DArray` from a PRIOR kernel (a surface read of the location returns the fresh value; the texture read does not). The CUDA programming guide guarantees texture/surface caches are coherent ACROSS kernel launches -- only a texture or surface read of an address written earlier in the SAME kernel is undefined -- so if the write and the read are separate launches the pattern is in-spec on NVIDIA and HIP should match it. Whether ROCm here is a HIP runtime bug (the texture cache not invalidated at the launch boundary), a CDNA-specific hardware detail, or the app relying on intra-kernel coherence is UNDER INVESTIGATION; do not assume it is a hardware limitation. Workaround that is correct regardless: read the consumer via the surface, recreate the texture object after the writes, or add an explicit invalidation. (popsift make_dog reads an all-zero DoG; OPEN.)

- Library swaps. cuBLAS -> hipBLAS, cuFFT -> hipFFT, cuRAND -> hipRAND, cuSPARSE -> hipSPARSE, cuDNN -> MIOpen, Thrust/CUB -> rocThrust/hipCUB. APIs are mostly 1:1; watch handle types and a few signature differences (for example hipBLAS v2 enums).

## Validation policy

- A port is validated only when the project's real test suite builds and passes on a real AMD GPU for the target arch, with no regression in non-GPU tests.
- A CPU-only docker build (image `rocm/dev-ubuntu-24.04` or similar) proves the code compiles and links under ROCm. It cannot observe any Fault-class bug above, since no GPU runs. Add it as a CI tripwire for contributors without AMD hardware, never as the sole gate.
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
- 2026-05-30 -- texture read returned stale/zero after a surf2DLayeredwrite to the same layered array from a PRIOR kernel (gfx90a/CDNA2); CUDA guarantees cross-launch texture/surface cache coherency (only intra-kernel post-write reads are undefined), so this is either a HIP runtime bug, a CDNA detail, or the app relying on intra-kernel coherence -- under investigation, do not assume hardware; workaround: read via the surface, recreate the texture, or invalidate -- popsift (OPEN, blocking)
- 2026-05-30 -- inside a .cu compiled as HIP, host-side memcpy/memset can resolve to HIP's __device__ overloads once <hip/hip_runtime.h> is in scope (host helper fails to compile); include <cstring>/<cstdlib> BEFORE hip_runtime in the compat header so the libc host decls win -- gpuRIR
- 2026-05-30 -- IPO/LTO + the HIP toolchain breaks pybind11 modules: INTERPROCEDURAL_OPTIMIZATION leaves the .so as slim LTO bitcode with no PyInit_* (ImportError, tiny .so), because the HIP link step does not finalize LTO; disable IPO for the HIP build (it is usually optional) -- gpuRIR
- 2026-05-30 -- the 1D-texture linear-filter rejection is the same fault class as 2D (popsift): a cudaFilterModeLinear + cudaReadModeElementType float texture is rejected at create time on AMD regardless of dims; create it cudaFilterModePoint and lerp in software (point-fetch 2 neighbors, -0.5 texel-center convention) -- gpuRIR
- 2026-05-30 -- RIR/audio validation: the direct path is the FIRST significant arrival (round(dist/c*Fs)), NOT the global max; constructive early reflections can exceed it in a reverberant room, so a global-argmax peak check gives a false negative -- gpuRIR
