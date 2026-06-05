# CuRast notes

## Blocking issues (linux-gfx90a)

### 1. Project is Windows-only upstream

The upstream project explicitly does not support Linux. From the README:
> "Main challenge: We're using the windows API for memory mapping and unbuffered IO"

Missing Linux implementations:
- `MappedFile.h`: Contains literal `TODO` placeholder for Linux mmap implementation (line 36)
- `UnbufferedFile`: Entirely Windows-only (uses CreateFileA, FILE_FLAG_NO_BUFFERING, OVERLAPPED I/O)
- Platform helpers use Windows-specific Win32 API extensively

This is not a ROCm/HIP porting issue -- the project cannot even compile on Linux with CUDA.

### 2. nvrtc + nvJitLink LTO workflow has no HIP equivalent

The project uses a sophisticated runtime compilation pipeline:
1. `nvrtcCompileProgram` with `--dlink-time-opt` and `--relocatable-device-code=true`
2. `nvrtcGetLTOIR` to extract LTO IR
3. `nvJitLinkCreate/AddData/Complete` to link LTO IR modules
4. `nvJitLinkGetLinkedCubin` to produce final device code

hiprtc does not have:
- LTO IR intermediate representation (it compiles directly to HSACO/code objects)
- nvJitLink equivalent for linking multiple modules with LTO

Porting options:
a) Ahead-of-time compilation only (remove runtime compilation feature)
b) Redesign for hiprtc single-step compilation (one monolithic source)
c) Use offline compilation with clang-offload-bundler

Option (a) is simplest but loses the hot-reload feature. Option (b) requires significant restructuring.

### 3. CUDA-Vulkan interop

The interop code (`CudaVulkanSharedMemory.h`, `VulkanCudaSharedMemory.h`) already has Linux FD path using `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`. The HIP equivalents exist:
- `cuMemCreate` -> `hipMemCreate` (ROCm 5.2+)
- `cuMemMap` -> `hipMemMap`
- `cuMemExportToShareableHandle` -> `hipMemExportToShareableHandle`
- `cuMemImportFromShareableHandle` -> `hipMemImportFromShareableHandle`

However, this has NOT been validated with AMD GPU + Vulkan. The virtual memory APIs are relatively recent and may have compatibility gaps.

### 4. HIP kernel API mapping

The kernel code uses:
- `cooperative_groups::tiled_partition<32>` -- HIP supports this
- `cg::coalesced_threads().match_any()` -- HIP supports this
- `surf2Dwrite` -- HIP supports surface operations
- `__ffs`, `__ldg`, `warp.shfl`, `warp.ballot` -- all have HIP equivalents

These are tractable once the platform and runtime compilation issues are resolved.

## Recommendation

This project requires:
1. Linux platform support in upstream (or our fork) BEFORE any ROCm port
2. Decision on runtime compilation redesign

The ROCm/HIP port is blocked on these prerequisites. Consider:
- Waiting for upstream Linux support
- Implementing Linux mmap/IO ourselves (adds scope beyond CUDA-to-HIP)
- Starting with option (a) ahead-of-time compilation

## Files requiring changes

### Must implement for Linux (platform, not HIP)
- `src/MappedFile.h` -- add mmap implementation
- `src/unsuck_platform_specific.cpp` -- add UnbufferedFile Linux implementation (O_DIRECT + io_uring or posix_fadvise)

### Must redesign for HIP (runtime compilation)
- `src/CudaModularProgram.h` -- complete rewrite for hiprtc
- `src/CudaModule` struct -- nvrtc->hiprtc mapping

### Standard CUDA-to-HIP porting
- `CMakeLists.txt` -- USE_HIP option, enable_language(HIP)
- `cmake/common.cmake` -- ADD_HIP() function, find hiprtc
- `src/CURuntime.h` -- cuda.h -> hip/hip_runtime.h
- `src/Timer.h` -- CUevent -> hipEvent_t
- `src/CudaVulkanSharedMemory.h` -- cuMem* -> hipMem*
- `src/VulkanCudaSharedMemory.h` -- cuMem* -> hipMem*
- `src/main.cpp` -- cuInit/cuCtxCreate -> hipInit/hipCtxCreate
- All `.cu` files -- include cuda_to_hip.h compat header
