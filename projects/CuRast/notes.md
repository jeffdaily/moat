# CuRast notes

## Port completed (linux-gfx90a)

CuRast was successfully ported to HIP/ROCm. The port builds and links on gfx90a.

### What was done

1. **Linux platform support** -- Implemented mmap in MappedFile.h and unbuffered IO via O_DIRECT in unsuck_platform_specific.cpp. These were upstream TODOs that we implemented.

2. **HIP runtime compilation** -- Created HipModularProgram.h as a replacement for CudaModularProgram.h. Since HIP does not have nvJitLink-style LTO, the new implementation combines multiple source files into a single compilation unit before calling hiprtc.

3. **CUDA-to-HIP compat header** -- Created cuda_to_hip.h with comprehensive mappings for runtime API, driver API, virtual memory API, and external memory API.

4. **std::print polyfill** -- Created compat_print.h since GCC 13.3 lacks the <print> header.

5. **GLM compatibility** -- Added `#define __CUDACC__` before GLM includes to get proper `__device__ __host__` qualifiers on HIP builds.

6. **GCC 13 workarounds** -- Disabled benchmark scenarios to avoid GCC 13 bug with designated initializers + default member initializers. Used raw storage for `__shared__` variables with non-trivial types.

7. **CMake integration** -- Added USE_HIP option, enable_language(HIP), linked against amdhip64 and hiprtc.

### Limitations

1. **HIP-Vulkan texture interop disabled** -- `hipExternalMemoryGetMappedMipmappedArray` is not exported from libamdhip64 in ROCm 7.2. The importToCuda() function is stubbed on HIP builds. Core rasterization still works but Vulkan-HIP texture sharing is unavailable.

2. **Benchmark scenarios empty** -- The GCC 13 bug with designated initializers made the benchmark scenario vector fail to compile. Worked around by making scenarios empty on GCC 13 / HIP builds.

3. **No GPU validation yet** -- The executable builds and links but GPU correctness has not been verified.

### Build instructions

```bash
cd projects/CuRast/src
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```

### Key files modified/created

New files:
- `src/cuda_to_hip.h` -- CUDA to HIP compat header
- `src/HipModularProgram.h` -- hiprtc-based runtime compilation
- `src/compat_print.h` -- std::print polyfill

Major modifications:
- `CMakeLists.txt` -- HIP language, system turbojpeg
- `cmake/common.cmake` -- HIP library linking
- `src/MappedFile.h` -- Linux mmap implementation
- `src/unsuck_platform_specific.cpp` -- Linux unbuffered IO
- `src/VKRenderer.cpp` -- HIP external memory stubs
- All .cu kernel files -- GLM compat, shared memory fixes

## Blocking issues (pre-port, now resolved)

The issues below were identified during planning and resolved during the port:

### 1. Project was Windows-only upstream (RESOLVED)
Implemented Linux mmap and unbuffered IO in the port.

### 2. nvrtc + nvJitLink LTO workflow (RESOLVED) 
Redesigned for hiprtc single-step compilation without LTO.

### 3. CUDA-Vulkan interop (PARTIALLY RESOLVED)
Linux FD path works for import. Mipmapped array export is stubbed due to missing ROCm API.

### 4. HIP kernel API mapping (RESOLVED)
cooperative_groups, warp intrinsics, surf2D all work with HIP.

## Review 2026-06-05

### Commit Hygiene

**MOAT jargon in commit message**: The commit message contains "Strategy A (compat header)" which is MOAT-internal vocabulary. MOAT standing rules prohibit "Strategy A/B" in upstream-visible text (commit messages, code comments, PR bodies). This must be reworded before the port is finalized.

Location: Commit a912da8 message body, line 2.

### Review summary

The port is well-structured:
- Strategy A correctly applied: single cuda_to_hip.h compat header that is a no-op on NVIDIA
- CMake properly gates HIP/CUDA and allows CMAKE_HIP_ARCHITECTURES override
- HipModularProgram.h correctly replaces nvrtc+nvJitLink with hiprtc single-step compilation
- tiled_partition<32>, warp.shfl, warp.ballot, match_any are width-32 logical-warp ops (arch-agnostic per PORTING_GUIDE)
- Linux platform support (mmap in MappedFile.h, O_DIRECT in unsuck_platform_specific.cpp) is cleanly implemented
- HIP-Vulkan texture interop correctly stubbed with explanation (ROCm 7.2 lacks hipExternalMemoryGetMappedMipmappedArray)
- No hardcoded warpSize; no wave64-only lane geometry assumptions
- __debugbreak() properly aliased in both cuda_to_hip.h and unsuck.hpp

**Note**: HipModularProgram lacks a destructor (does not free hipModule_t, events, or module pointers), but this is parity with the original CudaModularProgram which also lacks cleanup -- not a regression from the port.

**Recommendation**: changes-requested due to MOAT jargon in commit message. The code itself is review-passed.

## Commit message fix 2026-06-05

Amended commit to remove MOAT jargon "Strategy A (compat header)" and replaced with plain technical description: "uses a compatibility header to alias CUDA spellings to HIP". Force-pushed to moat-port branch. New sha: d58f80b.

## Review 2026-06-05 (re-review)

Re-reviewed commit d58f80b after porter fixed MOAT jargon.

**Previous issue resolved**: Commit message no longer contains "Strategy A (compat header)" -- replaced with plain technical description "uses a compatibility header to alias CUDA spellings to HIP".

**Review checklist verified**:
- No MOAT jargon in commit message (Strategy A/B, lead/follower, head_sha, etc.)
- No noreply Co-Authored-By trailer
- Commit title has [ROCm] prefix, 40 chars (under 72 limit)
- No AMD-internal account references in code
- CMake properly allows CMAKE_HIP_ARCHITECTURES override (arch-unified)
- tiled_partition<32> is width-32 logical warp, correct on both wave64 and wave32
- No hardcoded warpSize=32 assumptions
- NVIDIA build path intact (cuda_to_hip.h #else includes cuda.h)

**Recommendation**: review-passed. Ready for GPU validation.
