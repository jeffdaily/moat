# metaeuk notes

## Build (linux-gfx90a)
```bash
cmake -S projects/metaeuk/src -B projects/metaeuk/build \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build projects/metaeuk/build -j16
```

## Port notes
- metaeuk vendors MMseqs2 (lib/mmseqs) which vendors libmarv (the GPU library)
- Port follows same Strategy A as standalone MMseqs2 port (jeffdaily/MMseqs2@moat-port)
- cuda_to_hip.h maps runtime symbols + SIMD intrinsics (via marv_simd_amd.cuh)
- hip_compat/ redirects <cuda_fp16.h> and <cooperative_groups*> includes
- Kernel configs select conservative sm75 set on AMD (half2 path, 64KB shared)
- cg::reduce calls replaced with portable marv_tile_reduce on HIP
- libmarv built as shared library with HIP_RESOLVE_DEVICE_SYMBOLS=ON
- rocThrust namespace mapped: thrust::cuda -> thrust::hip via using declarations

## Review 2026-06-05

### Summary

Port adds HIP/ROCm support to metaeuk's vendored libmarv GPU alignment library using Strategy A (compat header + LANGUAGE HIP). The port is clean and follows the same pattern as the validated standalone MMseqs2 port.

### Port Correctness

No issues found. The port correctly:
- Force-includes cuda_to_hip.h to map CUDA runtime symbols
- Emulates 14 SIMD intrinsics (__vadd2, __vmaxs2, __viaddmax_s16x2, __vibmax_u16x2, etc.) with correct semantics (wrapping add for __viaddmax, saturating for __vaddus4/__vsubus4)
- Provides shuffle overloads for packed types (short2, int2, float2, int3)
- Replaces cg::reduce with marv_tile_reduce using butterfly reduction over group.shfl_xor

### Fault Classes

No issues found:
- **Warp size**: All tile widths are <= 32 (groupsize in {4,8,16,32}), explicit parameters. Wave-agnostic.
- **Lane masks**: The __shfl_*_sync macros drop the mask and forward to maskless __shfl_* (correct for width-bounded logical-warp shuffles)
- **Rule-of-five**: cub::SwitchDevice has proper RAII (deleted copy ctor/assignment, guarded destructor)
- **Library swaps**: rocThrust correctly mapped via namespace injection
- **Kernel config dispatch**: AMD branch returns sm75 config set (avoids CC==9 collision with Hopper DPX path)

### Minimal Footprint

No issues found. Host C++ is untouched. USE_HIP guards are minimal and only where genuinely needed. CUDA path is byte-identical.

### Build System

No issues found:
- enable_language(HIP) used correctly
- CMAKE_HIP_ARCHITECTURES defaults to gfx90a only when not defined (followers can override)
- HIP_ARCHITECTURES reads from ${CMAKE_HIP_ARCHITECTURES}
- --offload-compress used to control object size
- CUDA build path preserved in else() block

### Commit Hygiene

No issues found:
- Title: "[ROCm] Add HIP/ROCm support for AMD GPUs" (40 chars, starts with [ROCm])
- Body mentions Claude by name
- No Co-Authored-By noreply trailer
- Author is jeff.daily@amd.com (jeffdaily)
- No MOAT jargon in the diff

### Recommendation

**Approve** -- the port is correct and ready for GPU validation.
