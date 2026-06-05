# HEonGPU notes

## Build

Library builds successfully on linux-gfx90a:

```bash
cd projects/HEonGPU/src
mkdir build && cd build
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

Output: `src/libheongpu.a` plus dependencies (`libntt-1.0.a`, `libfft-1.0.a`, `librngongpu-1.0.a`)

## Known Issues

### Test/Example Build Issue (CMake flag propagation)

When tests or examples are enabled, pure C++ files receive HIP compile flags (`-x hip --offload-arch=gfx90a`) because they link to the heongpu HIP library. CMake propagates HIP language properties to linked targets. This causes g++ to fail since it doesn't understand HIP flags.

Workarounds attempted:
- Setting `LINKER_LANGUAGE CXX` on test targets
- Using `hip::host` instead of `hip::device`
- Setting `HIP_ARCHITECTURES ""` on test targets

None fully resolved the flag propagation. This needs CMake-level investigation to understand how to prevent INTERFACE property propagation for HIP.

The library itself is unaffected and functions correctly.

## Port Details

### Key adaptations:

1. **cuda_to_hip.h header**: Central compatibility header mapping CUDA runtime symbols to HIP equivalents, including cuRAND->hipRAND mappings and warp size abstraction.

2. **rmm_hip_stub/**: Minimal RMM implementation for HIP since the real RMM does not support HIP. Implements `device_uvector`, `device_buffer`, `pool_memory_resource`, `statistics_resource_adaptor`, `pinned_memory_resource`.

3. **hip_compat/**: Shim headers for thirdparty code that includes `cuda_runtime.h` and `curand_kernel.h` directly.

4. **PTX inline assembly**: Replaced in `bigintegerarith.cuh` and `GPU-NTT/modular_arith.cuh` with portable C++ using `__umul64hi` intrinsic.

5. **Device function linking**: Made `SmallForwardNTT`/`SmallInverseNTT` inline in header to avoid cross-TU device linking issues (HIP doesn't support CUDA's CUDA_SEPARABLE_COMPILATION the same way).

6. **Warp shuffles**: HIP uses `__shfl_down(val, offset)` without mask; CUDA uses `__shfl_down_sync(mask, val, offset)`.

7. **Warp size**: Changed hardcoded `32` to runtime `warpSize` for gfx90a wave64 compatibility.

8. **HostVector**: Added explicit copy/move assignment operators (clang stricter than NVCC about std::vector inheritance).

9. **hipPointerAttribute_t**: HIP uses `.type` member, not `.memoryType`.
