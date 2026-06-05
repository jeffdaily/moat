# cuSZ notes

## linux-gfx90a port

### Build

```bash
cd projects/cuSZ/src
mkdir build-hip && cd build-hip
cmake .. -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```

### Test

```bash
HIP_VISIBLE_DEVICES=0 ./hipsz --version
HIP_VISIBLE_DEVICES=0 ./hipsz -z -i <input.f32> -t f32 -m rel -e 1e-4 -l <dims>
HIP_VISIBLE_DEVICES=0 ./hipsz -x -i <input.f32.cusza> --compare <input.f32>
```

### Port summary

The existing HIP support was bitrotted with empty stub files and incomplete cmake configuration. Fixed by:

1. Complete rewrite of cmake/hip.cmake mirroring the cuda.cmake structure
2. Added HIP runtime includes and CUDA-to-HIP translation macros to all .hip files
3. Fixed forward declaration mismatch in spline3.inl
4. Added missing HIP implementations for psz::module functions (GPU_identical, GPU_extrema, GPU_find_max_error, GPU_assess_quality, GPU_calculate_errors)
5. Fixed portable layer headers to conditionally include HIP or CUDA headers
6. Added CUDA-named function aliases in verinfo.hip for CLI compatibility
7. Fixed PROPER_RUNTIME to use ROCM enum value (not HIP which doesn't exist in the enum)
8. Fixed variadic shuffle intrinsic macros to support both 3-arg and 4-arg forms

### Known issues

- The --report time,cr flag causes a crash (std::out_of_range in unordered_map)
- Large data compression may produce oversized output (possible Huffman codec issue)
- Tests not yet fully validated

### Gotchas

- The _portable_runtime enum uses ROCM (value 4), not HIP. The HIP enum value is in _portable_toolkit.
- Most .hip files need both c_cu2hip_0_translation.h (for type/function macros) and c_cu2hip_1_fix_primitives.h (for warp intrinsics)
- The shuffle intrinsic macros need to be variadic to support both 3-arg and 4-arg forms
- Double-precision atomicAdd on HIP uses unsafeAtomicAdd, not the CAS-loop emulation used on older CUDA
