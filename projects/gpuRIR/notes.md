# gpuRIR notes

## Porting 2026-05-30

Ported to ROCm/HIP for gfx90a (MI250X, wave64) on ROCm 7.2.1. Strategy A
(colmap model) + a cuFFT->hipFFT and cuRAND->hipRAND library swap.

### Build classification
Pure CMake building a pybind11 MODULE (`gpuRIR_bind`) from one `.cu`
(`src/gpuRIR_cuda.cu`) + a C++ host file (`src/python_bind.cpp`). Not a
pytorch extension. `pip install .` drives CMake via a CMakeExtension shim.

### CUDA surface
One `.cu`. Kernels use only shared memory + `__syncthreads` reductions --
**no warp intrinsics, no `__constant__`, no `__shfl/__ballot/__any`**, so the
wave64 / popsift warp-packing fault class does NOT apply. cuFFT (R2C/C2R 1D
plans for the FFT convolution), cuRAND host API (uniform noise for the
diffuse tail), a 1D LUT texture (default-on), `cuda_fp16` half2 mixed
precision (off by default), and 3D pitched copies.

### Port changes (all HIP-guarded; CUDA path byte-for-byte unchanged)
- `src/cuda_to_hip.h` (new): on USE_HIP include hip_runtime + hip_fp16 +
  hipfft + hiprand and alias the cuda/cufft/curand symbols (names from
  PyTorch's hipify map). `#else` includes the original CUDA headers verbatim.
- `src/gpuRIR_cuda.cu`: swapped the 6 CUDA `#include`s for the shim; added
  two small `#if defined(USE_HIP)` branches (texture fix) + one in
  `activate_mixed_precision`.
- `CMakeLists.txt`: `option(USE_HIP)`; under HIP -> project LANGUAGES CXX HIP,
  `enable_language(HIP)`, mark the `.cu` `LANGUAGE HIP`, HIP_ARCHITECTURES
  gfx90a, `-DUSE_HIP`, force-include the shim, link `hip::hipfft hip::hiprand`
  instead of cuFFT/cuRAND, and **skip find_package(CUDA)**.
- `setup.py`: forward `$CMAKE_ARGS` so `pip install .` is hands-free.

### Fault classes / gotchas hit
1. **Texture linear filter (popsift class).** The default LUT path uses a 1D
   cudaArray texture with `cudaFilterModeLinear` + `cudaReadModeElementType`
   -- AMD rejects hw linear filtering on element-read float textures. Fix:
   on HIP create the texture `cudaFilterModePoint` and do the linear
   interpolation in software in `image_sample_lut` (point-fetch 2 neighbors +
   lerp, CUDA's unnormalized -0.5 texel-center convention). Verified: LUT vs
   exact-sinc paths agree to 0.43% of peak.
2. **`__fmaf_rz` missing in HIP** (only `__fmaf_rn`), exactly like CudaSift's
   `__fmul_rz`. The only use computes a LUT index coordinate, so round-mode
   is immaterial -> aliased to `__fmaf_rn` in the shim.
3. **Host `memcpy`/`memset` resolve to HIP `__device__` overloads** inside a
   `.cu` once `<hip/hip_runtime.h>` is in scope (the `PadData` host helper
   failed to compile). Fix: include `<cstring>`/`<cstdlib>` BEFORE
   hip_runtime in the shim so the libc host decls win overload resolution.
4. **IPO/LTO breaks the pybind11 module under the HIP toolchain.** With
   `INTERPROCEDURAL_OPTIMIZATION TRUE` the module `.so` came out as slim LTO
   bitcode with no `PyInit_gpuRIR_bind` (ImportError, 5 KB .so). The HIP link
   step does not finalize LTO. Fix: skip IPO when USE_HIP (it is explicitly
   optional here). After: 340 KB .so with `PyInit_gpuRIR_bind` exported.
5. **Mixed precision not ported.** The half2 device kernels are inside
   `#if __CUDA_ARCH__ >= 530` (undefined on HIP) so they compile to empty
   stubs. But host `cuda_arch` = major*100+minor*10 is 900 on gfx90a, which
   would enable the (broken) path. Guarded `activate_mixed_precision` to
   report unsupported on HIP, forcing it off so the stubs are never launched.
   Default config (LUT on, mp off) is unaffected.
6. `cudaMemcpyToArray` is still declared (deprecated) in ROCm 7.2 with the
   CUDA signature, so it aliases directly -- no call-site rewrite needed
   (unlike CudaSift on an older ROCm). `<cufftXt.h>` is included upstream but
   none of its symbols are used; dropped on the HIP branch.

### Build + validation
- Build: `pip install ./projects/gpuRIR/src --no-build-isolation` with
  `CMAKE_ARGS="-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm"`.
  Links libhipfft/libhiprand/librocfft/librocrand/libamdhip64, no CUDA.
- Validated on gfx90a GPU 3 (`HIP_VISIBLE_DEVICES=3`):
  agent_space/gpuRIR_validate.py -- shoebox room 4x5x3, src (1,1,1.5) ->
  rcv (3,4,1.5), dist 3.6056 m, expected direct-path sample
  round(dist/c*Fs)=168. Result: RIR finite, non-zero; first significant
  arrival (>10% of peak) at sample 168 for BOTH LUT and exact-sinc paths,
  silent before the direct path; hipFFT convolution (simulateTrajectory)
  finite + non-zero. Repo `examples/example.py` (2 src x 3 rcv, cardioid
  mic) also runs clean (exit 0). NOTE: the direct path is the first arrival,
  NOT the global max -- constructive early reflections can be larger in a
  reverberant room; a global-argmax check is wrong.
