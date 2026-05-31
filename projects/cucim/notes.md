# cucim notes

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated RAPIDS).

## Fork / branch
- Fork: https://github.com/jeffdaily/cucim (jeffdaily user account; `--org jeffdaily` is rejected, jeffdaily is a user not an org -- use `gh repo fork rapidsai/cucim --clone=false`).
- Actions disabled on the fork (`gh api -X PUT repos/jeffdaily/cucim/actions/permissions -F enabled=false`).
- Port branch: `moat-port` off origin/main @ abd0ff0; fork default branch stays a clean upstream mirror.

## Environment (gfx90a host, ROCm 7.2.1)
- 4x MI250X GCDs (gfx90a, wave64). GPUs 0 and 1 free (~11 MB used); 2 and 3 in use. Use GPU 0 via HIP_VISIBLE_DEVICES=0.
- conda env `py_3.12` (/opt/conda/envs/py_3.12), python 3.12.13. 128 cores; cap builds -j16.

## Phase 1 setup -- the load-bearing item: ROCm CuPy (RESOLVED)
- A ROCm CuPy is ALREADY installed: `cupy-rocm-7-0` 14.1.0 (NOT a CUDA wheel as the plan feared). Its `runtime.is_hip` is True once it imports.
- The actual blocker was a NumPy ABI mismatch, not a CUDA-vs-ROCm wheel: `cupy-rocm-7-0` 14.1.0 requires `numpy>=2.0,<2.6` but the env shipped numpy 1.26.4 -> `ImportError: numpy.core.multiarray failed to import` (the classic numpy-2-built-extension-vs-numpy-1 break). Fix is a numpy>=2.0 upgrade (done in a dedicated venv to avoid mutating the shared conda env; see below).
- scikit-image in env is 0.22.0; cuCIM run-deps want >=0.23.2,<0.27.0 (numeric reference parity). Aligned in the venv.

## Phase 1 venv (repeatable)
The shared conda env `py_3.12` has numpy 1.26.4 (too old for cupy-rocm-7-0) and other live MOAT projects. To avoid mutating it, Phase 1 uses a dedicated venv that inherits the conda site-packages (so it reuses the installed `cupy-rocm-7-0`) and shadows numpy/scikit-image with newer pins:
```
python3 -m venv --system-site-packages /var/lib/jenkins/moat/agent_space/cucim-venv
source /var/lib/jenkins/moat/agent_space/cucim-venv/bin/activate
pip install "numpy>=2.0,<2.6" "scikit-image>=0.23.2,<0.27.0" click "lazy-loader>=0.4"
python -c "from cupy.cuda import runtime; print(runtime.is_hip)"   # True
```
cucim itself is pure Python (skimage half); do NOT `pip install` it (the rapids_build_backend pulls cupy-cuda13x + nvidia-nvimgcodec-cu13 and a network rapids backend). Just put it on the path:
```
PYTHONPATH=/var/lib/jenkins/moat/projects/cucim/src/python/cucim/src
HIP_VISIBLE_DEVICES=0 python -m pytest <path>/python/cucim/src/cucim/skimage ... -q
```

## Phase 1 code fix -- HIPRTC std:: / fixed-width-int coverage (the one real Python-side HIP gap)
- Two vendored kernel preambles include `<cupy/cuda_workaround.h>` for `std::` type traits (is_floating_point/is_signed/enable_if) + fixed-width int types. That CuPy header guards its ENTIRE body on `#ifdef __CUDACC_RTC__` (NVRTC-only). HIPRTC defines `__HIPCC_RTC__`, NOT `__CUDACC_RTC__` (confirmed by a RawModule macro probe; HIPRTC also defines `__HIP_DEVICE_COMPILE__`), so on ROCm the header is a NO-OP: `std::*` traits and `uint64_t`/`int8_t` are undefined (they live in `__hip_internal::`), and every kernel built from these preambles fails to compile.
- Stock CuPy already fixed this in cupyx.scipy.ndimage._filters_core via an `if runtime.is_hip:` branch that emits `#include <type_traits>` (+ `<cupy/float16.cuh>`) and DROPS the `std::is_floating_point<float16>` specializations. cucim VENDORS the older pre-fix preamble, so it inherited the bug.
- Fix (mirrors CuPy upstream, guarded by `runtime.is_hip`):
  - `python/cucim/src/cucim/skimage/_vendored/_ndimage_filters_core.py`: HIP branch emits `<cupy/math_constants.h>` + `<cupy/float16.cuh>` + `<type_traits>`, no float16 trait specialization.
  - `python/cucim/src/cucim/skimage/measure/_regionprops_gpu_utils.py`: HIP branch emits `<cstdint>` + `<type_traits>`.
  HIPRTC requires the FIRST line of a program to be `#include` (CuPy comment); the branches honor that.
- The 3 runtime-NVRTC `.cu` (histogram_median, blob, _hessian_det_appx) needed NO change -- no warp intrinsics, no std:: gap. Verified median/blob_dog/blob_doh compile+run.
- Commit on moat-port: `[ROCm] cucim.skimage: fix HIPRTC std:: coverage for CuPy kernels` (089ed1a; amended as validation completes).

## Phase 1 validation (gfx90a, GPU 0/1)
- Full `src` suite = 13,039 tests. COLD JIT is the bottleneck: each test parameterization triggers a fresh HIPRTC compile; first pass is very slow (~minutes per dozen files), but CuPy caches compiled kernels to ~/.cupy/kernel_cache so re-runs are fast. Running SERIALLY per single-GPU guidance.
- test_gaussian.py: 65/65 PASS. median+correlate+blob: all pass (the patched preamble + histogram_median/blob/hessian kernels). Broad subset across morphology/transform/measure/segmentation/restoration/registration/color/exposure/thresholding/core.operations: passing, 0 failures.

## Phase 2 -- libcucim C++ core under HIP (Strategy A, host-only; NO device code)
- Core cudart surface is SMALL and host-only (cpp/src only): cudaMalloc/Free/Memcpy/Memcpy2D, cudaError_t/Success/ErrorInvalidValue/GetErrorString, cudaPointerGetAttributes/cudaPointerAttributes, cudaMemoryType{,Unregistered,Host,Device,Managed}, memcpy-kind enums. (cudaMallocHost/Managed/Memset are PLUGIN/test-only, added to the header for the C++ tests.) All 1:1 hip* host spellings.
- Files: ADD `cpp/include/cucim/util/cuda_to_hip.h` (the single compat header, `<cstring>/<cstdlib>` before hip runtime per gpuRIR lesson). Route `cpp/include/cucim/util/cuda.h` and the 3 direct `<cuda_runtime.h>` includes (memory_manager.cpp, batch_data_processor.cpp, cufile_driver.cpp) + 3 test sources through it.
- GDS/cuFile: has NO ROCm equivalent. The gds/ stub dlopen()s libcufile.so at runtime and the driver falls back to POSIX when absent; with `-DCUCIM_SUPPORT_GDS=OFF` the dlopen is skipped. But the stub + cufile_driver.cpp reference cuFile TYPES at compile time, and `<cufile.h>` does not exist on ROCm -> ADD minimal `cpp/include/cucim/3rdparty/hip_compat/cufile.h` (CUfileError_t/Handle_t/Descr_t/DrvProps_t + CU_FILE_* enums + cufileop_status_error()), on the HIP include path only.
- CMake: top-level `option(USE_HIP)`; under USE_HIP `find_package(hip REQUIRED)` instead of CUDAToolkit, default GDS OFF, swap `CUDA::cudart`->`hip::host` (var CUCIM_GPU_RUNTIME) in cpp/, cpp/tests/, and skip `find_package(CUDAToolkit COMPONENTS cuFile)` in gds/. NO enable_language(HIP) (no device code). Lowered top-level cmake_minimum_required 4.0->3.26 (4.0 is an over-strict floor; box has 3.31.6).
- SuperBuild gotcha: the FetchContent deps libcuckoo + boost-header-only apply git patches that are NOT idempotent. If a configure is interrupted mid-way, re-running fails with "patch does not apply" (already patched). Fix: do ONE clean configure on a fresh build dir (rm -rf build-rocm) so each patch applies once to a freshly-cloned source. Boost clones many submodule libs -> first configure is slow (~7+ min, mostly downloads).
- Scoped OUT on ROCm: GDS (cuFile), nvjpeg (cuslide GPU JPEG decoder -> CPU libjpeg path), nvimgcodec/cuslide2 (not built).
