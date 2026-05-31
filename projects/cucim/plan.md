# cuCIM port plan (linux-gfx90a)

## Project
- Name: cucim
- Upstream: https://github.com/rapidsai/cucim (RAPIDS cuCIM -- GPU image processing + slide IO)
- Default branch: main
- Head at clone: abd0ff0 ("Add SECURITY.md (#1093)")
- Lead platform: linux-gfx90a (MI250X / CDNA2, ROCm 7.2.1), wavefront 64
- Inter-project MOAT deps: NONE. Confirmed `set-deps cucim` -> `depends_on = []`. No rmm, raft, cudf, cuvs, cuml, kvikio anywhere (the only "rmm" grep hits are a typo in a kernel name `moments_normmalize...` and a stale comment in cuslide2/CMakeLists.txt). cuCIM does NOT build on rmm.

## TL;DR recommendation: SCOPED PORT, lead value is the CuPy/skimage half (already AMD-aware), C++ slide-IO is a separate, NVIDIA-IO-bound concern
cuCIM is two decoupled halves:
1. **`cucim.skimage` + `cucim.core.operations`** (pure Python on **CuPy**; 82 + ~13 GPU test files). Zero coupling to the C++ library (grep for `cucim.clara`/`CuImage` from skimage returns nothing). All GPU work is CuPy `ElementwiseKernel`/`RawModule`/`ReductionKernel` (33 source files) plus 3 runtime-NVRTC `.cu` files. This half ALREADY carries upstream HIP/ROCm awareness (vendored from CuPy's ROCm-aware `cupyx.scipy.ndimage`; `runtime.is_hip` guards present). It runs on CuPy's ROCm/HIP backend with **no C++ build and effectively no source port** -- the value here is to make it install + GPU-validate on a ROCm CuPy and fix any residual HIP gaps.
2. **`cucim.clara` -> libcucim C++** (TIFF/SVS digital-pathology slide IO via pybind11; 7 test files). This half is welded to NVIDIA-only IO accelerators: **nvjpeg** (hard-linked), **nvimgcodec** (dlopen-wrapped, vendored header), **cuFile/GDS** (dlopen stub). It compiles NO device code (no `.cu`, no `enable_language(CUDA)`); it links the CUDA *runtime* (cudart) for host-side memory/stream ops only.

Disposition: this is NOT an "already-supported, skip" project and NOT a classic `.cu`->HIP kernel port either. It is a **scoped port**: (Phase 1, the real deliverable) bring up + GPU-validate the CuPy `skimage` half on ROCm and close any HIP gaps; (Phase 2, host-side) port the libcucim C++ core's CUDA-runtime usage to HIP and make slide IO build/run on ROCm with nvjpeg/nvimgcodec/cuFile scoped OUT (CPU/openslide JPEG decode fallback path), since those three NVIDIA IO libraries have no drop-in ROCm equivalent and dominate the C++ GPU surface while the kernel surface is essentially nil. Do NOT attempt to port nvjpeg/nvimgcodec/cuFile.

## Existing AMD support
- Classification: **improvable / partial-upstream**. The CuPy-backed `skimage` half is built to run on AMD already (it is CuPy code; CuPy supports ROCm incl. gfx90a, and the vendored kernels carry explicit HIP workarounds and `runtime.is_hip` branches, e.g. `_vendored/_ndimage_interpolation.py:508` raises on HIP texture-accel, `_ndimage_interp_kernels.py:29` has a HIP `#include` workaround). So upstream did not ignore AMD; it just never makes it a tested/supported target and the C++ IO half is NVIDIA-only.
- Decision: PROCEED with a scoped port (do not skip). MOAT value = (a) prove + lock the CuPy half on gfx90a as a tested target and fix residual HIP gaps; (b) get the C++ slide-IO core to build and run under HIP with the NVIDIA IO accelerators scoped out. Both add value an OpenCL/Vulkan-only situation would not -- here there is no AMD path that is actually exercised.
- Upstreamability caveat (from notes.md): RAPIDS is NVIDIA-affiliated; an upstream PR may not be welcome. Treat the upstream-PR gate as likely-decline; the port still has standalone MOAT value (a working jeffdaily fork on ROCm).

## Build classification
- **Strategy A (pure CMake) for the C++ half** + a separate Python/scikit-build-style packaging step. Evidence:
  - Top-level `CMakeLists.txt:23` `project(libcucim ... LANGUAGES CXX)` and `:55` `find_package(CUDAToolkit REQUIRED)` -- CXX only, links cudart; **no `enable_language(CUDA)` anywhere** (grep for `enable_language(CUDA`/`LANGUAGE CUDA` finds only commented-out lines).
  - No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`. So it is NOT a pytorch extension; Strategy B does not apply.
  - The C++ build is multi-stage and driven by `./run build_local` (see `run:274-382`): build+install `libcucim` (top-level CMake) first, then each plugin (`cucim.kit.cuslide`, `cucim.kit.cuslide2`, `cucim.kit.cumed`) as a **standalone CMake project** that does `find_package(cucim CONFIG REQUIRED)`. Plugins are NOT add_subdirectory of the top-level build.
  - The Python package (`python/cucim`, `rapids-build-backend` + setuptools per `dependencies.yaml` py_build_cucim) assumes the C++ `.so` files were already built and copied into `python/cucim/src/cucim/clara/` before `pip install .` (explicit note in dependencies.yaml:94-98). The Python build itself compiles no CUDA.
- Caveat: the 3 `.cu` files are NOT compiled by CMake at all. They are loaded as text at runtime by CuPy `cp.RawModule(code=...)` (see `feature/_hessian_det_appx.py:68`, `feature/blob.py:78`, `filters/_median_hist.py` `_get_median_rawkernel`). On ROCm, CuPy compiles them with HIPRTC -- no project build change, no compat header needed for these.

## Port strategy
- **Phase 1 (primary deliverable) -- CuPy/skimage half, no source port:**
  - Install a ROCm CuPy build (`cupy-rocm-6-x` wheel matching ROCm 7.x, or build CuPy from source against /opt/rocm) so `from cupy.cuda import runtime; runtime.is_hip == True`. The CuPy currently in the env is a CUDA wheel and is broken on a NumPy-ABI mismatch (`numpy.core.multiarray failed to import`) -- this is an env-setup issue, not a code blocker.
  - `pip install` the `python/cucim` package with the C++ `clara` `.so` either absent or built (Phase 2). The skimage import path does not need clara.
  - Run the `src` (skimage + core.operations) GPU test suite on gfx90a. Fix only residual HIP gaps the suite surfaces -- these live in `python/cucim/src/cucim/skimage/_vendored/` and in the 3 runtime `.cu` (warp-size class -- but see inventory: the kernels use NO warp intrinsics, only `__syncthreads` + shared mem, so wave64 risk is low). Any fix is Python-side and guarded by `runtime.is_hip`, matching the existing upstream pattern.
- **Phase 2 (host-side) -- libcucim C++ core via Strategy A compat header:**
  - Add one `cuda_to_hip.h` compat header (per PORTING_GUIDE Strategy A step 1) aliasing the host CUDA-runtime spellings the core uses (`cudaMalloc/Free/Memcpy/Memset/MallocHost/FreeHost/MallocManaged/MallocPitch`, `cudaStream_t`/`StreamCreateWithFlags`/`StreamSynchronize`/`StreamDestroy`/`StreamNonBlocking`, `cudaPointerGetAttributes`/`cudaPointerAttributes`/`cudaMemoryType*`, `cudaError_t`/`cudaSuccess`/`cudaGetErrorString`, `cudaMemcpyKind` enums, `cudaGetDeviceCount`). These are ~30 distinct symbols (full list enumerated in CUDA surface inventory) -- all 1:1 host-runtime spellings to `hip*`. Gate `cucim/util/cuda.h` (`#if CUCIM_SUPPORT_CUDA` -> include hip runtime under USE_HIP) and the `<cuda_runtime.h>` includes.
  - Because the core compiles NO `.cu`, there is NO `enable_language(HIP)` / `LANGUAGE HIP` retag step -- the colmap-model step 2 is a no-op here. The work is purely: make the host TUs that call cudart find HIP, and make CMake link `hip::host`/amdhip64 instead of `CUDA::cudart` under a `USE_HIP` option.
  - Scope OUT nvjpeg/nvimgcodec/cuFile on ROCm:
    - cuFile/GDS: configure with `-DCUCIM_SUPPORT_GDS=OFF`. There is already a clean no-GDS path: `gds/src/cufile_stub.cpp:44,81` `#if !CUCIM_SUPPORT_GDS return;` and the stub `dlopen`s `libcufile.so.0` at runtime returning `CU_FILE_DRIVER_NOT_INITIALIZED` if absent. So GDS-off builds and runs (falls back to POSIX file IO). No cuFile port.
    - nvjpeg (cuslide plugin): hard-linked (`cuslide/CMakeLists.txt:217` `if(TARGET CUDA::nvjpeg_static) ... CUDA::nvjpeg`). On ROCm there is no nvjpeg. Decode JPEG TIFF tiles via the existing CPU/libjpeg path (cuslide already has `libjpeg_turbo`/openslide decode paths) -- i.e. compile cuslide with the nvjpeg GPU decoder `#if`-disabled under USE_HIP and route to the CPU decoder. The `nvjpeg_processor` GPU-batch loader is an optimization, not the only decode path.
    - nvimgcodec (cuslide2 plugin): dlopen-wrapped via `nvimgcodec_dynlink/nvimgcodec_wrap.cc` (vendored 110KB `nvimgcodec.h`). cuslide2 is the newer experimental slide reader. Simplest scope: do NOT build cuslide2 on ROCm (it is optional; `run` only builds it if requested) and rely on cuslide (CPU JPEG path) for slide IO. Revisit only if a slide-IO GPU test demands it.
  - `cumed` plugin: medical-image plugin; check whether it has any NVIDIA-IO coupling (it appears CUDA-runtime-only like the core) and bring it up with the same compat header, or scope out if it adds nvjpeg.

## CUDA surface inventory
- **Device kernels (`__global__`/`__device__`): only 3 `.cu`, all runtime-NVRTC via CuPy, none compiled by the build:**
  - `python/cucim/src/cucim/skimage/filters/cuda/histogram_median.cu` -- histogram-based median filter. Block-level algorithm: `__shared__` histograms, `__syncthreads`, `min/max`, integer atomics-free. NO warp intrinsics, NO hardcoded 32. Parameterized at runtime by `_median_hist.py` (`HIST_SIZE`, `HIST_INT_T`, etc.).
  - `python/cucim/src/cucim/skimage/feature/cuda/blob.cu` -- `_prune_blobs`. Reviewed: no `__shfl`/`__ballot`/`warpSize` (the only "32" is in a doc comment).
  - `python/cucim/src/cucim/skimage/feature/cuda/_hessian_det_appx.cu` -- Hessian determinant approximation. Same: no warp intrinsics.
  - Plus the broad CuPy `ElementwiseKernel`/`ReductionKernel`/`RawModule` surface across 33 skimage source files -- all compiled by CuPy's backend (NVRTC on CUDA, HIPRTC on ROCm), no project-level porting.
- **Host CUDA-runtime usage in C++ (libcucim core + plugins), to alias in the compat header** (counts from grep of `cpp/`): cudaMemcpy(42), cudaError(36), cudaSuccess(21), cudaStream(21), cudaMemoryTypeUnregistered(18), cudaMemcpyHostToDevice(17), cudaFree(17), cudaMemcpyDeviceToHost(16), cudaMalloc(12), cudaMemset(10), cudaStreamSynchronize(6), cudaPointerGetAttributes(6), cudaMemcpyDeviceToDevice(6), cudaGetErrorString(6), cudaPointerAttributes(5), cudaMallocHost(5), cudaMallocManaged(4), cudaGetDeviceCount(4), cudaMemoryTypeDevice(3), cudaMemoryType(3), cudaStreamCreateWithFlags(2), cudaStreamNonBlocking(2), cudaMemoryTypeManaged(2), cudaStreamDestroy(1), cudaMemoryTypeHost(1), cudaMallocPitch(1), cudaFreeHost(1), cudaErrorInvalidValue(1). All have 1:1 `hip*` host spellings.
  - Key host files: `cpp/src/memory/memory_manager.cpp`, `cpp/src/cuimage.cpp`, `cpp/src/io/device_type.cpp`, `cpp/src/loader/{batch_data_processor,thread_batch_data_loader}.cpp`, `cpp/src/cache/image_cache_per_process.cpp`, `cpp/src/filesystem/cufile_driver.cpp`, `cpp/include/cucim/util/cuda.h` (CUDA_TRY/CUDA_ERROR macros + NVJPEG_TRY/ERROR).
- **NVIDIA-only libraries (scope out, no ROCm equivalent):**
  - nvjpeg: `nvjpegCreate/Destroy/JpegStateCreate/DecodeBatched*/Handle_t/Image_t/...` in `cuslide/src/cuslide/{jpeg/libnvjpeg.*,loader/nvjpeg_processor.*}`, `tiff/ifd.cpp`. Hard-linked. -> disable under USE_HIP, use CPU JPEG path.
  - nvimgcodec: full `nvimgcodec*` API in `cuslide2/src/cuslide/{nvimgcodec/*,loader/nvimgcodec_processor.*}`, dlopen-wrapped. -> do not build cuslide2 on ROCm.
  - cuFile/GDS: `cuFile*`/`CUfile*` in `cpp/src/filesystem/cufile_driver.cpp`, `gds/`. dlopen stub with `!CUCIM_SUPPORT_GDS` path. -> `-DCUCIM_SUPPORT_GDS=OFF`.
- **Textures/surfaces:** none in the C++ core. In CuPy skimage: a `_vendored/_texture.py` texture path exists but is already explicitly disabled on HIP (`runtime.is_hip` -> RuntimeError, "HIP does not support texture acceleration"); cuCIM does not use it by default ("We do not use this texture-based implementation in cuCIM"). No action.
- **Pinned/managed memory:** `cudaMallocHost`/`cudaMallocManaged` in the core -> `hipHostMalloc`/`hipMallocManaged` (1:1). NOTE the managed-memory atomics caveat below.
- **Streams/events:** `cudaStreamCreateWithFlags(cudaStreamNonBlocking)` + sync/destroy -> hip equivalents (1:1). No `cudaEvent_t` timing in the core surface.
- **cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB:** none in cuCIM C++. (CuPy provides its own ROCm-backed equivalents for the Python layer.)

## Risk list
- **Warp size (wave64) -- LOW.** The 3 compiled-by-CuPy kernels use only `__syncthreads` + block shared memory, no `__shfl`/`__ballot`/`warpSize`/hardcoded 32. The broad CuPy `ElementwiseKernel`/`Reduction` surface is CuPy's responsibility on ROCm. Residual wave64 bugs, if any, would surface as numeric/test failures in the skimage suite and would be fixed Python-side guarded by `runtime.is_hip`. Watch `_vendored` reduction/scan kernels specifically.
- **NumPy/CuPy ABI in the env -- setup, not code.** Current CuPy import fails with `numpy.core.multiarray failed to import` (CUDA wheel vs ROCm box). Porter/validator must install a matching ROCm CuPy. Not a porting blocker.
- **cudaMallocManaged atomics on gfx90a (PORTING_GUIDE, cudaKDTree).** Core uses `cudaMallocManaged` (4 sites). int atomicMin/Max are silently dropped on coarse-grained managed memory on CDNA2. cuCIM's managed allocations are image buffers, not atomic targets, so likely safe -- but verify no kernel does atomicMin/Max into a managed buffer.
- **CXX11 ABI.** Top-level forces `-D_GLIBCXX_USE_CXX11_ABI=0` (CMakeLists.txt:94). A ROCm CuPy / ROCm libs may be built with ABI=1; if the C++ tests link against anything ABI-sensitive, watch for mismatches. The skimage half is pure Python so unaffected.
- **`_GLIBCXX_USE_CXX11_ABI=0` + HIP runtime headers.** hipcc/clang host pass must accept the old ABI flag; normally fine for host-only C++.
- **Rule-of-five on resource handles:** none in scope (no texture/stream RAII wrappers with the colmap double-destroy hazard found; streams are created/destroyed explicitly). Low risk; re-check `cufile_driver.cpp` handle lifecycle once GDS is off (should be inert).
- **Slide-IO functional parity without nvjpeg:** disabling the GPU JPEG decoder changes performance and possibly the exact decode bytes vs CPU libjpeg. The clara tests compare decoded regions; ensure the CPU decode path is the reference (it is the openslide-compatible path) so tests still pass.
- **scikit-image version pin:** env has skimage 0.22.0; cuCIM run-deps want `>=0.23.2,<0.27.0`. Align before running the skimage suite (some tests compare against host skimage reference output).

## File-by-file change list (Phase 2 C++; Phase 1 is install+validate, source-touch only if a test fails)
- ADD `cpp/include/cucim/util/cuda_to_hip.h` (new) -- the single compat header: under `USE_HIP`/`__HIP_PLATFORM_AMD__` include `<hip/hip_runtime.h>` and alias the ~30 host cudart symbols listed above to `hip*`; else include `<cuda_runtime.h>`. Include `<cstring>`/`<cstdlib>` before the hip runtime (PORTING_GUIDE gpuRIR lesson) since core host files use memcpy/memset.
- EDIT `cpp/include/cucim/util/cuda.h` -- route the `#if CUCIM_SUPPORT_CUDA` include through the compat header; keep CUDA_TRY/CUDA_ERROR (they use `cudaGetErrorString`/`cudaSuccess`, both aliased). Guard NVJPEG_TRY/NVJPEG_ERROR so they are not referenced under USE_HIP.
- EDIT top-level `CMakeLists.txt` -- add `option(USE_HIP ...)`; under USE_HIP, `find_package(hip REQUIRED)` and link `hip::host` instead of (or in addition to) the cudart path; do NOT add `enable_language(HIP)` (no device code). Default `CUCIM_SUPPORT_GDS=OFF` and the nvjpeg/nvimgcodec scope-outs under USE_HIP. Keep arch configurable per PORTING_GUIDE (no literal gfx90a) -- though with no device code, HIP_ARCHITECTURES is moot for the core.
- EDIT `cpp/CMakeLists.txt` -- propagate USE_HIP define; swap `CUDA::cudart` link to the HIP runtime under USE_HIP.
- EDIT `cpp/plugins/cucim.kit.cuslide/CMakeLists.txt` + `src/cuslide/{jpeg/libnvjpeg.*,loader/nvjpeg_processor.*,tiff/ifd.cpp}` -- `#if`-disable the nvjpeg GPU decode path under USE_HIP; route to CPU JPEG decoder. Drop the `CUDA::nvjpeg*` link under USE_HIP.
- (cuslide2) -- exclude from the ROCm build (do not invoke `build_local_cuslide2_`); no source change needed.
- `gds/` -- no source change; configure `-DCUCIM_SUPPORT_GDS=OFF` (existing stub path handles it).
- Python: NO source change expected in Phase 1; if the skimage GPU suite surfaces a HIP gap, fix it in `python/cucim/src/cucim/skimage/_vendored/...` guarded by `runtime.is_hip` (mirror the existing upstream pattern). The 3 `.cu` likely need no change (no warp intrinsics).

## Build commands (gfx90a)
Phase 1 (the primary deliverable -- CuPy skimage half):
```
# 1. Install a ROCm CuPy matching ROCm 7.x (wheel or source build against /opt/rocm); verify:
python3 -c "from cupy.cuda import runtime; print('is_hip', runtime.is_hip)"   # expect True
# 2. Align scikit-image to cuCIM's pin and install the python package WITHOUT the clara .so:
pip install "scikit-image>=0.23.2,<0.27.0"
pip install -e python/cucim   # rapids-build-backend/setuptools; compiles no CUDA
```
Phase 2 (C++ libcucim core under HIP, IO scoped out):
```
cmake -S projects/cucim/src -B projects/cucim/src/build-rocm -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCUCIM_SUPPORT_GDS=OFF \
  -DCUCIM_SUPPORT_NVTX=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/cucim/src/build-rocm --target cucim
cmake --build projects/cucim/src/build-rocm --target install
# then build cuslide (nvjpeg disabled under USE_HIP); skip cuslide2
```
(A CPU-only ROCm docker compile-check is allowed as a smoketest only, never the GPU gate.)

## Test plan
- **Primary GPU validation (Phase 1):** the skimage + core.operations suite on gfx90a. Exact command (from the upstream `run`/CI, adapted to one GPU -- run SERIALLY, not -n8, per PORTING_GUIDE MPPI lesson about single-GPU contention):
  ```
  cd python/cucim && pytest -v src       # 82 skimage test files + ~13 core.operations test files, all CuPy/GPU
  ```
  These are the real GPU tests for the compute half; many assert numeric agreement vs host scikit-image, so they are strong correctness gates. Subset while iterating: `pytest -v src/cucim/skimage/filters` (covers histogram_median.cu), `.../feature` (blob.cu, _hessian_det_appx.cu).
- **C++ slide-IO tests (Phase 2):** `cd python/cucim && pytest -v tests/unit` (the `clara` binding tests: `test_tiff_read_region.py`, `test_load_image*.py`, `test_image_cache.py`, `test_batch_decoding.py`). These need the built `clara` `.so`. With nvjpeg off they exercise the CPU decode path; ensure no regression vs the openslide reference. The C++ gtest/Catch2 tests under `cpp/tests` and `cpp/plugins/*/tests` (`test_read_region`, `test_cufile`, `test_metadata`) can also run; `test_cufile` is expected to no-op/skip with GDS off.
- **Non-GPU regression set (must not regress):** the pure-Python/host portions of `tests/unit` (version/init), and any skimage test that runs on a CPU fallback. The whole `src` suite is GPU; there is no large CPU-only unit set to protect beyond import/version.
- **Determinism:** for the median/blob/hessian kernels, run the relevant skimage tests twice and confirm bit-stable output (they are block-deterministic; no atomic-ordered reductions in those 3 kernels).

## Open questions
1. Is a ROCm CuPy wheel available for ROCm 7.2.1 here, or must CuPy be built from source against /opt/rocm? (Determines Phase-1 setup effort; CuPy upstream officially supports ROCm incl. gfx90a, so source build is the fallback.) This is the single most load-bearing setup item.
2. Scope confirmation: is bringing up the CuPy `skimage` half alone (Phase 1) an acceptable MOAT "port" for cuCIM, with the C++ slide-IO core as Phase 2 best-effort? The kernel surface is entirely in the CuPy half; the C++ half is host-runtime + NVIDIA-IO glue with no device kernels.
3. Does the `cumed` plugin pull any NVIDIA-only IO (it looked CUDA-runtime-only)? If clean, include it; if it adds nvjpeg/nvimgcodec, scope it out like cuslide2.
4. Upstream-PR posture: RAPIDS is NVIDIA-affiliated and notes.md flags upstreaming as unlikely. Confirm whether to skip the upstream-PR gate and treat the jeffdaily fork as the deliverable.

## Delta plan: gfx1100 / gfx1151 (followers, on demand)
- The CuPy half: CuPy on RDNA (wave32) -- re-run the `src` suite; any wave-size divergence is CuPy's, surfaced as numeric test failures.
- The C++ core compiles no device code, so wave size is irrelevant to it; a follower needs only `-DCMAKE_HIP_ARCHITECTURES=<arch>` if/when device code is ever added. No re-plan expected.
