# Fast-Poisson-Image-Editing notes

## Summary

fpie (Fast Poisson Image Editing): parallel Poisson image-editing / seamless-cloning solver
with many interchangeable backends behind one Python API. The CUDA backend
(`fpie/core/cuda/`, pybind11 module `core_cuda`) is a Jacobi stencil/reduction solver. Ported
to ROCm/HIP for gfx90a via Strategy A (compat header + `.cu` marked `LANGUAGE HIP`, CUDA path
untouched, all HIP behind `USE_HIP`). No prior AMD support existed (genuine fresh port).

## Build classification: pure CMake (Strategy A)

NOT a pytorch extension. setup.py (CMakeBuild) drives a plain `cmake && make`; the CUDA backend
is `pybind11_add_module(core_cuda solver.cc equ.cu grid.cu utils.cu)` linked to `cudart`
(fpie/core/cuda/CMakeLists.txt). pybind11 is git-cloned by the top-level CMakeLists at configure
time (gitignored).

## Files changed (port)

- `fpie/core/cuda/cuda_to_hip.h` (NEW): the only HIP-aware file. On USE_HIP includes
  <hip/hip_runtime.h> and `#define`s the 11 cudaXxx runtime symbols to hip; else includes the
  CUDA headers (<cuda.h>, <cuda_runtime.h>, <driver_functions.h>). Vector types need no alias.
- `fpie/core/cuda/utils.h`: include the compat header instead of the raw CUDA headers; guard the
  custom `operator+=(float3&,float3)` with `#if !defined(USE_HIP)` (fault class below).
- `fpie/core/cuda/CMakeLists.txt`: `option(USE_HIP OFF)`; HIP branch does `enable_language(HIP)`,
  defaults CMAKE_HIP_ARCHITECTURES to gfx90a only when unset, marks the .cu `LANGUAGE HIP`,
  `pybind11_add_module(... NO_EXTRAS ...)`, sets HIP_ARCHITECTURES, defines USE_HIP; else branch
  is the original CUDA path verbatim.
- `CMakeLists.txt` (top): gate the cuda subdir on USE_HIP (find_package(CUDA) fails on a ROCm
  host), else keep the original find_package(CUDA) guard.

USE_HIP defaults OFF everywhere, so a stock NVIDIA/CUDA build is unchanged.

## CUDA surface (small, fully enumerated)

Runtime: cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaMemcpy{Host->Device,Device->Host},
cudaDeviceSynchronize, cudaGetDeviceCount, cudaGetDeviceProperties, cudaDeviceProp, cudaError_t.
Headers: <cuda.h>, <cuda_runtime.h>, <driver_functions.h>. Vector types int4/float3 +
make_int4/make_float3 (native in HIP). Kernel model only. NO warp primitives, NO textures/
surfaces, NO cuBLAS/cuFFT/cuRAND/Thrust/CUB, NO atomics.

## Fault classes hit / not hit

- HIP vector-type operator collision (HIT, the only source fix): HIP's `float3`
  (`HIP_vector_type<float,3>`) ships a componentwise member `operator+=`, so the project's free
  `operator+=(float3&,float3)` (utils.h:11) made EVERY `+=` ambiguous (14 errors in equ.cu, 2 in
  grid.cu). CUDA's float3 is a plain struct with no operators, so it still needs the free one.
  Fix: `#if !defined(USE_HIP)` around just that operator; HIP's built-in has identical semantics.
  The project's other free operators (`+`, `-`, scalar `*`, scalar `/`, `int4*int`) do NOT
  collide (HIP provides only `*`/`/` as vector-vector friends, not the float3*scalar overloads),
  so they were left untouched.
- pybind11 LTO + HIP link (HIT, build-only): pybind11_add_module injects `-flto=auto
  -fno-fat-lto-objects` into compile+link flags; the HIP link step does not finalize LTO, so the
  module shipped with no exported `PyInit_core_cuda` -> ImportError ("dynamic module does not
  define module export function"). `INTERPROCEDURAL_OPTIMIZATION OFF` on the target is NOT
  enough (pybind11 sets the flags directly, not via the CMake IPO property). Fix: pass
  `NO_EXTRAS` to pybind11_add_module on the HIP target (drops the LTO flags). After: `PyInit_*`
  exported, import works.
- driver_functions.h (HANDLED): not present in ROCm; included only on the CUDA side of the
  compat header (it provided pitched-ptr helpers this code never uses).
- Out-of-bounds stencil reads (NOT triggered): grid.cu reads `X-m3, X-3, X+3, X+m3` neighbors
  without clamping, but the Python layer guarantees a >=1px inactive (mask==0) border on every
  side (process.py: zero-out edge then crop x.min()-1 .. x.max()+2), and kernels only write where
  mask is set, so all neighbors are in-bounds. equ.cu gathers neighbors via index array `A` and
  guards id==0; slot 0 is allocated. Confirmed by reading the host code AND by the grid solver
  running bit-exact on real image data on gfx90a. No clamp needed.
- Warp size / wave64 (NOT triggered): the error reduction uses NO warp primitive and assumes NO
  lane count. error_sum_equ_kernel (equ.cu:162) reduces via __shared__ + a single-thread serial
  loop; error_grid_kernel (grid.cu:73) reduces per-block by thread (0,0) and globally by block
  (0,0), both serial. Hence run-to-run deterministic on wave64 and no kWarpSize constant needed.

## Build (gfx90a, GPU 1)

    cmake <src> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python3)
    make -j core_cuda

`ldd core_cuda*.so` -> libamdhip64.so.7 (HIP runtime, not cudart). For a follower arch, only
`-DCMAKE_HIP_ARCHITECTURES=<arch>` changes; no source edit.

## Validation (REAL GPU, HIP_VISIBLE_DEVICES=1, gfx90a / MI250X)

Sample blend = the project's canonical test1 image set (tests/data.txt: DIP2018 test1
src/mask/tgt), placement `-h1 -150 -w1 -50`, 5000 Jacobi iterations, gradient=max.

- equ solver:  max|cuda - numpy| = 0.0 (bit-identical output image), deterministic (run1==run2),
  abs residual [0,0,0].
- grid solver: max|cuda - numpy| = 0.0 (bit-identical), deterministic, abs residual [0,0,0]
  (numpy reference residual ~0.06; cuda converged at least as well).
- End-to-end CLI (fpie.cli.main, the `fpie` console-script path) with `-b cuda` for both
  `--method equ` and `--method grid` writes valid 427x770 blended images (~17-20 ms/run).
- Upstream smoke suite `pytest tests/test_smoke.py`: 7 passed, 1 skipped (the skipped one is the
  OpenMP-vs-numpy CPU comparison; openmp not installed into the validation venv). No regression.

Harness: agent_space/validate_fpie.py (compares cuda vs numpy + determinism).

Gotcha: `python3 -m fpie.cli` does NOTHING (exits 0, no output) because cli.py has no
`if __name__=="__main__"` block -- it only defines `main()`. Use the `fpie` console script or
call `fpie.cli.main()` (as the validation harness does). Not a port issue.

## Env notes

- cv2 (opencv-python-headless) pulls numpy 2.x which breaks numba 0.64 (needs numpy<2.3); pin
  `numpy==1.26.2` so numba + cv2 + the core path all import.
- taichi not installed (optional backend); not needed for the cuda port validation.
