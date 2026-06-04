# espresso notes

ESPResSo molecular dynamics. Scoped CORE-GPU HIP port (P3M electrostatics +
dipolar direct sum + GpuParticleData + pinned-host allocator + cuda init).
waLBerla LB/EK GPU is explicitly OUT OF SCOPE.

- Fork: https://github.com/jeffdaily/espresso  (branch `moat-port`)
- head_sha: 00149db4459acda290e52a38b2e0e88c4f81d8fd
- Based on upstream eafab258df3b (5.0.1 release notes)
- Strategy A (compat header + LANGUAGE HIP), NVIDIA path byte-identical.

## Build recipe (gfx90a, ROCm 7.2.1)

WITH_WALBERLA=OFF collapses the GPU surface to the 7 core .cu files and avoids
the egress-heavy waLBerla GitLab fetch. heffte/Kokkos/Cabana are still
FetchContent'd from GitHub (kokkos.git, ECP-copa/Cabana, icl-utk-edu/heffte) but
are CPU-only here.

```
export ROCM_PATH=/opt/rocm
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DESPRESSO_BUILD_WITH_WALBERLA=OFF -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON
cmake --build build-hip -j16
```

Build deps were all already present on the gfx90a Jenkins host (nothing to
apt/conda-install): CMake 4.0.3, Ninja, OpenMPI 4.1.6, full libboost-all-dev,
FFTW3, Python 3.12 + numpy/scipy/cython 3.2.5, libhwloc-dev, ROCm 7.2.1
(hipfft + thrust headers under /opt/rocm/include).

Multi-arch fat binary verified: both gfx90a and gfx1100 device code objects
emit in src/core/espresso_core.so (followers need no source change):
`llvm-objdump --offloading build-hip/src/core/espresso_core.so | grep gfx`.

## Test data targets (not built by default)

The python tests import shared helpers and the samples read script files that a
custom target copies into the build tree. ctest fails with
`ModuleNotFoundError: unittest_decorators` / `FileNotFoundError ...local_samples`
until you build them:
```
cmake --build build-hip --target python_test_data   # unittest_decorators, tests_common, data/
cmake --build build-hip --target cuda_test          # C++ unit test (EXCLUDE_FROM_ALL)
```
The `sample_*_with_gpu` / `benchmark_*_with_gpu` ctest failures are missing
`local_samples/*.py` (a samples copy target was not built), NOT GPU faults and
NOT regressions; they fail the same way on the CUDA build.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=0)

Core CPU-vs-GPU / GPU-vs-analytic consistency gates -- 11/11 PASS:
- coulomb_cloud_wall, coulomb_cloud_wall_duplicated (P3M GPU vs reference)
- p3m_madelung (P3M GPU vs analytic Madelung constant -- proves hipFFT
  R2C/C2R normalization parity; a factor-of-N error would fail here)
- p3m_electrostatic_pressure
- dawaanr-and-dds-gpu, dipolar_direct_summation (DipolarDirectSum GPU vs CPU
  forces+torques)
- dipolar_interface, coulomb_interface, dipole_field_tracking, gpu_availability
- cuda_test (C++ pinned-host allocator unit test)

Non-GPU baseline (`ctest -LE gpu`) unaffected (p3m_fft, mmm1d CPU tests pass).
LB/EK GPU tests (lb*, ek*) need waLBerla GPU -> out of scope, not regressions.

## Fault classes hit

1. __CUDACC__ vs __HIPCC__ (the big one). clang for HIP defines __HIPCC__/__HIP__
   but NOT __CUDACC__. ESPResSo gated `DEVICE_QUALIFIER` (= __host__ __device__)
   on __CUDACC__ only (src/utils/include/utils/device_qualifier.hpp), so the
   shared bspline/sinc helpers became host-only under HIP and could not be called
   from device code. Same trap in the "do not include CUDA headers" guard in
   cuda/utils.cuh. Fix: accept __HIPCC__ too.
2. Rule-of-five on cuFFT handles (colmap pattern). P3MGpuFftPlan destroyed its two
   cufftHandle plans unconditionally; a default-constructed / never-initialized
   holder destroyed garbage handles -> ROCm faults (CUDA tolerates). Fix: default
   handles to 0, guarded destroy that resets to 0. Arch-unified.
3. -ffp-contract=on pinned in CMAKE_HIP_FLAGS (clang-HIP defaults to fast) so the
   B-spline weight chains match CUDA/CPU FMA formation (CV-CUDA fault class).
4. hip::device interface-compile-options poisoning: linking roc::rocthrust pulls
   hip::device, whose INTERFACE_COMPILE_OPTIONS inject `-x hip` (wrapped in
   $<COMPILE_LANGUAGE:CXX>) into EVERY host .cpp of the consumer -> g++ chokes on
   `--offload-arch`. Fix: do not link roc::rocthrust (header-only, resolved from
   the ROCm include path); link only hip::hipfft (pulls hip::host, safe).
5. clang-HIP does not implicitly include the runtime header (nvcc does), so the
   .cu files needed the shim include explicitly; CudaHostAllocator.cu had no cuda
   include at all (relied on nvcc auto-include).
6. hipHostMalloc is a template(T**, size_t, flags=0) and would not bind the
   2-arg void** cudaMallocHost call (nor the nullptr probe in cuda_test). Provided
   an inline 2-arg forwarder in the shim instead of a macro.

No warp intrinsics / __shfl / hardcoded-32 anywhere in src/core, so no wave-size
fix was needed; the gfx90a/gfx1100 fat binary covers the followers.

## Environment notes

- Codebase moved well past the planner's view: P3M now uses heffte as its FFT
  backend (p3m_heffte.cpp), and Kokkos/Cabana are mandatory FetchContent deps.
  heffte's CUDA backend (Heffte_ENABLE_CUDA) and the OpenMP_CUDA component are
  both gated off on the HIP path (they are not what the GPU P3M uses; our GPU FFT
  is the hipFFT in p3m_gpu_cuda.cu).
- The HIP language path bypasses the nvcc>=12 / CMAKE_CUDA_COMPILER_ID gate
  entirely, so the C++20 CUDA-version checks never run.

## CMake option

`option(USE_HIP ...)` (default OFF). With ESPRESSO_BUILD_WITH_CUDA=ON it enables
language HIP, defaults CMAKE_HIP_ARCHITECTURES to gfx90a only-if-unset, flips the
7 core .cu to LANGUAGE HIP, links hip::hipfft, and pins -ffp-contract=on.
