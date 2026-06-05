# GOMC notes

## Build

### HIP/ROCm (AMD GPUs)

```bash
mkdir build_hip && cd build_hip
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```

For other AMD architectures, change `-DCMAKE_HIP_ARCHITECTURES` (e.g., gfx1100 for RDNA3).

Build produces 4 GPU executables:
- GOMC_GPU_NVT (NVT ensemble)
- GOMC_GPU_NPT (NPT ensemble)
- GOMC_GPU_GCMC (Grand Canonical Monte Carlo)
- GOMC_GPU_GEMC (Gibbs Ensemble Monte Carlo)

### CUDA (NVIDIA GPUs)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Test

All 4 GPU executables tested successfully on AMD gfx90a:

1. GPU_GEMC on AR_KR noble gas system (3000 steps, simple VDW system)
2. GPU_GEMC on ISOPEN_NEOPEN (5000 steps, complex molecules with MEMC moves)
3. GPU_GCMC on PEN_HEX (200 steps, Ewald electrostatics)
4. GPU_NVT on AR_KR single box (1000 steps)
5. GPU_NPT on AR_KR single box (1000 steps with volume moves)

Quick validation command:
```bash
cd test/input/Systems/AR_KR/SingleRun
HIP_VISIBLE_DEVICES=0 ./GOMC_GPU_GEMC in.conf
```

## Validation (2026-06-05)

Platform: linux-gfx90a
GPU: AMD Instinct MI250X / MI250 (gfx90a)
ROCm: 7.2.4
Build: Strategy A (compat header + LANGUAGE HIP)

All 4 GPU executables built and validated:
- GPU_NVT: PASS (1000 steps, 0.004s)
- GPU_NPT: PASS (1000 steps, 0.263s)
- GPU_GCMC: PASS (200 steps, 0.196s)
- GPU_GEMC: PASS (3000 steps, 0.017s; 5000 steps, 2.19s)

GPU kernels tested:
- Energy calculations (VDW, electrostatic)
- Force calculations
- Ewald sums for long-range electrostatics
- Molecule displacement, rotation, transfer moves
- Volume moves (NPT)

All simulations completed successfully with expected Monte Carlo acceptance rates.

## Validation (2026-06-05, linux-gfx1100)

Platform: linux-gfx1100
GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3)
ROCm: 7.2.4
Build: Strategy A (compat header + LANGUAGE HIP)

All 4 GPU executables built and validated successfully on gfx1100:

1. GPU_NVT: PASS
   - Test: NVT/neon_T_34_00_K (1M steps)
   - Runtime: 4.48s
   - Acceptance: 68.9% displacement moves
   
2. GPU_NPT: PASS
   - Test: NPT/water_580_00_K (10k steps, SPCE water)
   - Runtime: 4.22s
   - Acceptance: 100% displacement, 53.9% rotation, 55.6% volume moves
   - Ewald summation active
   
3. GPU_GCMC: PASS
   - Test: GCMC/argon (1M steps, 2-box grand canonical)
   - Runtime: 30.8s
   - Acceptance: 52.0% displacement, 50.8% mol-transfer
   
4. GPU_GEMC: PASS
   - Test: NVT_GEMC/pure_fluid/argon_T_115_00_K (1M steps)
   - Runtime: 46.1s
   - Acceptance: 51.7%/61.2% displacement, 4.8%/4.4% mol-transfer, 54.2% volume-transfer

All Monte Carlo moves functioning correctly on RDNA3:
- Displacement, rotation, regrowth, volume transfer, molecule transfer
- Ewald reciprocal-space electrostatics
- VDW and Coulomb energy/force calculations
- hipCUB device reductions

## Porting notes

### Random123 library

The bundled Random123 library needed several defines for HIP device code:
- `R123_CUDA_DEVICE __device__` -- mark array methods as device functions
- `R123_THROW(x) abort()` -- disable exceptions in device code
- `R123_USE_SSE 0` etc. -- disable x86 SSE intrinsics on GPU
- Use `r123::double2` explicitly to avoid collision with HIP's `double2` type

These defines are placed in `TransformParticlesCUDAKernel.{cu,cuh}` before including Random123 headers.

### hipCUB flag pollution

The `hip::hipcub` CMake target propagates `-x hip --offload-arch=<arch>` flags to all compilation units via `hip::device`. This causes g++ (the CXX compiler) to fail on host .cpp files. The workaround is to manually include hipCUB/rocprim directories and link only `amdhip64`, avoiding the imported target's flag pollution.

### HIP separable compilation

The port uses `HIP_SEPARABLE_COMPILATION ON` and `-fgpu-rdc` to allow `__device__` functions defined in one translation unit to be called from another (e.g., `CalcEnGPU` defined in CalculateEnergyCUDAKernel.cu and called from CalculateForceCUDAKernel.cu).
