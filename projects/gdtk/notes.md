# gdtk notes

## Build for HIP/ROCm

```bash
# Build for AMD MI200 series (gfx90a)
cd src/chicken
make HIP=1 HIP_ARCH=gfx90a install

# Build for other AMD GPUs
make HIP=1 HIP_ARCH=gfx1100 install  # RDNA3
```

## Unit tests

```bash
cd src/chicken/test
make HIP=1 test
```

## GPU validation (shock tube)

```bash
export HIP_VISIBLE_DEVICES=0
export DGD=$HOME/gdtkinst
export PATH=$DGD/bin:$PATH
cd examples/chicken/shock-tube
chkn-prep --job=sod --binary
chkn-run --job=sod --binary
chkn-post --job=sod --binary --vtk-xml --tindx=all
```

## Port notes

- Strategy A variant: Makefile + compat header (not CMake)
- cuda_to_hip.h maps cudaMalloc/cudaFree/cudaMemcpy and error handling APIs
- Changed #ifdef CUDA to #if defined(CUDA) || defined(HIP) for GPU code paths
- No warp intrinsics, no textures, no cuBLAS/cuFFT dependencies (simple port)
- The upstream test_chicken.py has a bug (missing binaryData arg) but GPU simulation passes
- nlohmann/json.hpp warnings are from upstream (deprecated literal operator syntax)

## Review 2026-06-05

Reviewed moat-port branch (b611ce03 vs base 253b4592). No issues found.

**Port correctness**: Strategy A variant (Makefile + compat header) appropriate for this Makefile-based CUDA project. The cuda_to_hip.h header correctly maps all 10 CUDA runtime symbols used (cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaError_t, cudaSuccess, cudaGetLastError, cudaGetErrorString, cudaGetDeviceCount).

**Fault classes**:
- No warp intrinsics (__shfl*, __ballot, __activemask) -- no wave64/wave32 hazards
- No textures or pitched 2D binds
- No cuBLAS/cuFFT/cuRAND/cuSPARSE dependencies
- Atomics (atomicMin on long long int, atomicAdd on int) use device memory allocated via cudaMalloc -- safe on gfx90a
- Thread block size (128 via Config::threads_per_GPUblock) is warp-agnostic
- All 9 kernels have proper bounds checking (if (i < cfg.nActiveCells) or if (i < cfg.nFaces))
- No shared memory usage

**Minimal footprint**: CUDA spelling preserved in source; compat header only force-included for HIP builds. Preprocessor guards correctly use `#if defined(CUDA) || defined(HIP)` pattern.

**Build system**: HIP=1 option integrates correctly alongside GPU=1 (CUDA); HIP_ARCH configurable (default gfx90a); test makefile has matching HIP=1 path.

**Commit hygiene**: Title "[ROCm] Add HIP support for AMD GPU builds" (41 chars), no Co-Authored-By noreply trailer, mentions Claude, has Test Plan section with literal commands, uses jeffdaily account.

Verdict: **PASS** -- ready for validation.

## Validation 2026-06-05

Platform: linux-gfx90a (AMD Instinct MI250X, HIP_VISIBLE_DEVICES=3)
Commit: b611ce03f7f301a0b1bd93b7dca361ffa5ca2387

### Build

Built from scratch with HIP=1 HIP_ARCH=gfx90a:

```bash
cd src/chicken
make HIP=1 HIP_ARCH=gfx90a install -j$(nproc)
```

Build succeeded with only expected warnings:
- nlohmann/json.hpp deprecated literal operator warnings (upstream)
- Unused hipFree return value warnings (5 instances in block.cu cleanup)

Binary installed to /var/lib/jenkins/gdtkinst/bin/chkn-run.

### Unit tests

All 4 unit tests passed:

```bash
cd src/chicken/test
make HIP=1 test
```

- gas_test: PASS (GasModel and GasState thermodynamics)
- rsla_test: PASS (linear algebra solver)
- vector3_test: PASS (3D vector operations)
- spline_test: PASS (cubic spline interpolation)

### GPU simulations

1. **shock-tube** (Sod shock tube problem):
   - 400 cells, 100 time steps
   - Completed successfully in 0.339s
   - HIP device detected and used
   - Final time: t=6.069e-04

2. **isentropic-vortex** (convected vortex):
   - 28800 cells, 880 time steps
   - Completed successfully in 2.800s
   - HIP device detected and used
   - Final time: t=5.006e-02
   - 49 flow field snapshots written

### Verdict

**VALIDATED**: All unit tests and GPU simulations passed on gfx90a. The HIP port correctly utilizes AMD GPU acceleration with no runtime errors or numerical issues.

## Validation 2026-06-05 (gfx1100)

Platform: linux-gfx1100 (AMD Radeon RX 7900 XTX, HIP_VISIBLE_DEVICES=3)
Commit: b611ce03f7f301a0b1bd93b7dca361ffa5ca2387

### Build

Built from scratch with HIP=1 HIP_ARCH=gfx1100:

```bash
cd src/chicken
make HIP=1 HIP_ARCH=gfx1100 install -j$(nproc)
```

Build succeeded with only expected warnings:
- nlohmann/json.hpp deprecated literal operator warnings (upstream)
- Unused hipFree return value warnings (5 instances in block.cu cleanup)

Binary installed to /var/lib/jenkins/gdtkinst/bin/chkn-run.

### Unit tests

All 4 unit tests passed:

```bash
cd src/chicken/test
make HIP=1 test
```

- gas_test: PASS (GasModel and GasState thermodynamics)
- rsla_test: PASS (linear algebra solver)
- vector3_test: PASS (3D vector operations)
- spline_test: PASS (cubic spline interpolation)

### GPU simulations

1. **shock-tube** (Sod shock tube problem):
   - 400 cells, 100 time steps
   - Completed successfully in 0.146s
   - HIP device detected and used
   - Final time: t=6.069e-04

2. **isentropic-vortex** (convected vortex):
   - 28800 cells, 880 time steps
   - Completed successfully in 1.040s
   - HIP device detected and used
   - Final time: t=5.005e-02
   - 49 flow field snapshots written

### Verdict

**VALIDATED**: All unit tests and GPU simulations passed on gfx1100. The HIP port correctly utilizes AMD RDNA3 GPU acceleration with no runtime errors or numerical issues.
