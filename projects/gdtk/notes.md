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

## Validation 2026-06-08 (windows-gfx1201)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gcnArchName=gfx1201, HIP_VISIBLE_DEVICES=0)
Commit: b611ce03f7f301a0b1bd93b7dca361ffa5ca2387

### Toolchain notes

The upstream Makefile is Linux-oriented (uses `make`, POSIX sed/cp). On Windows, the build was
performed manually via the ROCm venv hipcc.exe. gmake from Strawberry Perl was used for unit tests.
zlib was sourced from conan2 (`C:/Users/Shark44/.conan2/p/b/zlib94c8e5ebac5da/p/{include,lib}/zlib.lib`).
TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll) were copied into the bin/ install directory to override the Adrenalin
System32 DLL (which would hang on first GPU kernel dispatch without them).

### Build

```
ROCM_VENV_SCRIPTS="B:/develop/TheRock/external-builds/pytorch/.venv/Scripts"
ROCM_DEVEL="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
ROCM_CORE="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
export PATH="${ROCM_VENV_SCRIPTS}:${ROCM_DEVEL}/bin:${ROCM_DEVEL}/lib/llvm/bin:${ROCM_CORE}/bin:${PATH}"
ZLIB_INC="C:/Users/Shark44/.conan2/p/b/zlib94c8e5ebac5da/p/include"
ZLIB_LIB="C:/Users/Shark44/.conan2/p/b/zlib94c8e5ebac5da/p/lib"
cd src/chicken
sed -e 's/PUT_REVISION_STRING_HERE/b611ce0/' main.cu > main_with_rev_string.cu
hipcc.exe -DHIP -std=c++17 -O2 --offload-arch=gfx1201 -o chkn-run.exe -DIDEAL_AIR \
    -include cuda_to_hip.h -I"${ZLIB_INC}" main_with_rev_string.cu -L"${ZLIB_LIB}" -lzlib -fopenmp
```

Build succeeded with expected warnings (nlohmann/json deprecated literal operator,
hipFree nodiscard, sscanf MSVC deprecation on the host-side compilation).

### Unit tests

```
cd src/chicken/test
/c/Strawberry/c/bin/gmake.exe HIP=1 test
```

All 4 unit tests passed:
- gas_test: PASS (GasModel and GasState thermodynamics)
- rsla_test: PASS (linear algebra solver)
- vector3_test: PASS (3D vector operations)
- spline_test: PASS (cubic spline interpolation)

### GPU simulations

1. **shock-tube** (Sod shock tube problem):
   - 400 cells, 100 time steps
   - Completed successfully in 92.894s (slow due to Windows GPU dispatch overhead on small problem)
   - HIP device detected and used (1 device found)
   - Final time: t=6.069e-04 -- MATCHES gfx90a and gfx1100 exactly
   - Result: PASS

2. **isentropic-vortex** (convected vortex):
   - 28800 cells, up to 150000 time steps (max_time=50ms)
   - Uses sbp_asf flux calculator and exchange BCs on all boundaries
   - FAILED: immediately at step 1, ALL ~28,800 cells reported "Bad cell" via GPU printf
   - Error: "Stage 2, could not copy bad_cell_count from gpu to host cpu." (hipMemcpy failed after the bad-cell explosion)
   - Done in 124s (the runtime_error was caught by main(), simulation terminated)
   - The Linux runs (gfx90a and gfx1100) both completed with 880 steps to t=5.005e-02 with no bad cells.

### Diagnosis

The shock-tube passes with identical final time to Linux (ausmdv flux, wall_with_slip BCs).
The isentropic-vortex catastrophically fails at step 1 on gfx1201 only (sbp_asf flux + self-exchange
periodic BCs, 1 block wrapping onto itself). All 28,800 cells become bad simultaneously at the first
time step, indicating a numerical blow-up.

The sbp_asf kernel uses a 4-point stencil (fsL1, fsL0, fsR0, fsR1) and the exchange BC copies
ghost cell FlowState values across from the opposite boundary in the same block (read-write across
threads). This is the only code path not exercised by the passing shock-tube.

Possible causes:
1. gfx1201 FP-accumulation divergence in the sbp_asf kernel causing NaN/inf from step 1
2. Data race in the exchange BC when all boundary faces are set by `apply_convective_boundary_condition`
   in parallel (threads reading and writing the same cells array) -- but this would fail on all platforms
3. Wave32 vs wave64 difference: gfx1201 is wave32 (all AMD GFX are wave32 except gfx90a which is wave64),
   same as gfx1100, so this is not the distinguishing factor between gfx1100 (pass) and gfx1201 (fail).

The blow-up being gfx1201-specific (not gfx1100) despite both being wave32 RDNA points to an RDNA4
arithmetic divergence in the sbp_asf formulation.

### Verdict

**VALIDATION-FAILED**: The primary shock-tube validation PASSED on gfx1201 with correct numerics
(t=6.069e-04 matching Linux). However, the isentropic-vortex (sbp_asf + exchange BCs) catastrophically
fails at step 1 with all cells bad, a gfx1201-specific regression not seen on gfx90a or gfx1100.
Bouncing to porter for investigation of the sbp_asf/exchange-BC path on RDNA4.
