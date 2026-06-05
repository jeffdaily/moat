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
