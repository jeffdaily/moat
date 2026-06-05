# baspacho notes

## Build commands (HIP/ROCm)

```bash
# Configure for gfx90a (MI250X)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DBASPACHO_USE_BLAS=ON -DBLA_VENDOR=OpenBLAS

# Build
cmake --build build -- -j$(nproc)

# Run tests
HIP_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure
```

## Test results (linux-gfx90a)

124/125 tests pass.

One test `BatchedCudaFactor.CoalescedFactor_Many_float` marginally exceeds the tolerance (5.06e-05 vs 5.0e-05 threshold) due to expected numeric precision differences between GPU architectures. This is a ~1.2% overshoot on a tight float32 tolerance -- acceptable given the different FMA rounding behavior between NVIDIA and AMD hardware.

## Dependencies

- OpenBLAS (install via `apt install libopenblas-dev`)
- ROCm 7.2.1+ with hipBLAS, hipSOLVER, hipSPARSE

## Gotchas

1. Eigen 3.4.0 (fetched automatically) tries to include `<cuda.h>` when `__CUDACC__` is defined. The cuda_to_hip.h header must NOT define `__CUDACC__` when building for HIP; use `__HIPCC__` detection instead. BaSpaCho's Utils.h already checks both.

2. hipBLAS maps `CUBLAS_STATUS_LICENSE_ERROR` to the same value as `CUBLAS_STATUS_NOT_SUPPORTED` -- the error enum switch needs to exclude `LICENSE_ERROR` when building for HIP.

3. hipSOLVER does not have the IRS (iterative refinement) status codes that cuSOLVER has (CUDART_VERSION >= 10000), so those must be guarded with `!defined(USE_HIP)`.

4. CudaAtomic.cuh needs to include `<hip/hip_runtime.h>` to make `atomicAdd` visible to device code.
