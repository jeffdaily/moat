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
