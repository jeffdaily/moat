# cuPDLPx notes

## Build (linux-gfx90a)

```bash
cd projects/cuPDLPx/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCUPDLPX_BUILD_CLI=ON -DCUPDLPX_BUILD_TESTS=OFF -DCUPDLPX_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

For other architectures:
```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 ...  # RDNA3
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 ...  # Windows
```

## Test

```bash
# Download test LP instance
wget https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz

# Run solver (gfx90a MI250X)
HIP_VISIBLE_DEVICES=0 ./build/cupdlpx 2club200v15p5scn.mps.gz . -v

# Expected output:
# Status: OPTIMAL
# Primal objective: ~-121.22
# Relative residuals < 1e-4
```

## Port details

- Strategy A: cuda_to_hip.h compat header, LANGUAGE HIP in CMake
- cuBLAS -> hipBLAS, cuSPARSE -> hipSPARSE, CUB -> hipCUB
- CUDA Graph API maps cleanly to HIP Graph
- No warp intrinsics, no textures: no warp-size issues

## Gotchas

- hipCUB header is C++ only (includes `<algorithm>`); the compat header guards it
  with `#ifdef __cplusplus` so C files compile cleanly
- cusparseSpMVOp is not available in hipSPARSE; the port forces `CUPDLPX_HAS_SPMVOP=0`
  so the standard cusparseSpMV path is used (this is already the default path in upstream
  cuPDLPx unless you have CUDA 13.1+)
- cublasDnrm2_v2_64 maps to hipblasDnrm2 (32-bit size_t); LP problem sizes fit comfortably
