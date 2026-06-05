# metaeuk notes

## Build (linux-gfx90a)
```bash
cmake -S projects/metaeuk/src -B projects/metaeuk/build \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build projects/metaeuk/build -j16
```

## Port notes
- metaeuk vendors MMseqs2 (lib/mmseqs) which vendors libmarv (the GPU library)
- Port follows same Strategy A as standalone MMseqs2 port (jeffdaily/MMseqs2@moat-port)
- cuda_to_hip.h maps runtime symbols + SIMD intrinsics (via marv_simd_amd.cuh)
- hip_compat/ redirects <cuda_fp16.h> and <cooperative_groups*> includes
- Kernel configs select conservative sm75 set on AMD (half2 path, 64KB shared)
- cg::reduce calls replaced with portable marv_tile_reduce on HIP
- libmarv built as shared library with HIP_RESOLVE_DEVICE_SYMBOLS=ON
- rocThrust namespace mapped: thrust::cuda -> thrust::hip via using declarations
