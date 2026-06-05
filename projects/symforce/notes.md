# symforce notes

## Build

The Caspar module (CUDA code generation and execution backend) runtime library can be built standalone for HIP:

```bash
cd symforce/caspar/source/runtime
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
make -j$(nproc)
```

For generated Caspar libraries (via the Python codegen pipeline), pass `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a` to cmake.

## Port details

### HIP cooperative groups gaps

HIP (ROCm 7.2.1) lacks several CUDA cooperative_groups features:
- `cg::reduce` - replaced with manual butterfly shfl_xor reduction
- `cg::labeled_partition` - replaced with match_any + masked butterfly reduction
- `cg::memcpy_async` / `cg::wait` - replaced with synchronous block-strided copy

These replacements are guarded by `#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)` so CUDA behavior is unchanged.

### Wave64 safety

The code uses `tiled_partition<32>` which creates 32-lane tiles. On wave64 (gfx90a), HIP CG supports 32-lane tiles within a 64-lane wavefront. The SumStore two-level reduction and other 32-wide operations work correctly as they operate within the tile, not the full wavefront.

### Gotchas

- HIP's `group_dim()` is not const; had to use `blockIdx.x * blockDim.x` directly for HIP
