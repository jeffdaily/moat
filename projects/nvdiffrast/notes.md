# nvdiffrast notes

## 2026-06-05: gfx90a (wave64) blocking issue

### Build Status
The nvdiffrast HIP port compiles successfully on gfx90a using the ATLAS fork's hipraster module:
```bash
cd /var/lib/jenkins/moat/projects/nvdiffrast/src
pip install -e . --no-build-isolation
```
Extension builds: `_nvdiffrast_c.cpython-312-x86_64-linux-gnu.so` (47MB)

### Runtime Failure
Running `samples/torch/triangle.py` crashes with a GPU memory fault:
```
Memory access fault by GPU node-5 (Agent handle: ...) on address ...
Reason: Unknown
```
The crash occurs in `binRasterKernel` during the first rasterization call.

### Root Cause: Wave64 Incompatibility
The cudaraster/hipraster rasterizer algorithm is fundamentally designed for 32-lane warps (CUDA's warp size). On gfx90a (CDNA2), the wavefront size is 64 lanes with no wave32 mode. Key issues:

1. **Lane indexing**: Code uses `threadIdx.x + threadIdx.y * 32` assuming 32 threads per warp
2. **Lane masks**: All lane masks are 32-bit (`U32`), but wave64 uses 64-bit masks
3. **Prefix sums**: Warp-level prefix sums assume exactly 32 lanes
4. **Shared memory**: Arrays sized for 32-lane warps
5. **Ballot operations**: ~147 uses of `__ballot`, `__any`, `__all`, `getLaneMask*` with 32-bit assumptions

The ATLAS fork (used as reference) targets gfx1100 (RDNA3, wave32) only. Their documented fixes (L-036 getLaneMaskLe UB, L-038/L-043/L-044 bounds checks) apply to wave32 only.

### Assessed Options

1. **Logical warps (32 threads in 64-lane wave)**: Would require fundamentally different synchronization semantics. The algorithm relies on all 32 threads being lockstep, which doesn't apply when they're distributed across a 64-lane wave.

2. **Wave64 rewrite**: Would require rewriting ~147 warp intrinsic uses across 6 files (BinRaster.inl, CoarseRaster.inl, CoarseRasterSimple.inl, FineRaster.inl, TriangleSetup.inl, Util.inl). Each use would need:
   - 64-bit lane masks
   - Different prefix sum structure
   - Adjusted shared memory sizes
   - Changes to kernel launch bounds

3. **Alternative rasterization backend**: The OpenGL backend was deprecated in nvdiffrast; only the CUDA/HIP rasterizer remains. No CPU fallback exists.

### Recommendation
Mark gfx90a as blocked; proceed with gfx1100 (wave32) validation. The port is architecturally sound for RDNA3 and later wave32-capable GPUs. CDNA architectures (gfx90a, MI100, MI250X) cannot run this code without a fundamental algorithmic rewrite.

### Files Modified
- `setup.py` -- HIP detection, hipraster source selection
- `csrc/common/framework.h` -- HIP/ROCm header includes
- `csrc/common/common.h` -- HIP intrinsic macros
- `csrc/torch/torch_common.inl` -- HIP-compatible device check
- `csrc/common/hipraster/` -- Parallel HIP rasterizer (from ATLAS reference)
