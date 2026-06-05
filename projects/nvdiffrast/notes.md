# nvdiffrast notes

## Validation 2026-06-05 (linux-gfx1100)

### Status: validation-failed

**Reason**: No moat-port branch exists to validate.

The lead platform (linux-gfx90a) was blocked during the porting phase due to fundamental wave64 incompatibility before any code was committed to a moat-port branch. The gfx1100 platform was moved to port-ready expecting to validate an existing wave32 port, but no such branch exists in the jeffdaily/nvdiffrast fork.

**Situation**:
- gfx90a: blocked at "porting" state (wave64 architectural issue prevents the cudaraster algorithm from working)
- gfx1100: set to "port-ready" but no moat-port branch to validate
- head_sha: null (no commits exist on any port branch)

**Next Action Required**: This project needs a porter to create the initial wave32-compatible port for gfx1100, following the plan's recommendation to use the ATLAS fork (https://github.com/ATLAS-0321/nvdiffrast-rocm) as a reference. The port will be gfx1100-only (wave32), with gfx90a remaining blocked indefinitely due to the architectural incompatibility.

---

## 2026-06-05: Wave64 Porting Attempt (gfx90a)

### Approach Taken
Attempted to use cooperative groups `tiled_partition<32>` to create 32-lane logical warps that work on both wave32 (gfx1100) and wave64 (gfx90a). Changes made:
- Created `WarpConfig.hpp` with per-arch warp size constants and logical warp operations
- Redirected `__ballot_sync`, `__any_sync`, `__all_sync` to cooperative groups tile operations
- Updated lane mask functions to use 32-lane tile semantics

### Test Results
Basic cooperative groups operations work correctly:
- `tile.ballot(pred)` returns correct 32-bit mask per tile
- `tile.thread_rank()` equals `threadIdx.x` (correct tile partitioning)
- Shuffle-based prefix sum works correctly within 32-lane tiles

However, the rasterizer crashes in `binRasterKernel` with:
```
HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception
```

### Root Cause Analysis
The cudaraster algorithm uses LDS-based warp-scan patterns that assume all threads WITHIN a divergent code block can sync via `__syncwarp`. On wave64:

1. A 64-lane wavefront contains TWO 32-thread logical warps (e.g., `threadIdx.y=0` and `threadIdx.y=1`)
2. Code patterns like:
   ```cpp
   if (threadIdx.y == 0 && someCondition) {
       // Warp scan with __syncwarp
       val += ptr[-1]; __syncwarp(mask);
       *ptr = val;     __syncwarp(mask);
   }
   ```
   ...cause the `__syncwarp` (mapped to `__builtin_amdgcn_wave_barrier`) to wait for ALL 64 lanes
3. Threads with `threadIdx.y == 1` never reach the barrier (they skip the `if` block)
4. Result: deadlock or undefined behavior

The fundamental issue is that `__syncwarp` on wave64 syncs the ENTIRE 64-lane wavefront, not just the active threads. CUDA's `__syncwarp(mask)` can sync a subset, but HIP has no equivalent on wave64.

### Required Fix (Not Implemented - Too Substantial)
To fix wave64 support, the rasterizer would need:
1. Replace ALL LDS-based warp scans with shuffle-based scans (tested: shuffle-based prefix sum works correctly)
2. Remove `__syncwarp` from divergent code paths or restructure them to be convergent
3. Use `tile.sync()` only at points where all 32 threads in the tile are guaranteed to reach

This is a significant algorithmic rewrite of the warp-scan patterns across 6+ files (~150 occurrences of warp primitives).

### Recommendation
Mark gfx90a as blocked. Proceed with gfx1100 (wave32) validation where the existing wave32-compatible port should work. The cooperative groups infrastructure added here will help if wave64 support is revisited.

---

## 2026-06-05: gfx90a (wave64) blocking issue (original analysis)

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
