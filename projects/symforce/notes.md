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

## Validation

The runtime library (libcaspar_runtime.a) compiles successfully for gfx90a. Object files contain HIP device code:
```
llvm-objdump --offloading build/CMakeFiles/caspar_runtime.dir/shared_indices.cu.o
# shows: hipv4-amdgcn-amd-amdhsa--gfx90a
```

Full integration testing requires symforce to be installed, which needs modifications to symforce's main CMakeLists.txt to support HIP. The caspar examples (kernel_example, bal, multiple_factors) test GPU execution with PyTorch but cannot run until the main symforce package supports USE_HIP.

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
- `atomicAdd_block` is undefined for HIP; aliased to `atomicAdd` (HIP shared-memory atomics are block-scoped by definition)
- `-ffast-math` causes NaN backward passes on HIP (clang's -fassociative-math reassociates online-softmax/layernorm reductions); use `-ffp-contract=fast -fno-math-errno` instead
- `cg::labeled_partition` butterfly reduction doesn't work for non-contiguous label groups (lanes at arbitrary positions share a label, but XOR only pairs specific distances); use per-lane atomicAdd fallback instead (shared-memory atomics are fast)

## Review 2026-06-05

Re-review after porter fixes. All three issues addressed correctly:

1. **labeled_partition per-lane atomicAdd**: Correct replacement for cg::labeled_partition. The butterfly approach fails for non-contiguous label groups (lanes at arbitrary positions share a label, but XOR only pairs specific distances). Per-lane atomicAdd to shared memory is simple and correct.

2. **atomicAdd_block alias**: Correct. HIP shared-memory atomics are block-scoped by definition, so atomicAdd_block -> atomicAdd is semantically correct.

3. **-ffast-math -> -ffp-contract=fast -fno-math-errno**: Correct per PORTING_GUIDE. clang's -ffast-math enables -fassociative-math which can NaN backward passes; the chosen flags retain FMA contraction without dangerous associativity reordering.

Minor cleanup items for PR-prep (not blockers):
- cuda_to_hip.h:54-104 contains dead code (reduce_max, match_any_mask, labeled_reduce_sum, CG_LABELED_REDUCE_SUM never used)
- Commit body contains "Strategy A" (MOAT vocabulary) -- scrub before upstream PR

Ready for validation.
