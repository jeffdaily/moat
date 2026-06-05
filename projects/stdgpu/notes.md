# stdgpu notes

## Build Configuration (gfx90a, ROCm 7.2.x)

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DSTDGPU_BUILD_EXAMPLES=ON \
  -DSTDGPU_BUILD_TESTS=ON \
  -DSTDGPU_BUILD_BENCHMARKS=OFF

cmake --build build --config Release --parallel $(nproc)
```

## Port Summary

The port consists of two commits:

1. **CMake HIP language fix**: Mark all `.cpp` source files that link to stdgpu/rocThrust as `LANGUAGE HIP` to avoid g++ receiving HIP flags from rocThrust's `hip::device` target.

2. **Wave64/wave32 spinlock livelock fix**: Add `wave_lock_serialize()` helper to serialize contending lanes via `__ballot()`, fixing infinite hangs in concurrent containers.

## Wave64 Livelock Fix

On AMD GPUs (CDNA wave64, RDNA wave32), there is no per-lane forward-progress guarantee. When multiple lanes of the same wavefront contend for the same lock via try_lock() retry loops, the winning lane cannot make progress because it is stuck at SIMT reconvergence waiting on the losers who spin forever.

The fix uses `__ballot()` to elect one active lane at a time. Each lane takes its turn attempting the lock/operation, avoiding the reconvergence deadlock.

Files modified:
- `src/stdgpu/impl/wave_lock.h` (new): The wave-serialization primitive
- `src/stdgpu/impl/unordered_base_detail.cuh`: insert() and erase() retry loops
- `src/stdgpu/impl/deque_detail.cuh`: push_back/push_front/pop_back/pop_front
- `src/stdgpu/impl/vector_detail.cuh`: push_back/pop_back

## Test Results (gfx90a, wave64)

All 702 tests PASS, including previously hanging tests:
- `*insert_range_unique_parallel*` (4 tests)
- `*erase_range_unique_parallel*` (4 tests)
- `*simultaneous_push*` (5 tests)

## Known Upstream Issue

stdgpu GitHub: https://github.com/stotko/stdgpu
The HIP backend is marked "experimental" upstream. The upstream implementation assumes CUDA's Independent Thread Scheduling, which does not apply to AMD's SIMT model.
