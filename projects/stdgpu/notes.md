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

## Review 2026-06-05

Reviewed commit 718d206 (wave64/wave32 spinlock livelock fix).

**Verdict: APPROVE**

The wave_lock_serialize() pattern is correct per PORTING_GUIDE line 282:
- Uses `__ballot(1)` to get the active lane mask (returns 64-bit on HIP)
- Elects one lane at a time via `__ffsll()` and clears each lane's bit after its turn
- Properly guarded with `#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)`
- CUDA path is a no-op (Independent Thread Scheduling handles it)

All contention sites wrapped:
- unordered_base: insert() and erase() retry loops (lines 978, 1027)
- deque: push_back/push_front/pop_back/pop_front (4 sites)
- vector: push_back/pop_back (2 sites)

Namespace usage is correct:
- wave_lock.h declares in `stdgpu::detail`
- unordered_base_detail.cuh (inside detail ns) calls `wave_lock_serialize()`
- deque_detail.cuh and vector_detail.cuh (inside stdgpu ns) call `detail::wave_lock_serialize()`

No issues found:
- No hardcoded 32 or 0xffffffff masks (uses 1ull << elected)
- No MOAT jargon in comments
- Commit title under 72 chars, [ROCm] prefix present
- No noreply trailer
- Author is jeffdaily account
- All 702 tests pass per commit message

Minor note (non-blocking): Copyright header says "Copyright 2024 Advanced Micro Devices" but commit date is 2026. Cosmetic only.

## Validation 2026-06-05 (linux-gfx90a)

**Platform:** AMD Instinct MI250X (gfx90a), ROCm 7.2.1, wave64

**Build command:**
```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DSTDGPU_BUILD_TESTS=ON

cmake --build build -j$(nproc)
```

**Build result:** SUCCESS (benchmark target failed due to CXX file receiving HIP flags, but all tests and examples built successfully)

**Test command:**
```bash
build/bin/teststdgpu
```

**Test results:** PASS
- All 702 tests passed in both runs (93.5s and 97.7s)
- Zero memory leaks (364762/364762 device, 360829/360829 host)

**Critical tests verified (previously hanging on wave64):**
- `stdgpu_unordered_map.insert_range_unique_parallel` - PASS (14ms)
- `stdgpu_unordered_map.insert_range_unique_parallel_custom_execution_policy` - PASS (14ms)
- `stdgpu_unordered_map.erase_range_unique_parallel` - PASS (15ms)
- `stdgpu_unordered_map.erase_range_unique_parallel_custom_execution_policy` - PASS (15ms)
- `stdgpu_unordered_set.insert_range_unique_parallel` - PASS (14ms)
- `stdgpu_unordered_set.insert_range_unique_parallel_custom_execution_policy` - PASS (14ms)
- `stdgpu_unordered_set.erase_range_unique_parallel` - PASS (15ms)
- `stdgpu_unordered_set.erase_range_unique_parallel_custom_execution_policy` - PASS (15ms)
- `stdgpu_deque.simultaneous_push_back_and_pop_back` - PASS (7ms)
- `stdgpu_deque.simultaneous_push_front_and_pop_front` - PASS (7ms)
- `stdgpu_deque.simultaneous_push_front_and_pop_back` - PASS (7ms)
- `stdgpu_deque.simultaneous_push_back_and_pop_front` - PASS (7ms)
- `stdgpu_vector.simultaneous_push_back_and_pop_back` - PASS (3ms)

**Determinism:** Confirmed (ran test suite twice, all 702 tests passed both times)

**Verdict:** VALIDATED at 718d206
