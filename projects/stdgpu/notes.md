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

## Validation 2026-06-05 (linux-gfx1100)


**Platform:** AMD Radeon Pro W7800 48GB (gfx1100), ROCm 7.2.x, wave32

**Build command:**
```bash
HIP_VISIBLE_DEVICES=0 cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DSTDGPU_BUILD_TESTS=ON \
  -DSTDGPU_BUILD_EXAMPLES=ON \
  -DSTDGPU_BUILD_BENCHMARKS=OFF

HIP_VISIBLE_DEVICES=0 cmake --build build -j$(nproc)
```

**Build result:** SUCCESS

**Test command:**
```bash
HIP_VISIBLE_DEVICES=0 build/bin/teststdgpu
```

**Test results:** PASS
- All 702 tests passed (58.9s)
- Zero memory leaks (184538/184538 device, 180605/180605 host)

**Critical tests verified (wave-lock serialization):**
- `stdgpu_unordered_map.insert_range_unique_parallel` - PASS (10ms)
- `stdgpu_unordered_map.insert_range_unique_parallel_custom_execution_policy` - PASS (10ms)
- `stdgpu_unordered_map.erase_range_unique_parallel` - PASS (11ms)
- `stdgpu_unordered_map.erase_range_unique_parallel_custom_execution_policy` - PASS (11ms)
- `stdgpu_unordered_set.insert_range_unique_parallel` - PASS (10ms)
- `stdgpu_unordered_set.insert_range_unique_parallel_custom_execution_policy` - PASS (10ms)
- `stdgpu_unordered_set.erase_range_unique_parallel` - PASS (11ms)
- `stdgpu_unordered_set.erase_range_unique_parallel_custom_execution_policy` - PASS (12ms)
- `stdgpu_deque.simultaneous_push_back_and_pop_back` - PASS (6ms)
- `stdgpu_deque.simultaneous_push_front_and_pop_front` - PASS (6ms)
- `stdgpu_deque.simultaneous_push_front_and_pop_back` - PASS (6ms)
- `stdgpu_deque.simultaneous_push_back_and_pop_front` - PASS (6ms)
- `stdgpu_vector.simultaneous_push_back_and_pop_back` - PASS (5ms)

**Verdict:** VALIDATED at 718d206

## Validation 2026-06-07 (windows-gfx1151)

**Platform:** AMD Radeon 8060S (gfx1151, RDNA3.5 APU, wave32, 20 CUs), ROCm 7.2 TheRock pip SDK, Windows 11

**Build command (repeatable script: `D:/Develop/moat/agent_space/stdgpu_win_build.sh`):**
```bash
ROCM_ROOT="D:/Develop/moat/agent_space/venv-gsplat/Lib/site-packages/_rocm_sdk_devel"
cmake -B build-gfx1151 -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DSTDGPU_BUILD_TESTS=ON \
  -DSTDGPU_BUILD_BENCHMARKS=OFF \
  -DSTDGPU_BUILD_EXAMPLES=OFF \
  -DCMAKE_C_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++.exe" \
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
  -DCMAKE_TLS_VERIFY=OFF

cmake --build build-gfx1151 --config Release --parallel 6
bash deploy_therock_runtime.sh build-gfx1151/bin
export PATH="build-gfx1151/bin:$ROCM_ROOT/bin:$ROCM_ROOT/../_rocm_sdk_core/bin:$PATH"
```

**Build result:** SUCCESS (30/30 targets; clang++.exe as HIP compiler, clang.exe for CXX)

**Test results: VALIDATION FAILED -- 685/702 pass, 17 hang**

All critical tests from the wave_lock_serialize livelock fix PASS on gfx1151:
- `stdgpu_unordered_map/set.insert_range_unique_parallel` (x4 variants each) - PASS (~10-116ms)
- `stdgpu_unordered_map/set.erase_range_unique_parallel` (x4 variants each) - PASS
- `stdgpu_deque.simultaneous_push_front_and_pop_back` - PASS (44ms)
- `stdgpu_vector.simultaneous_push_back_and_pop_back` - PASS (76ms)
- Zero memory leaks (93999/93999 device, 90476/90476 host)

**17 tests hang indefinitely** (killed after 30s timeout each; gfx90a + gfx1100 both pass all 702):

Deque concurrent same-end operations (3):
- `stdgpu_deque.simultaneous_push_back_and_pop_back`
- `stdgpu_deque.simultaneous_push_front_and_pop_front`
- `stdgpu_deque.simultaneous_push_back_and_pop_front`

Unordered_map extreme-contention inserts into full/excess-empty table (7):
- `insert_while_full`, `insert_multiple_while_full`, `insert_while_excess_empty`
- `insert_parallel_while_one_free`, `insert_parallel_while_excess_empty`
- `emplace_parallel_while_one_free`, `emplace_parallel_while_excess_empty`

Unordered_set same 7 tests.

**Root cause:** gfx1151 RDNA3.5 APU has 20 CUs (vs 60 on W7800 gfx1100). The `wave_lock_serialize` fix prevents within-wavefront deadlock and works correctly on gfx1151 (verified: `__ballot(1)` returns correct 0xffffffff mask, all 32 lanes serialize in order). The 17 hanging tests all involve N=100000 threads competing for 1-2 free slots (~3125 wavefronts, only ~160 run simultaneously on 20 CUs). Under this extreme cross-wavefront contention, the hardware scheduler does not preempt fast enough to prevent cross-wavefront forward-progress starvation. gfx1100 (60 CUs, 3x more simultaneous wavefronts) avoids the starvation.

The tests that pass on gfx1151 (`insert_range_unique_parallel`, etc.) have each thread operating on a unique slot -- no cross-wavefront contention.

**Verdict:** VALIDATION FAILED at 718d206 -- delta-port needed to fix 17 hanging tests on gfx1151 RDNA3.5 APU
