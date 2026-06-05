# stdgpu notes

## Build Configuration (gfx90a, ROCm 7.2.x)

The upstream HIP backend (STDGPU_BACKEND_HIP) builds after the following CMake fixes for ROCm 7.x flag propagation:

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

## Build Fixes Required

The HIP backend on ROCm 7.x has a CMake flag propagation issue: rocThrust -> hip::device adds `-x hip --offload-arch=gfx90a` to INTERFACE_COMPILE_OPTIONS gated on `$<COMPILE_LANGUAGE:CXX>`, causing g++ (the CXX compiler) to receive HIP flags it doesn't understand.

Fix: Mark all `.cpp` source files that link (directly or transitively) to stdgpu/rocThrust as `LANGUAGE HIP`:

1. `src/stdgpu/CMakeLists.txt`: Mark shared impl/*.cpp as HIP for HIP backend
2. `src/stdgpu/hip/CMakeLists.txt`: Mark hip/impl/*.cpp as HIP with TARGET_DIRECTORY
3. `examples/CMakeLists.txt`: Mark example .cpp files as HIP for HIP backend
4. `tests/stdgpu/CMakeLists.txt`: Mark test .cpp files as HIP for HIP backend

## Test Results (gfx90a, wave64)

### PASSING (510+ tests):
- All stdgpu_algorithm tests (25/25)
- All stdgpu_bit tests (15/15)
- All stdgpu_contract tests (9/9)
- All stdgpu_functional tests (26/26)
- All stdgpu_iterator tests (19/19)
- All stdgpu_limits tests (19/19)
- All stdgpu_memory tests (94/94)
- All stdgpu_atomic tests (138/138)
- All stdgpu_bitset tests (23/23)
- All stdgpu_mutex tests (11/11)
- Most stdgpu_unordered_map/set tests (excluding range insert/erase and duplicates)
- Many deque/vector tests (excluding concurrent push/pop)

### HANGING (deadlock/livelock on wave64):
- `*insert_range_unique_parallel*` tests
- `*insert_const_range_unique_parallel*` tests
- `*erase_range_unique_parallel*` tests
- `*erase_const_range_unique_parallel*` tests
- `simultaneous_push_back_and_pop_back` (deque)
- `simultaneous_push_front_and_pop_front` (deque)
- Other concurrent contention tests

## Root Cause Analysis

The hanging tests involve high concurrent contention on mutex/lock-based data structures (unordered_base, deque). The underlying issue appears to be:

1. The `try_insert`/`try_erase` functions use a retry loop on `failed_collision`
2. On wave64 (64 threads per wavefront), when many threads contend for the same mutex, they all spin in retry loops
3. This creates livelock conditions where threads cannot make forward progress
4. The single-element insert tests (`insert_unique_parallel`) work because they use individual element inserts with less contention
5. The range insert tests call the range-based `insert(begin, end)` which triggers different contention patterns

This is distinct from the wave64 DEDUP bug mentioned in PORTING_GUIDE (which is about duplicate key handling). The hangs affect even unique-key operations when using the range API.

## Recommendations

1. For downstream consumers (e.g., Open3D): Use the single-element insertion API (`for_each_index` with individual `insert()` calls) rather than range insertion
2. The atomics, bitset, mutex primitives themselves work correctly
3. A proper fix would require wave64-aware contention handling in the concurrent container algorithms

## Known Upstream Issue

stdgpu GitHub: https://github.com/stotko/stdgpu
The HIP backend is marked "experimental" upstream. These findings should be reported to upstream for potential fixes.
