# catboost notes

## Build
- Pure CMake (Strategy A). Generated from Yandex ya.make; root CMakeLists.txt dispatches per-platform `CMakeLists.<os>-<arch>-cuda.txt` on `HAVE_CUDA`. `cmake/cuda.cmake` does `enable_language(CUDA)` + `find_package(CUDAToolkit)` and defines custom `target_cuda_sources()/target_cuda_flags()/target_cuda_cflags()` helpers (per-source nvcc `--compiler-options`). `.cu` added via `target_cuda_sources(<tgt> PRIVATE ...)`.
- Build: `cmake -DHAVE_CUDA=yes -DCUDAToolkit_ROOT=... [CUDAARCHS=...]` OR `build/build_native.py --have-cuda --cuda-root-dir=...`. Python pkg `setup.py --with-cuda` drives the same CMake (NOT a torch cpp_extension build).
- NO torch anywhere -> Strategy A confirmed.
- ya-make constraint: CMakeLists edits "should be simple/additive" (ported back to ya.make by maintainers). Keep HIP CMake changes minimal.

## GPU code surface (~96K LOC under catboost/cuda + library/cpp/cuda)
- 82 .cu, ~450 .cuh. Hot path: histogram kernels (pointwise_hist2_*, pairwise_hist_*, greedy_subsets_searcher/hist_*, compute_hist_loop_*) + cuda_util/kernel primitives (reduce/sort/scan/segmented_*/partitions/transform).
- ~60 cuda* runtime symbols, all 1:1 HIP (incl. cudaGraph*/stream-capture/cudaMallocManaged).
- Libraries: Thrust(2)+CUB(21) -> rocThrust/hipCUB (header-only, no MOAT dep). cuSOLVER present in source but STUBBED in OSS build (linear_cusolver_stub.cu). NO cuBLAS/cuRAND/cuFFT/cuSPARSE. No driver API. No textures/surfaces.
- Inline PTX: only instructions.cuh (bfe.u32, ld.global.cg.*, prefetch.global.L1) + kernel_helpers.cuh (asm("trap;")) = 6 sites.
- Hand-rolled float2/float4 operators in library/cpp/cuda/wrappers/kernel_helpers.cuh (collide with HIP vector types).

## Top fault classes (see plan.md)
1. wave64 warp-synchronous reductions (kernel_helpers.cuh x2: WarpReduce/FastInBlockReduce/BlockReduceN/SharedReduce4/8/WarpReduceN) -- races on wave64.
2. shuffle mask literals 0xFFFFFFFF (and a buggy 0xFFFFFF) -- fail to compile on HIP (needs 64-bit mask).
3. hardcoded warpSize 32 + 32-lane histogram layout (no central warpSize abstraction; TArchProps lacks it).
4. float2/float4 operator collisions with HIP_vector_type.
5. inline PTX.
6. __CUDA_ARCH__ numeric guards (~125) undefined on HIP -> fallback branch.
7. rule-of-five on graph/event/stream RAII handles.

## Inter-project MOAT deps: NONE (depends_on = []).

## Tests (validator gate)
- C++ GPU unit tests: catboost/cuda/*/ut/ built as Yandex unittest_main exes (add_yunittest). Highest signal: catboost-cuda-cuda_util-ut (test_reduce/scan/sort/segmented_*/compression_gpu). Run SERIALLY on one GPU; fixed-seed bit-identical determinism = wave64 reduction-race fingerprint.
- E2E: train task_type=GPU model, compare metrics vs CPU within tolerance + run-to-run stability.
- Non-GPU regression: CPU build + CPU ut untouched by clean Strategy-A port.
