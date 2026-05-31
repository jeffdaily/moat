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

## PORTER (linux-gfx90a, ROCm 7.2.1, MI250X) -- build recipe

Fork: jeffdaily/catboost, branch moat-port. Actions disabled.

Host deps (NOT in tree; install once): `pip install 'conan>=2.0.5' 'Cython>=3.0.10'`; `conda install -n py_3.12 -c conda-forge openjdk` (Spark JNI find_package(Java)); the build's other tools (ragel/swig/yasm/openssl) come from conan. Host + device compiler = ROCm clang (`/opt/rocm/llvm/bin/clang++`); only `clang`/`clang++` on PATH IS ROCm clang here, which is ideal (consistent ABI for host C++ and HIP objects).

Repro scripts in agent_space/catboost/: `env.sh` (PATHs + CC/CXX), `configure.sh` (full cmake line), `build.sh`. One-time conan install: `cd src && conan install . --output-folder=build_hip --build=missing -s build_type=Release -s compiler.cppstd=20`. Then `bash agent_space/catboost/configure.sh`.

Key cmake knobs (all needed):
- `-DHAVE_HIP=ON` (new option). Root CMakeLists also `set(HAVE_CUDA ON)` under HAVE_HIP so the per-dir generated dispatchers (which gate the GPU subtree on HAVE_CUDA) include their *-cuda.txt; cuda.cmake takes its HIP branch first and `return()`s before the CUDA-enable block. Avoids editing dozens of generated dispatchers.
- `-DCMAKE_{C,CXX,HIP}_STANDARD_LIBRARIES="-lc -lm"` REQUIRED. catboost uses its bundled libcxxcuda11 (`-nostdinc++`) and does not pull libc by default under ROCm clang/lld; without these, host tools (flatc) and the ut exe fail to link (`undefined: stderr/abort/memset/__libc_start_main`). The exe's LINKER_LANGUAGE is HIP, so CMAKE_HIP_STANDARD_LIBRARIES matters too.
- conan toolchain for deps + `-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=cmake/conan_provider.cmake`.

Build the highest-signal GPU unit test: `ninja -C build_hip/cm -j16 catboost-cuda-cuda_util-ut`. Cap -j16 (shared 128-core box).

## PORTER -- fault fixes applied (commit b3ee4cc)

Compat: `library/cpp/cuda/wrappers/cuda_to_hip.h` force-included on HIP TUs (`-include` in CMAKE_HIP_FLAGS). `library/cpp/cuda/wrappers/hip_compat/` = forwarding shims (cuda_runtime.h/cuda_runtime_api.h/cuda_fp16.h/math_constants.h/cub-*->hipCUB/cooperative_groups.h), on the include path BEFORE /opt/rocm/include (host TUs include these toolkit names directly).

1. wave64 reductions (THE risk) -- both kernel_helpers.cuh. On HIP, FastInBlockReduce/BlockReduceN/SharedReduce4/SharedReduce8 drop the 32-lane warp-synchronous tail and run the __syncthreads tree to size 1 (MPPI lesson); the NKernel WarpReduce(data*) HIP path uses shared-mem stepping with __syncwarp (correct: all callers are non-divergent per-32-group, wavefront lockstep). Shuffle masks -> 64-bit CB_FULL_WARP_MASK (HIP `*_sync` static_assert sizeof(mask)==8; also fixes a latent 24-bit 0xFFFFFF in WarpReduceN). TTileReducer<64> uses the native 64-lane CG tile (CUDA caps thread_block_tile at 32, hence its asm("trap;")).
2. histogram_utils.cu ScanHistogramsImpl: `cub::WarpScan<double>` -> `cub::WarpScan<double, 32>`. hipCUB defaults LOGICAL_WARP_THREADS to the 64-wide wavefront, which would scan TWO features (warpId = tid/32) together. Width 32 is identical on CUDA. (ShuffleIndex<32> already fine; hipCUB ignores its mask arg.)
3. linear_solver.cu / dcg.cu: `__shfl_sync(0xFFFFFFFF, x, 0, logicalWarpSize)` -> CB_FULL_WARP_MASK (logicalWarpSize<=32 sub-warp broadcast, correct on wave64 because ShuffleReduce's lane-0 result and the width-N broadcast stay in-group).
4. float2/float4 (wrappers/kernel_helpers.cuh) + uint2 (cuda_util/kernel/operators.cuh) operator overloads guarded `#if !defined(USE_HIP)` (HIP_vector_type provides them). Kept the non-operator helpers (Dot4/FMA4/Max4/Sqrt4/ZeroAwareDivide).
5. instructions.cuh inline PTX (bfe.u32/ld.global.cg.*/prefetch) -> USE_HIP branch: bit-extract / cub::ThreadLoad<LOAD_CS> / no-op. asm("trap;") TTileReducer<128|256> -> __builtin_trap().
6. segmented_sort.cu: on HIP, seed tmpKeys/tmpValues with a copy of keys/values before the sort. rocPRIM's DeviceSegmentedRadixSort::SortPairs with a DoubleBuffer flips Current() to the alternate buffer and writes only the segment-covered ranges; the wrapper's full-range copy-back then propagated undefined gap elements (fingerprint: TestSegmentedSortNonContinous, keysAfterSort[21]==0 -- a gap index). Standalone repro agent_space/catboost/segsort{2,3,4}.cpp.
7. reduce.cu: `#include <cuda/std/version>` (+ <cuda/functional>) guarded `#if !defined(USE_HIP)`; the existing `thrust::plus` fallback is taken on HIP (no libcu++ in ROCm; rocThrust ships <thrust/functional.h>).
8. `sum = {0}` (update_part_props.cu) -> make_float4(0,0,0,0) (HIP_vector_type has no `= {0}` assign). NB evaluator.cu:193 `double4 = {0}` still pending (model-inference target, not yet built).
9. mvs.cu: hipCUB CacheModifiedInputIterator ctor takes non-const T* -> const_cast (USE_HIP).
10. `<< <`/`>> >` spaced kernel-launch tokens -> `<<<`/`>>>` (171 sites, 33 files; clang/hipcc requires no spaces, nvcc tolerated them). Mechanical sed; verified no operator<< / shift false positives.

cuda.cmake CUDA-vs-HIP gotchas:
- HIP's device_library_decls.h does `#define __local __attribute__((address_space(3)))` which corrupts the bundled libc++ `<deque>` (`__local()` member) -> `#undef __local` in the compat header after the hip include. Affects ANY host TU that includes a STL header after hip_runtime.
- bundled libc++ placement `operator new(size_t,void*)` is host-only (no __device__); nvcc supplies an implicit device placement-new, clang/hipcc does not -> rocPRIM device code fails. Compat header adds `__device__` placement new/delete under `__HIP_DEVICE_COMPILE__` (clang HIP target-overloading lets them coexist with the host-only ones).
- cudaPointerAttributes.type gated on `CUDART_VERSION >= 10000` -> define CUDART_VERSION 12000 (hipPointerAttribute_t has .type).
- `find_package(CUDAToolkit REQUIRED)` + `CUDA::toolkit` in build/internal/platform/cuda: cuda.cmake aliases CUDA::toolkit to hip::host+hip::device and no-ops the CUDAToolkit find_package (single chokepoint all GPU targets link).
- target_cuda_sources/flags/cflags and target_link_options overridden in the cuda.cmake HIP branch (strip -lcudart_static/-lcudadevrt/-lculibos, add HIP runtime to exes/shared libs). Zero edits to the 80+ generated per-target CMakeLists.

GPU validation (MI250X GCD, HIP_VISIBLE_DEVICES=0, serial): catboost-cuda-cuda_util-ut = 48/48 pass; two back-to-back serial runs give an identical pass set (deterministic -> no residual wave64 race). Pick a free GCD via rocm-smi (box has 4, others run concurrent jobs).

Full `catboost` CLI also builds (217MB, all histogram kernels + greedy_subsets_searcher). E2E: `catboost fit --task-type GPU --devices 0` on a 4000-row/10-feature synthetic Logloss problem (200 trees, depth 6, seed 42) gives test AUC 0.96341 vs the CPU path's 0.96494 (within 0.0015; GBDT GPU/CPU never bitwise-equal). Two same-seed GPU runs are bit-identical (0.9634065032) -> the GBDT GPU training path is deterministic on wave64. Dataset + commands in agent_space/catboost/ (train.tsv/test.tsv/train.cd, gpu_run/cpu_run train-dirs). lead linux-gfx90a -> ported at b7113fe.

Additional fixes beyond the list above (found building the full engine, all in commit b7113fe): clang two-phase-lookup `template` disambiguator on dependent member calls hist.AddPoints<>/AddPairs<> and impl->AddPointsImpl<> (compute_pair_hist_loop.cuh, compute_hist_loop_one/two_stats.cuh, hist_2_one_byte_base.cuh); TPartitionStatistics (gpu_structures.h) and TDataPartition (cuda_util/gpu_data/partitions.h) constructors made __host__ __device__ so hipCUB ThreadLoad can default-construct them in device code (partitions.h uses a self-contained Y_CUDA_HOST_DEVICE macro keyed on __CUDACC__/__HIPCC__ since it is included by host .cpp before any runtime); model evaluator.cu `uchar4/uint4/double4 = {0}` -> make_*4 (explicit-ctor copy-init); `#include <cfloat>` added to the compat header (FLT_MAX); one `<<< ... >> >` launch split across two lines in split_points.cu normalized to `<<<...>>>`.
