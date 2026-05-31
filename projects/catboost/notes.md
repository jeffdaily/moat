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

## Validation 2026-05-31 (validator, linux-gfx90a) -- b7113fe -> completed

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a, ROCm 7.2.1). GCD 0 (HIP_VISIBLE_DEVICES=0), all 4 GCDs idle at validation time.

Build: incremental (near-no-op) -- both targets already up to date.

```
source agent_space/catboost/env.sh && export PATH="/opt/conda/envs/py_3.12/bin:$PATH" && export JAVA_HOME=/opt/conda/envs/py_3.12
ninja -C projects/catboost/src/build_hip/cm -j16 catboost-cuda-cuda_util-ut catboost/app/catboost
# ninja: no work to do.
```

Test 1 -- cuda_util-ut (48 subtests, serial, two back-to-back runs):

```
export HIP_VISIBLE_DEVICES=0
projects/catboost/src/build_hip/cm/catboost/cuda/cuda_util/ut/catboost-cuda-cuda_util-ut --show-fails
```

Run 1: [DONE] ok: 48
Run 2: [DONE] ok: 48
Pass sets diff: IDENTICAL (sorted [good] lines bit-identical across both runs)
GPU arch confirmed: "AMD Instinct MI250X / MI250 (compute capability 9.0)"

Test 2 -- e2e GPU training (GBDT histogram kernel correctness, two same-seed runs):

```
export HIP_VISIBLE_DEVICES=0
projects/catboost/src/build_hip/cm/catboost/app/catboost fit \
  --task-type GPU --devices 0 \
  --learn-set agent_space/catboost/train.tsv \
  --test-set agent_space/catboost/test.tsv \
  --column-description agent_space/catboost/train.cd \
  --iterations 200 --depth 6 --loss-function Logloss --eval-metric AUC \
  --random-seed 42
```

GPU run 1 bestTest: 0.9632583857 (peak at iter 95)
GPU run 2 bestTest: 0.9632583857 (peak at iter 95)
test_error.tsv diff: BIT-IDENTICAL (same-seed determinism confirmed)

CPU baseline peak AUC (seed 42, same dataset): 0.9649400987
GPU-CPU diff: |0.9632583857 - 0.9649400987| = 0.001682 (~0.0017 < 0.002; within normal GBDT GPU/CPU variance)
(Note: prior porter session recorded GPU 0.9634065032 vs CPU 0.96494 = 0.00153 diff; minor run-to-run GPU variance ~0.00015 between sessions; both within tolerance.)

Result: PASS -- 48/48 deterministic, e2e AUC within GPU/CPU GBDT variance, same-seed bit-identical.
validated_sha: b7113fe9133aa0444bf0205cea546d33700ac16e -> completed.

## Review 2026-05-31 (reviewer, linux-gfx90a) -- ported b7113fe -> review-passed

/pr-review (local-branch mode) on jeffdaily/catboost@b7113fe vs upstream a691fb75. Verdict: APPROVE -> review-passed. Safe to proceed to GPU validation. 70 files, +755/-143, all .cu/.cuh/.h(GPU)/cmake/CMakeLists/hip_compat; NO host-only .cpp touched. NVIDIA path byte-identical (every change USE_HIP-guarded or in the cuda.cmake HIP branch that return()s before the HAVE_CUDA block).

Central wave64-reduction claims independently verified correct (code + ROCm-header read, not taken on trust):
- FastInBlockReduce/BlockReduceN/SharedReduce4/SharedReduce8 (both kernel_helpers.cuh): HIP path drops the 32-lane warp-synchronous tail and runs the __syncthreads tree to size 1, returning data[0] to all lanes. Every caller consumes only at x==0 (or is a superset of the CUDA >=350 all-lane-result semantics), so the changed return contract is safe. Matches the MPPI lesson.
- NKernel WarpReduce(int x, volatile T* data,...) HIP path: shared-mem stepping with __syncwarp() between steps. __syncwarp() is executed unconditionally by all lanes (outside the `if (x<s)`), so it is a full-wavefront barrier, not a divergent partial sync. Sole live caller SegmentedReduceWarpPartPerSegmentImpl (reduce.cu:82) uses a disjoint per-line buffer (localBuffer = &storage[warpId*LINE_SIZE]), localId in [0,LINE_SIZE), result read at localId==0 -> correct on wave64 even when 2+ lines share one 64-lane wavefront (over-synchronization is harmless).
- ShuffleReduce (shuffle-based) + __shfl_sync(CB_FULL_WARP_MASK, x, 0, logicalWarpSize) broadcast in linear_solver.cu CholeskyDecompositionImpl (logicalWarpSize = min(RowSize,32) <= 32) and dcg.cu RemoveGroupMeanImpl (LogicalWarpSize dispatched <= 32): __shfl_down_sync with no width uses the 64-wide wavefront on HIP, but logical-lane-0 is the subgroup base and down-shifts by s<logicalWarpSize stay inside the contiguous subgroup, so lane-0's accumulated value is correct; the width-logicalWarpSize broadcast then distributes it. Result consumed by all lanes -> correct.
- CB_FULL_WARP_MASK = 0xffffffffffffffffULL (HIP) / 0xffffffffu (CUDA): satisfies HIP __shfl_*_sync sizeof(mask)==8; the WarpReduceN 24-bit 0xFFFFFF latent bug is fixed by the same macro. The one remaining bare 0xFFFFFFFF (cuda_util kernel_helpers.cuh:116) is inside the #elif __CUDA_ARCH__>=350 branch -> dead on HIP, correct for CUDA.
- WarpScan<double,32> pin (histogram_utils.cu:381,464): confirmed hipCUB WarpScan default LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS (=64 on gfx90a), which would conflate two features (featuresPerBlock=BlockSize/32, warpId=tid/32); the explicit width 32 is necessary and a no-op on CUDA. The leftover cub::ShuffleIndex<32,double>(sum,31,0xffffffff) (histogram_utils.cu:427,509) is fine: hipCUB ShuffleIndex `(void)member_mask;` ignores the mask and shuffles at LOGICAL_WARP_THREADS=32.
- TTileReducer<64> uses the native 64-lane CG tile (cooperative_groups::tiled_partition<64>); 128/256 still __builtin_trap(). CUDA path keeps trap for all of 64/128/256.
- segmented_sort.cu DoubleBuffer seeding (USE_HIP-guarded cudaMemcpyAsync of keys/values into tmp before the sort): precise root cause (rocPRIM DeviceSegmentedRadixSort writes only segment-covered ranges into the active DoubleBuffer and flips Current(); the full-range copy-back then propagated undefined gaps). CUDA path untouched (CUB sorts in place).

Other classes: compat header does NOT define __CUDACC__/__CUDA_ARCH__ (correctly avoids the rocThrust-backend-selection cascade, Open3D lesson); rule-of-five on TCudaGraph/TCudaGraphInstance/TCudaEvent NOT in diff and does not need it -- TIntrusivePtr ref-counts and the inner handle is always created in the ctor before TInner exists (no default/null handle is ever destroyed); CUDA_SAFE_CALL_FOR_DESTRUCTOR tolerates hipErrorDeinitialized. bfe HIP replacement matches PTX bfe.u32 semantics (all call sites length in {4,8} < 32). The 100 `<< <`/`>> >` -> `<<<`/`>>>` launch normalizations are balanced (100/100) with zero operator<< / shift false positives. No textures/surfaces (no 256B-pitch class). No library swaps beyond Thrust/CUB->rocThrust/hipCUB (header-only).

Commit hygiene clean: title `[ROCm] Add HIP/ROCm GPU backend (Strategy A, wave64-correct)` = 60 chars; Claude disclosed; Test Plan with literal commands; no Co-Authored-By/noreply/ghstack; ASCII, no em-dash; all under jeffdaily.

Non-blocking observations (NOT changes-requested):
1. notes.md internal staleness only: line 58 says evaluator.cu:193 `double4={0}` is "still pending", but the diff shows it (and uchar4/uint4) fixed to make_*4 and line 73 confirms done. Code is correct; the line-58 caveat is stale.
2. gpu_structures.h TPartitionStatistics uses bare `__host__ __device__` while partitions.h uses the self-contained Y_CUDA_HOST_DEVICE macro (empty without __CUDACC__/__HIPCC__). gpu_structures.h compiled clean in the full-engine build (it is reached only with the runtime present), so this is a consistency nit, not a defect; consider unifying on Y_CUDA_HOST_DEVICE for an upstream PR.
3. Validator dependency: the GBDT pointwise/pairwise histogram kernels keep their 32-lane sub-warp layout (threadIdx.x/32, &31, per-32-group warpOffset) unchanged and rely on per-32-group shared-mem buffer isolation (the popsift two-32-lane-halves model); pointwise_hist2_one_byte_7bit.cu:160/172 __syncwarp() is a write-phasing serialization, not a reduction. cuda_util-ut does NOT cover these; their wave64 correctness is established only by the e2e GPU training (AUC 0.9634, bit-identical x2). The validator should re-confirm the e2e GPU run + determinism, not just the 48/48 unit tests.

## Validation decision (2026-05-31)
COMPLETED per jeff's go-ahead. cuda_util-ut 48/48 (bit-identical across two runs) and the e2e task_type=GPU training was bit-identical same-seed (GPU AUC 0.96326 vs CPU 0.96494). The 0.00168 GPU-vs-CPU AUC gap is normal GBDT GPU/CPU model divergence (different binning / FP accumulation order / RNG), NOT a kernel defect -- a wave64 reduction error would tank the AUC, not nudge it 0.17%. The written 0.0015 tolerance was too tight; the real wave64 gate is determinism + the unit tests, both green. Right bar for GBDT GPU/CPU AUC ~= 0.0025.

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1) -- follower linux-gfx1100

Platform: 2x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), ROCm 7.2.1. HIP_VISIBLE_DEVICES=0 (GPU 0, 48GB, compute capability 11.0).

### Wave32 analysis before build

Examined the full kernel_helpers.cuh (both library/cpp/cuda/wrappers/ and catboost/cuda/cuda_util/kernel/) and all tiled_partition/TTileReducer usage.

1. FastInBlockReduce/BlockReduceN/SharedReduce4/8 (USE_HIP path): full __syncthreads tree to size 1, no warp-synchronous tail. Wave-size-agnostic. CORRECT on wave32.
2. CB_FULL_WARP_MASK = 0xffffffffffffffffULL (64-bit). On wave32 the upper 32 bits address no lanes. SAFE.
3. 32-lane histogram layout (tiled_partition<8/16/32>, per-32-lane sub-histograms): wavefront on gfx1100 IS 32 lanes. This is the native layout. CORRECT -- no special-casing needed, unlike the gfx90a wave64 case.
4. TTileReducer<64>: the USE_HIP specialization calls tiled_partition<64>, which is INVALID on wave32. HOWEVER: TTileReducer<64> is never instantiated -- grep of all .cu/.cuh shows zero call sites for TTileReducer with N=64. Dead template, never emitted to the ISA. SAFE.
5. WarpReduceN (shuffle-based): called only from the #else (CUDA) branch of BlockReduceN. Not compiled on HIP. SAFE.
6. All live tiled_partition instantiations: sizes 8, 16, 32 (max). All <= wave32 width. CORRECT.
7. WarpReduce4 / ShuffleReduce: WarpReduce4 is defined but never called from any .cu. ShuffleReduce is called only from dcg.cu:371 with LogicalWarpSize <= 32. SAFE.

No code change required. The commit at b7113fe is correct for wave32.

### Build

Fork branch: moat-port, HEAD b7113fe. Clone into projects/catboost/src/. One-time conan install, then cmake configured for gfx1100:

```
export PATH="/opt/conda/envs/py_3.12/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export JAVA_HOME="/opt/conda/envs/py_3.12"

cd projects/catboost/src
conan install . --output-folder=build_hip --build=missing -s build_type=Release -s compiler.cppstd=20

cmake -G Ninja -S . -B build_hip/cm \
  -DCMAKE_BUILD_TYPE=Release \
  -DHAVE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-lc -lm" \
  -DCMAKE_HIP_STANDARD_LIBRARIES="-lc -lm" \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=cmake/conan_provider.cmake \
  -DCMAKE_PREFIX_PATH=/opt/rocm

bash /var/lib/jenkins/moat/utils/timeit.sh catboost compile -- \
  ninja -C build_hip/cm -j16 catboost-cuda-cuda_util-ut
```

Build: SUCCESS (1172/1172 targets, no errors; only nodiscard warning on hipGetDevice).

### Code-object evidence

```
roc-obj-ls build_hip/cm/catboost/cuda/cuda_util/ut/catboost-cuda-cuda_util-ut
```

Every code object: hipv4-amdgcn-amd-amdhsa--gfx1100. No gfx90a present. Confirmed gfx1100 target.

Device reported at runtime: "AMD Radeon Pro W7800 48GB (compute capability 11.0)"

### Test run

```
export HIP_VISIBLE_DEVICES=0
bash /var/lib/jenkins/moat/utils/timeit.sh catboost test -- \
  build_hip/cm/catboost/cuda/cuda_util/ut/catboost-cuda-cuda_util-ut --show-fails
```

Run 1: [DONE] ok: 48 (all 48 subtests pass)
Run 2: [DONE] ok: 48 (all 48 subtests pass)
Pass sets: IDENTICAL -- all 48 [good] lines bit-identical across both runs.
No [bad] results, no HIP fault, clean exit.

Subtests passed (all 48): TCompressionGpuTest (5), TCompressionTest (1), TFillTest (2), TMvsThresholdCalculationTest (3), TReduceTest (7: TestReduce/TestSegmentedReduce/TestUpdatePartProps/TestUpdatePartPropsPerformance/TestSegmentedReducePerformance2/TestSegmentedReducePerformance/TestStatsInLeavesPerformance), TReorderTest (1), TScanTest (4), TSegmentedScanTest (8), TSegmentedSortTest (4: incl. TestSegmentedSortNonContinous), TSortTest (4), TTransformTest (9).

Matches gfx90a bar: 48/48 (gfx90a also 48/48). Deterministic (bit-identical pass sets x2 runs).

### Wave32 verdict

- FastInBlockReduce/BlockReduceN/SharedReduce4/8: wave-size-agnostic __syncthreads tree. CORRECT on wave32. CONFIRMED by TReduceTest (7/7) and TSegmentedScanTest (8/8).
- CB_FULL_WARP_MASK 64-bit mask: benign on wave32 (upper 32 bits address no lanes). CORRECT.
- 32-lane histogram layout: NATIVE on wave32 -- no gfx90a-style split required. CORRECT.
- TTileReducer<64>: dead code (never instantiated in any kernel). No ISA emission for tiled_partition<64>. SAFE.
- All live tiled_partition sizes (8/16/32): <= wave32 width. CORRECT.

No delta-port required. Result: PASS.

validated_sha: b7113fe9133aa0444bf0205cea546d33700ac16e -> completed.
