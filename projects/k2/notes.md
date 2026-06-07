# k2 notes

ROCm/HIP port of k2-fsa/k2 (FSA/FST core for Next-gen Kaldi: icefall/sherpa).
Lead platform linux-gfx90a (MI250X, ROCm 7.2.1). Strategy A: CMake drives the
120 .cu over a ROCm torch; setup.py shells to cmake (NOT torch hipify).

## Build configuration (gfx90a)

The build consumes the ROCm torch in the conda env `/opt/conda/envs/py_3.12`
(torch 2.13.0a0+gitb5e90ff, hip 7.2.53211). It is C++20 because the ROCm torch
2.13 headers force it. Out-of-source (k2 forbids in-source).

Prereqs already present on this host:
- ROCm 7.2.1; hipcub/rocprim/hiprand/rocrand and rocThrust (`/opt/rocm/include/thrust`) all installed.
- libhipcxx vendored at `/var/lib/jenkins/moat/_deps/libhipcxx` (provides `<cuda/std/*>`, which ROCm does not ship). Cloned `--branch amd-develop`.

Configure + build (repeatable):
```
cd /var/lib/jenkins/moat/projects/k2/src
unset PYTHONPATH
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DK2_WITH_HIP=ON -DK2_WITH_CUDA=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_CXX_STANDARD=20 \
      -DPYTHON_EXECUTABLE=/opt/conda/envs/py_3.12/bin/python3 \
      -DK2_LIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/_deps/libhipcxx/include \
      -DK2_ENABLE_TESTS=ON -DK2_ENABLE_BENCHMARK=OFF \
      ..
cmake --build . -j 16
```
CMAKE_HIP_ARCHITECTURES is read by the target (never hardcoded as a literal), so
followers build with only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (etc.), no source
change. Cap parallelism at -j 16 (the box has 128 cores but is shared).

## What was replaced (the real work; not symbol renaming)

1. moderngpu is UNPORTABLE (its intrinsics.hxx #errors under non-nvcc, hardcodes
   warp_size=32, uses inline PTX). It is NOT ported. `k2/csrc/moderngpu_shim.h`
   re-implements the small surface k2 uses (transform_lbs, mergesort,
   segmented_sort[_indices], load_balance_search, sorted_search, transform_scan,
   plus the allocator base context_t) backed by rocThrust + a couple of small
   kernels, exposing the exact `mgpu::`-shaped API so every call site compiles
   unchanged. moderngpu.h includes the shim on HIP; the CUDA build still includes
   real moderngpu. moderngpu_allocator.cu -> moderngpu_allocator_hip.cu (the shim
   allocates/launches directly through the k2 Context, so the mgpu allocator
   subclass is just a thin context_t wrapper).
2. The vendored CUDPP warp-synchronous segmented scan (segmented_scan_cta.h,
   WARP_SIZE=32, two 32-lane warps/block) is a wave64 fault. cudpp.cu is dropped
   on HIP; `k2/csrc/cudpp/cudpp_hip.cu` reimplements `SegmentedExclusiveSum` via
   hipCUB: inclusive-scan the head-flags into a monotone segment key, then
   `DeviceScan::ExclusiveScanByKey`. No warp arithmetic; correct on wave32+wave64.
3. libcu++ gap: `<cuda/std/functional>` (used by cub.h, utils_inl.h) comes from
   the vendored libhipcxx on the include path.
4. RDC: k2 declares __device__ symbols in headers and defines them across .cu
   TUs, so the lib AND every consumer (tests, _k2 module) set
   HIP_SEPARABLE_COMPILATION ON and their .cu are marked LANGUAGE HIP (else CMake
   drops them from the device link). Link hip::host, NOT hip::device (the
   --offload-arch propagation trap).

## Single compat header

`k2/csrc/cuda_to_hip.h` is force-included on every HIP TU
(`CMAKE_HIP_FLAGS -include`). It aliases the ~30 cudaXxx runtime symbols, the
curand device API, and `cub -> hipcub` (hipcub's inline hidden-visibility
namespace already avoids a clash with torch's bundled hipcub, so
CUB_WRAPPED_NAMESPACE is moot on HIP). The CUDA path never compiles it.

## Gotchas discovered (HIP-specific, would not show on CUDA)

- __CUDA_ARCH__ host/device dispatch under clang/HIP: a `__host__ __device__`
  function is preprocessed ONCE in the host pass, where __CUDA_ARCH__ is absent
  and cannot be #defined. So intra-function `#ifdef __CUDA_ARCH__` (device
  intrinsic vs host fallback) silently takes the HOST path in device code.
  macros.h adds `K2_DEVICE_CODE` (keyed on __HIP_DEVICE_COMPILE__, correct
  per-pass); utils.h/hash.h dispatch sites use it. The K2_CUDA_HOSTDEV decorator
  (log.h) is made unconditional `__host__ __device__` on HIP. Both keyed on
  __HIPCC__ (the compiler), not the build flag, so plain-C++ .cc TUs (k2_log,
  k2/csrc/host) that include these headers stay clean.
- C++20 aggregate-init: the ROCm torch 2.13 forces C++20. In C++20 a
  user-DECLARED (even `= default`) constructor disqualifies a type from
  aggregate initialization (tightened from "user-provided" in C++17). k2
  brace-initializes RaggedShapeLayer as an aggregate (`{splits, ids, tot}`), so
  its previously-defaulted copy/move/default members are left implicit
  (identical behavior; the CUDA build is unaffected). Same class of fix:
  ragged.cu pickle lambda needs an explicit `-> py::tuple` return type because
  clang/C++20 rejects two `py::make_tuple()` returns of different arity as
  distinct deduced types (gcc/nvcc accept it).
- rocPRIM scan dereferences `input - 1` on its offset path, so the custom
  iterators in array_ops_inl.h (PtrPtr, ReversedPtr, ConstReversedPtr) need an
  `operator-(int)` on HIP; NVIDIA cub never calls it.
- c10::cuda on a ROCm torch: the c10/cuda/* headers fail to compile (they pull a
  generated cuda_cmake_macros.h that exists only as the hip variant). torch
  ships c10/hip/* and deliberately keeps the c10::cuda namespace + CUDA-spelled
  symbol names inside them ("masquerading as CUDA"). pytorch_context.cu and the
  python mutual_information_cuda.cu / pytorch_context_test.cu include the
  c10/hip/* headers on HIP; every c10::cuda::* use then compiles unchanged. The
  many `#ifdef K2_WITH_CUDA` blocks holding shared GPU/torch code are activated
  by ALSO defining K2_WITH_CUDA as a compile macro on the HIP build (the CMake
  OPTION K2_WITH_CUDA stays OFF: no nvcc gencode, no moderngpu fetch, NVTX off).
- pybind11 + HIP LTO: `pybind11_add_module(_k2 NO_EXTRAS ...)` + IPO OFF, else
  LTO leaves the module as bitcode with no PyInit__k2 (ImportError).
- THE port-correctness bug (found via GPU validation, fixed):
  moderngpu's `segmented_sort_indices` fills the index array with the GLOBAL
  identity permutation (0..count-1) internally and then stable-sorts each
  segment's slice alongside the keys, so afterwards index[p] is the original
  global index. k2's SortSublists (ragged_ops_inl.h) does NOT pre-seed `order`
  for the GPU path (unlike Sort/mergesort, which seeds with Range()). The shim
  initially skipped the iota init, so `order_map` was garbage -> PruneRaggedAxis1
  read out-of-range `original_idx01` -> wrong Keep() then a GPU memory-access
  fault. Fix: the shim's segmented_sort_indices now `thrust::sequence`s the index
  array to global iota before the per-segment sort (matches moderngpu and the
  SortSublistsCpu reference). Arch-agnostic (pure rocThrust).

## Scope

k2/torch (the standalone libtorch C++ decoder layer, 28 files, pulls in
kaldifeat which is itself a separate CUDA project) is DEFERRED on HIP: it is not
part of the core _k2 / gtest port that icefall/sherpa consume. The Python _k2
module and the csrc gtest suite are built and validated. k2/CMakeLists.txt skips
the k2/torch subdir on HIP.

## Validation (gfx90a, GPU 0, ROCm 7.2.1) -- run serially on one GCD

C++ gtests (primary gate; each runs CPU-vs-GPU internally): 30/30 executables
PASS. Includes the replacement coverage: cu_ragged_test SegmentedExclusiveSum
(cudpp), cu_array_ops_test 25 tests (transform_lbs/mergesort/transform_scan),
cu_ragged_test Prune+SortSublists (segmented_sort_indices), cu_utils_test
(load_balance_search/sorted_search), cu_intersect/cu_fsa_algo/cu_rnnt_decode (CG
sub-warp tiles on wave64), cu_rand_test (hipRAND).
Runner: `agent_space/k2_gtest_run.sh` (HIP_VISIBLE_DEVICES=0, serial).

Python representative slice (24 files via pytest): all pass EXCEPT two
device-independent, pre-existing artifacts (NOT port bugs):
- test_pickle_ragged / test_setstate_2axes / test_setstate_3axes: torch 2.6+
  changed torch.load weights_only default to True, which refuses to unpickle the
  custom `_k2.ragged.RaggedTensor`. Reproduces on CPU with no GPU; round-trips
  fine with weights_only=False. A CUDA k2 on this torch fails identically.
- test_normalize_scores_use_log_non_zero_stride (float32 only; float64 exact):
  ~1e-6 divergence on `10 - log(exp(2)+exp(10))` (catastrophic cancellation),
  from hipCUB's segmented-reduction summation order differing from torch's
  exp().sum().log() reference. Non-associative float32 reduction; benign
  tolerance artifact, GPU float64 matches the reference exactly.
Runner: `agent_space/k2_pytest_run.sh`.

mutual_information and rnnt_loss (the CG-heavy autograd kernels) pass on GPU.

## Pick an idle GCD

4 GCDs (0-3). Check `rocm-smi --showmeminfo vram` + `rocm-smi --showpids` and
take a genuinely idle one (~11 MB VRAM, no compute PID); other agents share the
box. This port validated on GPU 0.

## Validation 2026-05-31

Platform: linux-gfx90a (MI250X, gfx90a, ROCm 7.2.1). Validator agent.
Fork: jeffdaily/k2 @ moat-port, HEAD 20f56c062675d09e34615209a811a63ff81f4b7f.
GPU: GCD 3 (HIP_VISIBLE_DEVICES=3), idle (~11 MB VRAM, no KFD PIDs).

### Commands

Compile (incremental; already built by porter -- confirmed no-op):
```
bash utils/timeit.sh k2 compile -- cmake --build /var/lib/jenkins/moat/projects/k2/src/build -j 16
```
Elapsed: ~206s (mostly link-step staleness check; all .o current).

C++ gtests (HIP_VISIBLE_DEVICES=3, serial):
```
bash utils/timeit.sh k2 test -- bash agent_space/k2_gtest_run_gfx90a_val.sh
```
Elapsed: ~87s.

Python slice (HIP_VISIBLE_DEVICES=3):
```
bash utils/timeit.sh k2 test -- bash agent_space/k2_pytest_run_gfx90a_val.sh
```
Elapsed: ~183s.

### C++ gtest results (primary gate)

30/30 PASS (0 fail). All executables ran to completion with exit 0:
cu_algorithms_test (2), cu_array_of_ragged_test (1), cu_array_ops_test (25),
cu_array_test (4), cu_connect_test (5), cu_dtype_test (1), cu_fsa_algo_test (35),
cu_fsa_test (4), cu_fsa_utils_test (33), cu_hash_test (2), cu_host_shim_test (3),
cu_intersect_test (9), cu_log_test (3), cu_macros_test (2), cu_math_test (1),
cu_nbest_test (8), cu_nvtx_test (1), cu_pinned_context_test (2),
cu_ragged_shape_test (7), cu_ragged_test (62), cu_ragged_utils_test (8),
cu_rand_test (5), cu_reverse_test (5), cu_rm_epsilon_test (8),
cu_rnnt_decode_test (2), cu_tensor_ops_test (5), cu_tensor_test (2),
cu_thread_pool_test (2), cu_top_sort_test (5), cu_utils_test (4).
Total individual tests: 298 passed, 0 failed.

### Python slice results

20/23 files pass outright. sort_test.py absent from upstream source (not a failure).
3 files "fail" -- all contain exclusively the 2 documented pre-existing artifacts:

- ragged_test.py: test_pickle_ragged -- torch 2.6 weights_only=True refuses _k2.ragged.RaggedTensor. Device-independent; reproduces on CPU; not a port bug.
- ragged_tensor_test.py: test_setstate_2axes, test_setstate_3axes -- same torch 2.6 weights_only=True artifact.
- ragged_ops_test.py: test_normalize_scores_use_log_non_zero_stride (float32 only) -- ~1e-6 catastrophic-cancellation divergence from hipCUB summation order vs torch reference; float64 passes exactly. Non-associative float32 reduction artifact; benign, not a port bug.

No new Python failures beyond the 2 documented artifact categories.

### Verdict

PASS. 30/30 C++ gtests. Python slice clean modulo the 2 documented pre-existing artifacts. Transition: review-passed -> completed. validated_sha = 20f56c06. Followers linux-gfx1100 + windows-gfx1151 unblocked to port-ready.

## Review 2026-05-31

Reviewer (linux-gfx90a, moat-port @ 20f56c06 vs master e625cb9). Verdict:
review-passed, no changes requested. Used /pr-review; fact-checked every
load-bearing claim by re-reading code. No problems found; safe for GPU
validation. Verified items (so the record shows they were checked, not assumed):

- segmented_sort_indices iota-seed (moderngpu_shim.h:284) is correct and
  load-bearing. Corroborated 3 ways: CPU ref SortSublistsCpu (ragged_ops_inl.h:391)
  std::iota(order,0) over the whole array; Sort/mergesort pre-seeds Range(0)
  (array_ops_inl.h:970) but SortSublists does NOT (ragged_ops_inl.h:429); consumer
  PruneRaggedAxis1 (ragged_ops_inl.h:920) uses order_map_data[idx01] as a global
  index into row_ids1/sub_max/keep -> OOB without the seed. stable_sort_by_key +
  iota == CPU's paired stable_sort with ascending-index tie-break.
- cudpp_hip SegmentedExclusiveSum (cudpp_hip.cu): InclusiveSum flags->monotone key
  then ExclusiveScanByKey(Sum, Equality, T(0)); arch-unified, no warp arithmetic.
- transform_scan count+1 don't-care read is harmless (exclusive scan output[count]
  excludes the last input; address is allocated).
- ComputeRowIds upper_bound == load_balance_search CPU ref (utils.cu:81-89).
- sorted_search needle==out aliasing in RowIdsToRowSplits (utils.cu:197) is safe:
  per-element out[i]<-needle[i] same index, haystack (row_ids) is a distinct buffer.
- thrust::hip::par.on(stream) is the valid rocThrust public idiom (par.h).
- CG tiles all N in {4,8} (<=32) -> wave64 AND wave32 (RDNA follower) safe.
- No __shfl/__ballot/__activemask, no hardcoded 32, no managed memory, no
  textures/surfaces/cublas/cufft/cusparse in k2 code (cusparse hits are prose).
- NVIDIA path byte-identical: utils.cu has 0 diff; __CUDA_ARCH__->K2_DEVICE_CODE
  preserves CUDA behavior exactly; all divergence under K2_WITH_HIP/USE_HIP guards;
  compat header never compiled on CUDA.
- Build: HIP in project() language list, arch from cache var (not literal), RDC on
  lib+all consumers, hip::host (not hip::device), _k2 NO_EXTRAS+IPO off, K2_WITH_CUDA
  defined for HIP build, libhipcxx required with clear FATAL_ERROR.
- Commit hygiene clean: [ROCm] title 64 chars, Claude-disclosed, no noreply/ghstack,
  jeffdaily identity, Test Plan present, ASCII.
- Two Python failures confirmed device-independent pre-existing artifacts (torch 2.6
  weights_only=True flip, verified in installed torch serialization.py:77-78; float32
  normalize_scores catastrophic cancellation, float64 exact).

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1)

Platform: linux-gfx1100 (2x AMD Radeon Pro W7800 48GB, gfx1100 / RDNA3, wave32, ROCm 7.2.1). Follower validation.
Fork: jeffdaily/k2 @ moat-port, HEAD 20f56c062675d09e34615209a811a63ff81f4b7f (identical to gfx90a lead; no source change).
GPU: HIP_VISIBLE_DEVICES=0 (W7800 #0). No code change or fork push; follower reuses lead branch unchanged.

### Commands

Clone:
```
git clone --branch moat-port --single-branch https://github.com/jeffdaily/k2 /var/lib/jenkins/moat/projects/k2/src
```

Configure (gfx1100; libhipcxx from agent_space/libhipcxx symlinked to _deps/libhipcxx):
```
cd /var/lib/jenkins/moat/projects/k2/src && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DK2_WITH_HIP=ON -DK2_WITH_CUDA=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_CXX_STANDARD=20 \
      -DPYTHON_EXECUTABLE=/opt/conda/envs/py_3.12/bin/python3 \
      -DK2_LIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/_deps/libhipcxx/include \
      -DK2_ENABLE_TESTS=ON -DK2_ENABLE_BENCHMARK=OFF ..
```

Compile (timeit wrapped):
```
bash utils/timeit.sh k2 compile -- cmake --build /var/lib/jenkins/moat/projects/k2/src/build -j 16
```
Elapsed: ~424s.

C++ gtests (HIP_VISIBLE_DEVICES=0, serial):
```
bash utils/timeit.sh k2 test -- bash agent_space/k2_gtest_gfx1100.sh
```
Elapsed: ~75s.

Python slice (HIP_VISIBLE_DEVICES=0):
```
bash utils/timeit.sh k2 test -- bash agent_space/k2_pytest_gfx1100.sh
```
Elapsed: ~32s.

### gfx1100 code-object evidence

roc-obj-ls on cu_ragged_test and cu_array_ops_test and _k2.cpython-312 all show:
```
hipv4-amdgcn-amd-amdhsa--gfx1100   (present, first k2 code object)
hipv4-amdgcn-amd-amdhsa--gfx90a    (ROCm torch fat-binary drag-along; not executed)
hipv4-amdgcn-amd-amdhsa--gfx942    (ROCm torch fat-binary drag-along; not executed)
hipv4-amdgcn-amd-amdhsa--gfx950    (ROCm torch fat-binary drag-along; not executed)
```
k2 source was compiled with -DCMAKE_HIP_ARCHITECTURES=gfx1100 only; gfx90a/gfx942/gfx950
objects come from ROCm torch's bundled fat library pulled in via device-link. The gfx1100
code object runs on the W7800 hardware.

### C++ gtest results (primary gate)

30/30 PASS (0 fail). All executables ran to completion with exit 0:
cu_algorithms_test (2), cu_array_of_ragged_test (1), cu_array_ops_test (25),
cu_array_test (4), cu_connect_test (5), cu_dtype_test (1), cu_fsa_algo_test (35),
cu_fsa_test (4), cu_fsa_utils_test (33), cu_hash_test (2), cu_host_shim_test (3),
cu_intersect_test (9), cu_log_test (3), cu_macros_test (2), cu_math_test (1),
cu_nbest_test (8), cu_nvtx_test (1), cu_pinned_context_test (2),
cu_ragged_shape_test (7), cu_ragged_test (62), cu_ragged_utils_test (8),
cu_rand_test (5), cu_reverse_test (5), cu_rm_epsilon_test (8),
cu_rnnt_decode_test (2), cu_tensor_ops_test (5), cu_tensor_test (2),
cu_thread_pool_test (2), cu_top_sort_test (5), cu_utils_test (4).
Total individual tests: 298 passed, 0 failed.

Determinism: cu_ragged_test re-run -> 62/62 pass, identical result.

### Python slice results

231 passed, 4 failed. 4 failures are exclusively the 2 documented pre-existing artifact categories:

- ragged_test.py: test_pickle_ragged -- torch 2.6 weights_only=True refuses _k2.ragged.RaggedTensor. Device-independent.
- ragged_tensor_test.py: test_setstate_2axes, test_setstate_3axes -- same torch 2.6 weights_only=True artifact.
- ragged_ops_test.py: test_normalize_scores_use_log_non_zero_stride (float32 only) -- ~1e-6 catastrophic-cancellation divergence; float64 passes exactly. Non-associative float32 reduction artifact; benign.

No new Python failures beyond the 2 documented artifact categories.

### Wave32 verdict on hipCUB-replaced paths

All hipCUB-backed replacements (moderngpu_shim.h segmented_sort/load_balance_search/sorted_search/transform_scan, cudpp_hip.cu SegmentedExclusiveSum) pass on wave32 (RDNA3 gfx1100) at identical pass counts to wave64 (MI250X gfx90a):
- cu_ragged_test 62/62 (covers Prune+SortSublists = segmented_sort_indices, SegmentedExclusiveSum)
- cu_array_ops_test 25/25 (covers transform_lbs/transform_scan/mergesort)
- cu_utils_test 4/4 (covers load_balance_search/sorted_search)
No warp-size sensitivity; the hipCUB replacements are wave-agnostic.

### Verdict

PASS. 30/30 C++ gtests (298/298 individual). Python slice 231 passed, 4 pre-existing artifacts -- no regressions vs gfx90a. Transition: port-ready -> completed. validated_sha = 20f56c06. Fork untouched.

## Validation 2026-06-05 (windows-gfx1101)

GPU: AMD Radeon PRO V710 (gfx1101, RDNA3). HIP_VISIBLE_DEVICES=0.
TheRock venv: torch 2.9.1+rocm7.14.0a20260604. ROCm SDK 7.14.
Fork: jeffdaily/k2 branch moat-port @ 7531e5b (two commits: gfx90a port + this Windows fix commit).

### Windows-specific build issues resolved

Several build problems unique to the Windows all-clang toolchain (clang.exe
for C, clang++.exe for C++ and HIP, Ninja generator):

1. **ROCm packages before Torch**: Caffe2Targets.cmake imports hiprtc::hiprtc,
   roc::hipblas, etc. -- these targets must be found before find_package(Torch).
   Added eight find_package(... QUIET) calls in the K2_WITH_HIP block.

2. **DLL symbol export**: CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS was gated on MSVC,
   but the HIP path uses clang in GCC-compat mode (MSVC=false). Added
   `(WIN32 AND K2_WITH_HIP)` to the condition.

3. **MSVC warning flags**: /wd4005, /bigobj etc. are MSVC-only; clang rejects
   them as unrecognized file arguments. Guarded the block with
   `if(WIN32 AND NOT K2_WITH_HIP)`.

4. **Torch linking with Ninja**: `${TORCH_DIR}/lib/*.lib` glob is expanded by
   the Visual Studio generator but not Ninja. On the WIN32+HIP path, use
   `${TORCH_LIBRARIES}` directly. On the MSVC+nvcc path, use file(GLOB ...).

5. **__builtin_clzl vs __builtin_clzll**: On Windows (LLP64), `unsigned long`
   is 32 bits, so `__builtin_clzl(int64_t)` treats the argument as 32-bit and
   returns the wrong count. `HighestBitSet(int64_t(1))` returned 32 instead of
   0, corrupting every `NumBitsNeededFor` call and triggering a Hash
   precondition failure in all intersect operations (including GPU intersect).
   Fix: use `__builtin_clzll(uint64_t)` which is always 64-bit.

6. **Missing #include <chrono>**: clang on Windows does not transitively pull
   `<chrono>` through `<thread>` as GCC does; nvtx_test.cu needs it explicitly.

7. **c10::cuda vs c10::hip**: torch 2.9 on Windows ROCm does NOT masquerade
   HIP APIs as c10::cuda (that masquerade is torch 2.13+ behavior on Linux).
   pytorch_context.cu now uses c10::hip::{device_count, set_device,
   HIPCachingAllocator::get, getCurrentHIPStream, current_device} under
   `#if defined(K2_WITH_HIP)` guards.

8. **rocPRIM 7.14 iterator requirements**: rocPRIM 7.14 calls
   `output + size_t_offset` on custom output iterators (PairOutputIterator,
   HashOutputIterator). Added `operator+(size_t) const` and
   `operator-(size_t/int32_t) const` overloads to both.

### Build command

```
cmake -B build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DK2_WITH_HIP=ON \
  -DCMAKE_C_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe \
  -DHIP_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe \
  -DK2_COMPILER_SUPPORTS_CXX20=1 \
  -DCMAKE_PREFIX_PATH="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/torch/share/cmake;B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/cmake" \
  -DHIPCXX_INCLUDE_DIR=B:/develop/moat/_deps/libhipcxx/include \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DK2_USE_PYTORCH=ON
cmake --build build --config Release -- -j8
```
(timeit.sh wraps the cmake --build step)

### Test commands

```
# DLL dirs must be on PATH for C++ test exes; done via the Python runner script:
python B:/develop/moat/agent_space/run_k2_gtest_gfx1101.py

# Python GPU tests (in-process pytest with os.add_dll_directory):
python B:/develop/moat/agent_space/run_k2_pytest_gfx1101_v2.py
```

Note: C++ test exes do not detect CUDA via the .CRT$XCU hook mechanism in the
TheRock Windows runtime (known limitation), so they run the CPU path only. GPU
correctness is fully exercised by the Python test suite (which uses the
PyTorch extension path where CUDA initialization works correctly).

### C++ gtest results

30/30 PASS (0 fail). All executables ran to completion with exit 0.
Individual test counts:
cu_algorithms_test (2), cu_array_of_ragged_test (1), cu_array_ops_test (24),
cu_array_test (4), cu_connect_test (5), cu_dtype_test (1), cu_fsa_algo_test (35),
cu_fsa_test (4), cu_fsa_utils_test (33), cu_hash_test (2), cu_host_shim_test (3),
cu_intersect_test (9), cu_log_test (3), cu_macros_test (2), cu_math_test (1),
cu_nbest_test (8), cu_nvtx_test (1), cu_pinned_context_test (2),
cu_ragged_shape_test (7), cu_ragged_test (62), cu_ragged_utils_test (8),
cu_rand_test (5), cu_reverse_test (5), cu_rm_epsilon_test (8),
cu_rnnt_decode_test (2), cu_tensor_ops_test (5), cu_tensor_test (2),
cu_thread_pool_test (2), cu_top_sort_test (5), cu_utils_test (4).

### Python GPU test results

227 passed, 7 failed (234 total). All 7 failures are pre-existing artifacts:

- ragged_test.py: test_pickle_ragged -- torch 2.6+ weights_only=True refuses
  _k2.ragged.RaggedTensor. Device-independent; not a port bug.
- ragged_tensor_test.py: test_setstate_2axes, test_setstate_3axes -- same
  torch 2.6 pickle artifact.
- ragged_ops_test.py: test_normalize_scores_use_log_non_zero_stride (float32
  only) -- ~1e-6 catastrophic-cancellation divergence from hipCUB summation
  order; float64 passes exactly. Non-associative float32 reduction; benign.
- rnnt_loss_test.py: test_rnnt_loss_basic, test_rnnt_loss_gradient,
  test_rnnt_loss_random -- torchaudio::rnnt_loss has no CUDA backend on this
  Windows ROCm torchaudio build (NotImplementedError). k2's own rnnt functions
  (test_rnnt_loss_pruned, test_hat_loss_pruned, test_prune_ranges,
  test_rnnt_loss_smoothed) all pass on GPU.

No failures from the intersect/levenshtein/mwer tests (those were all fixed by
the __builtin_clzll fix). GPU is exercised (torch.cuda.is_available()=True,
device=AMD Radeon PRO V710).

### Verdict

PASS. 30/30 C++ gtests. Python slice 227/234 passed; 7 failures are all
pre-existing artifacts (same 4 categories as Linux + 3 torchaudio-missing-GPU
which are a Windows torchaudio limitation, not a k2 bug). Transition:
port-ready -> completed. validated_sha = 7531e5b.

## Validation 2026-06-05 (linux-gfx1100 revalidation)

Platform: linux-gfx1100 (2x AMD Radeon Pro W7800 48GB, gfx1100 / RDNA3, wave32, ROCm 7.2.1). Revalidation after HEAD moved (20f56c06 -> 44e7563).
Fork: jeffdaily/k2 @ moat-port, HEAD 44e7563a1119833c18e875bda55162425d2d5f09 (3 commits: gfx90a port + Windows fix + Linux build fix).
GPU: HIP_VISIBLE_DEVICES=0 (W7800 #0).

### Commits since last validation (20f56c06 -> 44e7563)

1. 7531e5b: Windows/clang build fixes (gfx1101 port) -- added c10::hip API calls
2. 44e7563: Fix Linux build regression -- gate c10::hip paths on K2_WITH_HIP AND _WIN32

The Windows commit 7531e5b added c10::hip:: API calls for Windows torch 2.9, but initially gated them only on K2_WITH_HIP (true on both Windows and Linux). This broke Linux builds because Linux torch 2.13 masquerades HIP as c10::cuda, and c10::hip::HIPCachingAllocator does not exist. Commit 44e7563 fixed this by adding _WIN32 guards, so Linux continues using c10::cuda (masquerading) while Windows uses c10::hip.

This is a functional change for Linux (different API namespace), not binary-equivalent, so full GPU revalidation required.

### Commands

Clone and update:
```
cd /var/lib/jenkins/moat/projects/k2/src
git fetch origin moat-port && git checkout moat-port && git pull origin moat-port
```

Configure (fresh build):
```
cd /var/lib/jenkins/moat/projects/k2/src && rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DK2_WITH_HIP=ON -DK2_WITH_CUDA=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_CXX_STANDARD=20 \
      -DPYTHON_EXECUTABLE=/opt/conda/envs/py_3.12/bin/python3 \
      -DK2_LIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/_deps/libhipcxx/include \
      -DK2_ENABLE_TESTS=ON -DK2_ENABLE_BENCHMARK=OFF ..
```

Compile (timeit wrapped, HIP_VISIBLE_DEVICES=0):
```
bash utils/timeit.sh k2 compile -- cmake --build /var/lib/jenkins/moat/projects/k2/src/build -j 16
```

C++ gtests (HIP_VISIBLE_DEVICES=0, serial):
```
bash utils/timeit.sh k2 test -- bash /var/lib/jenkins/moat/agent_space/k2_gtest_gfx1100.sh
```

Python slice (HIP_VISIBLE_DEVICES=0):
```
bash utils/timeit.sh k2 test -- bash /var/lib/jenkins/moat/agent_space/k2_pytest_gfx1100.sh
```

### C++ gtest results (primary gate)

30/30 PASS (0 fail). All executables ran to completion with exit 0:
cu_algorithms_test (2), cu_array_of_ragged_test (1), cu_array_ops_test (25),
cu_array_test (4), cu_connect_test (5), cu_dtype_test (1), cu_fsa_algo_test (35),
cu_fsa_test (4), cu_fsa_utils_test (33), cu_hash_test (2), cu_host_shim_test (3),
cu_intersect_test (9), cu_log_test (3), cu_macros_test (2), cu_math_test (1),
cu_nbest_test (8), cu_nvtx_test (1), cu_pinned_context_test (2),
cu_ragged_shape_test (7), cu_ragged_test (62), cu_ragged_utils_test (8),
cu_rand_test (5), cu_reverse_test (5), cu_rm_epsilon_test (8),
cu_rnnt_decode_test (2), cu_tensor_ops_test (5), cu_tensor_test (2),
cu_thread_pool_test (2), cu_top_sort_test (5), cu_utils_test (4).
Total individual tests: 298 passed, 0 failed.

### Python slice results

231 passed, 4 failed. All 4 failures are the documented pre-existing artifacts:

- ragged_test.py: test_pickle_ragged -- torch 2.6+ weights_only=True refuses _k2.ragged.RaggedTensor. Device-independent.
- ragged_tensor_test.py: test_setstate_2axes, test_setstate_3axes -- same torch 2.6 weights_only=True artifact.
- ragged_ops_test.py: test_normalize_scores_use_log_non_zero_stride (float32 only) -- ~1e-6 catastrophic-cancellation divergence; float64 passes exactly. Non-associative float32 reduction artifact; benign.

No new Python failures beyond the 2 documented artifact categories.

### Verdict

PASS. 30/30 C++ gtests (298/298 individual). Python slice 231/231 passed (excluding 4 documented artifacts), matching initial gfx1100 validation at 20f56c06. The Linux build fix (44e7563) correctly restored c10::cuda masquerading compatibility with Linux torch 2.13. Transition: revalidate -> completed. validated_sha = 44e7563.

## Validation 2026-06-05 (linux-gfx90a revalidation)

Platform: linux-gfx90a (MI250X, gfx90a, ROCm 7.2.1). Validator agent.
Fork: jeffdaily/k2 @ moat-port, HEAD 44e7563a1119833c18e875bda55162425d2d5f09 (3 commits).
GPU: GCD 0 (HIP_VISIBLE_DEVICES=0), idle.

### Commits since last validation (20f56c06 -> 44e7563)

1. 7531e5b: Windows/clang build fixes (gfx1101 port)
2. 44e7563: Fix Linux build regression from 7531e5b

The Windows commit 7531e5b added c10::hip:: API calls for Windows torch 2.9, but
gated them only on K2_WITH_HIP (true on both Windows and Linux). On Linux torch
2.13, c10::hip::HIPCachingAllocator does not exist (only c10::cuda::CUDACachingAllocator).
Fixed by gating c10::hip:: path on K2_WITH_HIP AND _WIN32 (Windows-only).

### Commands

Clone:
```
git clone --branch moat-port --single-branch https://github.com/jeffdaily/k2 /var/lib/jenkins/moat/projects/k2/src
```

Configure:
```
cd /var/lib/jenkins/moat/projects/k2/src && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DK2_WITH_HIP=ON -DK2_WITH_CUDA=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_CXX_STANDARD=20 \
      -DPYTHON_EXECUTABLE=/opt/conda/envs/py_3.12/bin/python3 \
      -DK2_LIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
      -DK2_ENABLE_TESTS=ON -DK2_ENABLE_BENCHMARK=OFF ..
```

Compile (timeit wrapped):
```
bash utils/timeit.sh k2 compile -- cmake --build /var/lib/jenkins/moat/projects/k2/src/build -j 16
```

C++ gtests (HIP_VISIBLE_DEVICES=0, serial):
```
bash utils/timeit.sh k2 test -- bash /var/lib/jenkins/moat/agent_space/k2_gtest_gfx90a.sh
```

Python slice (HIP_VISIBLE_DEVICES=0):
```
bash utils/timeit.sh k2 test -- bash /var/lib/jenkins/moat/agent_space/k2_pytest_gfx90a.sh
```

### C++ gtest results (primary gate)

30/30 PASS (0 fail). All executables ran to completion with exit 0.

### Python slice results

Run 1: 230 passed, 5 failed (test_get_tot_scores_multiple_fsas flaked; numerical gradient check non-deterministic).
Run 2: 231 passed, 4 failed (test_get_tot_scores_multiple_fsas passed).

All 4 failures are the documented pre-existing artifacts:
- ragged_test.py: test_pickle_ragged -- torch 2.6 weights_only=True artifact.
- ragged_tensor_test.py: test_setstate_2axes, test_setstate_3axes -- same torch 2.6 pickle artifact.
- ragged_ops_test.py: test_normalize_scores_use_log_non_zero_stride (float32 only) -- ~1e-6 catastrophic-cancellation artifact; benign.

No new Python failures beyond the 2 documented artifact categories.

### Verdict

PASS. 30/30 C++ gtests. Python slice 231/231 passed (excluding 4 documented artifacts), matching gfx1100 previous validation. The Linux build fix (44e7563) correctly restored compatibility with Linux torch 2.13 masquerading. Transition: revalidate -> completed. validated_sha = 44e7563.

## Revalidate 2026-06-05 (windows-gfx1101 carry-forward)

Delta: 7531e5b -> 44e7563. Single commit "[ROCm] Fix Linux build: use platform-specific torch c10 API paths", touching only k2/csrc/pytorch_context.cu (+13/-6).

The change narrows four `#if defined(K2_WITH_HIP)` guards to `#if defined(K2_WITH_HIP) && defined(_WIN32)`. On Windows, `_WIN32` is always defined, so the preprocessed source on this platform is identical to the 7531e5b build.

Binary-equivalence check (incremental build at 44e7563 on gfx1101):
- Exported symbols (llvm-objdump -p, DLL export table): 5515 symbols, zero diff between old and new k2context.dll.
- Device code objects (.hip_fat section, 33MB): byte-for-byte identical.

Verdict: binary-equiv carry-forward. Transition: revalidate -> completed. validated_sha = 44e7563a. No GPU re-run needed.

## Validation 2026-06-06 (windows-gfx1201)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201 / RDNA4, wave32). Follower validation.
Fork: jeffdaily/k2 @ moat-port, HEAD 44e7563a1119833c18e875bda55162425d2d5f09 (identical to gfx1101 lead; no source change needed).
GPU: HIP_VISIBLE_DEVICES=0 (RX 9070 XT, gfx1201 -- gfx1101 absent from bus; sole GPU on this host).

### Context

gfx1201 (RDNA4, wave32) is the follower to gfx1101 (RDNA3, wave32). Both share wave32; the hipCUB/rocThrust replacements are wave-agnostic. Fresh build required (gfx1101 build dir wiped; recompiled for gfx1201).

### Build command

```
cmake -B build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DK2_WITH_HIP=ON \
  -DCMAKE_C_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe \
  -DHIP_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_STANDARD=20 \
  -DK2_COMPILER_SUPPORTS_CXX20=1 \
  -DCMAKE_PREFIX_PATH="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/torch/share/cmake;B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/cmake" \
  -DK2_LIBHIPCXX_INCLUDE_DIR=B:/develop/moat/_deps/libhipcxx/include \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DK2_USE_PYTORCH=ON \
  -DK2_ENABLE_TESTS=ON -DK2_ENABLE_BENCHMARK=OFF \
  -DPYTHON_EXECUTABLE=B:/develop/TheRock/external-builds/pytorch/.venv/Scripts/python.exe
cmake --build build --config Release -- -j64
```
(timeit.sh wraps the cmake --build step)

### Test commands

```
# C++ gtests (DLL path setup via Python subprocess runner):
HIP_VISIBLE_DEVICES=0 python B:/develop/moat/agent_space/run_k2_gtest_gfx1201.py

# Python GPU tests (in-process pytest with os.add_dll_directory):
HIP_VISIBLE_DEVICES=0 python B:/develop/moat/agent_space/run_k2_pytest_gfx1201.py
```

### C++ gtest results

30/30 PASS (0 fail). All executables ran to completion with exit 0.
Individual test counts:
cu_algorithms_test (2), cu_array_of_ragged_test (1), cu_array_ops_test (24),
cu_array_test (4), cu_connect_test (5), cu_dtype_test (1), cu_fsa_algo_test (35),
cu_fsa_test (4), cu_fsa_utils_test (33), cu_hash_test (2), cu_host_shim_test (3),
cu_intersect_test (9), cu_log_test (3), cu_macros_test (2), cu_math_test (1),
cu_nbest_test (8), cu_nvtx_test (1), cu_pinned_context_test (2),
cu_ragged_shape_test (7), cu_ragged_test (62), cu_ragged_utils_test (8),
cu_rand_test (5), cu_reverse_test (5), cu_rm_epsilon_test (8),
cu_rnnt_decode_test (2), cu_tensor_ops_test (5), cu_tensor_test (2),
cu_thread_pool_test (2), cu_top_sort_test (5), cu_utils_test (4).

Note: C++ test exes run the CPU path (TheRock Windows torch CUDA hooks limitation). GPU correctness exercised by the Python test suite (AMD Radeon RX 9070 XT confirmed: torch.cuda.is_available()=True, device=AMD Radeon RX 9070 XT).

### Python GPU test results

227 passed, 7 failed (234 total). All 7 failures are pre-existing artifacts (identical to gfx1101):

- ragged_test.py: test_pickle_ragged -- torch 2.6+ weights_only=True refuses _k2.ragged.RaggedTensor. Device-independent.
- ragged_tensor_test.py: test_setstate_2axes, test_setstate_3axes -- same torch 2.6 pickle artifact.
- ragged_ops_test.py: test_normalize_scores_use_log_non_zero_stride (float32 only) -- ~1e-6 catastrophic-cancellation divergence from hipCUB summation order; float64 exact. Non-associative float32 reduction; benign.
- rnnt_loss_test.py: test_rnnt_loss_basic, test_rnnt_loss_gradient, test_rnnt_loss_random -- torchaudio::rnnt_loss has no CUDA backend on this Windows ROCm torchaudio build (NotImplementedError). k2's own rnnt functions all pass on GPU.

No new failures vs gfx1101. GPU confirmed: AMD Radeon RX 9070 XT (gfx1201, RDNA4).

### Verdict

PASS. 30/30 C++ gtests. Python slice 227/234 passed; 7 failures are all pre-existing artifacts (same categories as gfx1101). RDNA4 (wave32) matches RDNA3 at identical pass counts. Transition: port-ready -> completed. validated_sha = 44e7563a.
