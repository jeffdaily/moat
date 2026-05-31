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
