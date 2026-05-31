# LMCache notes

Port type: finish/validate an existing upstream HIP path (Strategy B, torch
hipify). Lead platform linux-gfx90a (MI250X, CDNA2, ROCm 7.2.1). Upstream
LMCache/LMCache @ dev (base 7eb57ad). Fork jeffdaily/LMCache, branch
moat-port, HEAD e1d94420.

## What the port needed

Exactly one source change: in `setup.py` `rocm_extension()`, bump the `cxx`
flag from `-std=c++17` to `-std=c++20`.

Root cause: the BUILD_WITH_HIP=1 path compiles the plain `.cpp` sources
(pybind/mem_alloc/utils/recorders) with the `cxx` flags = `-std=c++17`, while
torch's ROCm BuildExtension compiles the `.hip` sources at `-std=c++20`. The
`.cpp` TUs include the same `<torch/all.h>` headers; on torch 2.13 those use
a C++20 `requires` constrained template (`c10/core/TensorImpl.h:2516`), so the
C++17 `.cpp` compile fails with `error: unknown type name 'requires'`. Upstream
never hit this because their ROCm CI is gfx942/gfx950 on an older torch whose
headers were C++17-clean (and their NVIDIA CI does not exercise the HIP path).

No kernel / wave64 / mem_alloc change was needed. The kernels coordinate only
via `__syncthreads()` (no warp intrinsics, atomics, or textures), so wave64 is
a no-op here. `hipHostRegister` / `hipHostGetDevicePointer` work as-is on
gfx90a for regular-pinned, NUMA-bound, and shm memory (all GPU-validated).

## Arch surface (env-driven, no source literal)

`rocm_extension()` does NOT pin an arch; torch's `_get_rocm_arch_flags()`
honors `PYTORCH_ROCM_ARCH` (verified: with `PYTORCH_ROCM_ARCH=gfx90a` the
compile gets `--offload-arch=gfx90a` only). The only `gfx942,gfx950` literals
in the repo are the ARG defaults in `docker/Dockerfile.rocm*`; they are not on
the build path used here and were left untouched (editing committed Dockerfile
arch lists would only churn the fork HEAD for followers). Followers
(gfx1100/gfx1151) build the same commit with just `PYTORCH_ROCM_ARCH=<arch>`,
no source edit.

## Build (gfx90a)

The project's own `hipify_wrapper()` runs first and rewrites `csrc/` in place
(`*.cu -> *.hip`, `pybind.cpp -> pybind_hip.cpp`, `mem_alloc.cpp ->
mem_alloc_hip.cpp`, `utils.cpp -> utils_hip.cpp`, `*.cuh -> *_hip.cuh`,
recorders -> `*_hip.*`). These match `rocm_extension()`'s `hip_sources` list
exactly; the apparent "missing .hip files" in the pristine tree are
build-generated, not a bug. Hipify is clean: 0 unmapped cuda* symbols, 14
kernel-launch rewrites. All generated files are already gitignored
(`.gitignore` `/csrc/*.hip`, `/csrc/*_hip.*`, `/csrc_hip`), so the fork commit
is just `setup.py`.

Repeatable build script lives in the fork clone at
`projects/LMCache/src/build_rocm_gfx90a.sh` (NOT committed to the fork; it is a
MOAT-side helper). It cleans the stale hipified mirror, then:

    BUILD_WITH_HIP=1 CXX=hipcc PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 \
      python3 setup.py build_ext --inplace

Note: `--no-build-isolation` is a pip flag, NOT a setup.py build_ext flag (the
plan listed both forms; only the pip form takes it). Build deps:
`pip install ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" setuptools_scm wheel`.

Builds `lmcache.c_ops` (HIP GPU ext) plus the pure-CPP `native_storage_ops`,
`lmcache_redis`, `lmcache_fs` in-place. Re-hipify on any `.cu`/`.cpp` edit
(`rm -f csrc/*.hip csrc/*_hip.* && rm -rf csrc_hip` before rebuild) -- the
Strategy-B stale-mirror gotcha.

## GPU validation (HIP_VISIBLE_DEVICES=<idle GCD>, gfx90a)

This host has 4 GCDs (rocm-smi GPU[0..3], all gfx90a MI250X). The task
suggested device 7 / 1/3/4/5/6 but only 0-3 exist here; used an idle one
(e.g. 3). Run serially on one GCD (single-GPU contention causes transient
flakes if parallelized).

Test deps installed (lightweight runtime, not full vLLM/SGLang):
`prometheus_client msgspec sortedcontainers safetensors blake3 py-cpuinfo
psutil pyyaml aiofiles aiofile aiohttp httpx pyzmq redis huggingface_hub
transformers pytest-benchmark`.

Results (all PASS; skips are env-gated, not failures):
- tests/v1/test_mem_kernels.py            56 passed   (mem_kernels.cu layout transforms, H2D/D2H, cudaHostGetDevicePointer)
- tests/v1/test_mp_mem_kernels.py         40 passed   (mp_mem_kernels.cu block transfer)
- tests/v1/test_c_ops_fallback_parity.py  \
  tests/v1/test_python_ops_fallback.py    105 passed, 1 skip  (c_ops == pure-Python ref for ALL ops, incl. encode_fast_new / decode_fast_new / decode_fast_prefsum / calculate_cdf / rotary_embedding_k_fused)
- tests/v1/test_memory_management.py      47 passed, 3 skip   (mem_alloc.cpp; 3 hugepage skips = host has no vm.nr_hugepages)
- tests/v1/test_gpu_connector.py          ~32 passed, 1 skip  (see flake note below)
- tests/v1/mp_observability/test_event_recorder.py \
  tests/v1/multiprocess/test_completion_recorder.py  17 passed (hipLaunchHostFunc host-callback path)

CacheGen arithmetic-coder determinism + round-trip (agent_space/lmcache/cachegen_determinism.py):
calculate_cdf -> encode_fast_new -> collect_bytes -> decode_fast_prefsum, 5
configs incl. 64-layer x 1024-channel x 256-token (11.2 MB). All: decode
recovers the encoded symbols byte-for-byte (round-trip exact) AND the CDF,
bytestream, lengths, and decoded output are bit-identical across two
same-seed runs (the integer-only coder is deterministic on gfx90a).

Non-GPU regression (must not regress; all PASS):
- tests/test_serde.py tests/v1/test_config.py tests/v1/test_token_database.py tests/v1/test_cache_policy.py  89 passed, 8 skip
- tests/v1/native_storage_ops/ tests/v1/storage_backend/test_fs_connector.py  330 passed
  (the pure-CPP native_storage_ops bitmap/ttl_lock/pattern_matcher are compiled by hipcc at c++17 and are unaffected)

## Gotchas / environment caveats (not gfx90a kernel issues)

- numpy vs torch.numpy() bridge: this torch (2.13.0a0, built against numpy
  1.x) raises "Numpy is not available" from tensor.numpy() under numpy >= 2.0.
  The gpu_connector tests call `.numpy()` (gpu_connectors.py:1481), so they
  need numpy < 2.0 (used 1.26.4). This is a host wheel-ABI mismatch, not a
  kernel fault.
- cupy-rocm-7-0 (14.1.0) was built against numpy 2.x and fails to import under
  numpy < 2.0 ("numpy.core.multiarray failed to import"). The two recorder
  tests import cupy, so they need numpy >= 2.0. This conflicts with the
  gpu_connector numpy<2 need -- the two test groups can't share one numpy in
  this env. Both groups were validated separately (recorders 17/17 under numpy
  2.2.6; connectors under 1.26.4). Neither is a gfx90a issue.
- test_pos_kernels.py requires `vllm` (imports vllm rotary_embedding) -- out of
  scope for a kernel port (vLLM-ROCm is heavy/engine-coupled). The pos_kernels
  fused-rope kernel is already covered by the parity test
  (test_2_compare[rotary_embedding_k_fused] PASSED), which compares c_ops vs
  the pure-Python reference directly.
- test_gpu_connector.py single-GPU flake: ~1-in-6 full-file runs, ONE
  parametrization fails an equality assert at tests/v1/utils.py:470 after a
  paged-KV async round-trip; the failing parametrization MOVES between runs
  and the flaky params pass 0/10 failures when run in isolation. This is the
  single-GPU test-harness contention/sync flake (PORTING_GUIDE line 165), not
  a deterministic kernel fault -- the underlying multi_layer_kv_transfer
  kernels pass deterministically in test_mem_kernels.py (56/56 every run).
- Hugepage register path: the host has no reserved hugepages and changing
  vm.nr_hugepages system-wide is out of bounds on this shared host. A direct
  microtest showed alloc_hugepage_pinned_ptr fails at `mmap` (MAP_HUGETLB), NOT
  at hipHostRegister -- so the register code is HIP-clean; only host config
  blocks the hugepage test. The identical hipHostRegister call on
  non-hugepage NUMA/shm memory is validated and passes.

## Inter-project MOAT deps

NONE. The c_ops extension has no build dependency on any other MOAT-ported
CUDA library (no cuBLAS/cuFFT/cuSPARSE/Thrust/CUB/CUTLASS/cuCollections/rmm/
raft). cupy-rocm-7-0 and vLLM-ROCm are pip/runtime deps, not build deps.
