# faiss notes

## Provenance / why adopted (2026-05-31)
Adopted at Jeff's suggestion after finding FAISS GPU code vendored inside Open3D.
Open3D does NOT use upstream facebookresearch/faiss as a submodule or fetched dep --
it VENDORS a SUBSET of FAISS's GPU warp-select kNN kernels directly as source under
cpp/open3d/core/nns/kernel/ (BlockSelect*, WarpShuffle.cuh, MergeNetwork, L2Select,
PtxUtils.cuh; MIT, "Copyright (c) Facebook, Inc."). The Open3D ROCm port ported that
subset in-place (PtxUtils inline-PTX -> HIP intrinsics; __shfl_*_sync masks ->
OPEN3D_FULL_WARP_MASK; unified kWarpSize=32 two-32-lane-halves model for wave64).

So Open3D's port covers only the kNN-selection subset. Upstream FAISS proper (the IVF/
IVFPQ/IVFFlat GPU indexes, StandardGpuResources, the full GpuIndex* hierarchy, cuVS
integration) is a MUCH larger standalone CUDA library and is NOT redundant with Open3D.
It is a strong MOAT target in its own right (popular GPU similarity search). The Open3D
nns/kernel port is a useful reference for the warp-select/selection-network HIP translation.

## Port disposition: ENABLE-AND-ADAPT (upstream ROCm support is mature/merged)
FAISS carries first-class, Meta-maintained ROCm support in `main` (FAISS_ENABLE_ROCM
switch -> enable_language(HIP) + a configure-time faiss/gpu/hipify.sh that translates
faiss/gpu/*.cu -> *.hip in place). This port DRIVES that path on ROCm 7.2.1 / gfx90a; it
is not a from-scratch CUDA->HIP conversion. Outcome on gfx90a: enable-only on the source
side except ONE hipify-driver fix (below). No arch-drift CMake edit, no hipBLAS-Ex fix,
no --offload-compress.

## Lead platform result: linux-gfx90a (MI250X, CDNA2 wave64), ROCm 7.2.1
- libfaiss.so + faiss_gpu_objs + all GPU gtests build clean (0 errors) at single arch gfx90a.
- GPU gate via ctest (per-process, serial, HIP_VISIBLE_DEVICES=1, OPENBLAS_NUM_THREADS=1):
  108/108 pass.
- TestGpuSelect (the raft/cuvs de-risking gate): 6/6, run twice, deterministic.

## THE ONE SOURCE CHANGE: hipify.sh device_functions.h doubled-prefix fix
faiss/gpu/utils/PtxUtils.cuh (under USE_AMD_ROCM) includes `<device_functions.h>`.
hipify-perl on ROCm 7.2.1 rewrites that to `<hip/hip/device_functions.h>` -- a DOUBLED
`hip/` prefix (hipify's CUDA->HIP header map prepends `hip/` without noticing the target
header already lives directly under hip/). The correct header is
`/opt/rocm/include/hip/device_functions.h`. Symptom: `fatal error:
'hip/hip/device_functions.h' file not found` on every PtxUtils consumer.
Fix: one post-hipify sed in faiss/gpu/hipify.sh, alongside the existing `<hipblas.h>` and
`<hiprand_kernel.h>` path fixups:
    sed -i 's@#include <hip/hip/device_functions.h>@#include <hip/device_functions.h>@' "$src"
Kept in the hipify driver (the project's own translation layer) so it survives a re-hipify;
it is the single most-likely-and-actual necessary edit, and the only one needed.

(The raft/cuvs vendored faiss_select copies do NOT hit this -- they don't carry PtxUtils'
device_functions.h include; this is specific to the full FAISS gpu/utils tree.)

## What did NOT need changing (validates the plan's "enable, don't fix" expectation)
- Arch propagation: `-DCMAKE_HIP_ARCHITECTURES=gfx90a` propagates to the HIP TUs as
  `--offload-arch=gfx90a` automatically (CMake 3.24 derives the HIP_ARCHITECTURES target prop
  from the cache var). The plan's anticipated gated `set_target_properties(faiss_gpu_objs
  PROPERTIES HIP_ARCHITECTURES ...)` was NOT needed. No literal gfx90a anywhere -> followers
  (gfx1100/gfx1151) build with only `-DCMAKE_HIP_ARCHITECTURES=<arch>`, zero source churn.
- hipBLAS GemmEx / GemmStridedBatchedEx: MatrixMult-inl.cuh already guards the compute-type
  enum on hipBLAS version (`HIPBLAS_COMPUTE_32F` when hipblasVersionMajor>=3 or v2+HIPBLAS_V2,
  else `HIPBLAS_R_32F`). Host hipBLAS is v3 with HIPBLAS_V2 -> the COMPUTE_32F branch compiles
  and runs; TestGpuDistance (the GEMM path) 28/28. No fixup.
- WarpShuffles.cuh: on HIP, CUDA_VERSION is undefined, so the `#else` MASKLESS builtins
  (`__shfl`/`__shfl_xor`, no _sync, no 0xffffffff mask) are active -- the wave-safe path with
  no `__hip_check_mask` abort risk. The `__shfl_sync(0xffffffff,...)` lines are dead on HIP.
- DeviceDefs.cuh: kWarpSize = rocprim::arch::wavefront::max_size() = 64 on gfx90a;
  GPU_MAX_SELECTION_K = 2048. Select.cuh/BlockSelect/WarpSelect fully kWarpSize-parameterized.
- --offload-compress: NOT needed. The single-arch gfx90a fatbin links into libfaiss.so with no
  R_X86_64_PC32 overflow. (Keep in mind for a future multi-arch / gfx1100+gfx90a fat build.)

## TWO host-environment artifacts that are NOT porting defects (do not chase)
1. OpenBLAS many-core heap corruption (host CPU BLAS bug, not GPU). TestGpuIndexFlat.LargeIndex
   aborts with `malloc(): corrupted top size` / `corrupted size vs prev_size` right after the
   OpenBLAS warning "precompiled NUM_THREADS exceeded, adding auxiliary array for thread
   metadata". System libopenblas 0.3.26 was built NUM_THREADS=64; this host has 128 cores, so
   OpenBLAS's >NUM_THREADS auxiliary-allocation path corrupts the glibc heap (reproduces at
   thread counts >1; OPENBLAS_NUM_THREADS=32 still aborts). OPENBLAS_NUM_THREADS=1 -> LargeIndex
   passes cleanly (idx diff 0). This is host-side CPU reference code (the test's brute-force CPU
   ground truth), would be identical on the upstream CUDA build on the same box; NOT a HIP/wave64
   issue. RUN THE GATE WITH OPENBLAS_NUM_THREADS=1.
2. Monolithic-binary teardown SIGSEGV. Running ./TestGpuIndexFlat as one process exits 139 AFTER
   printing "[ PASSED ] 18 tests" (a HIP-runtime atexit/teardown crash in the combined process).
   Per-process ctest (one process per test case) does not hit it; all 18 Flat cases pass
   individually. Likewise the TestGpuIndexIVFPQ.Float16Coarse / Add_IP "failures" only fire when
   many cases share ONE process: the global faiss RNG state advances across cases and pushes the
   float16 PQ approximate index past its 3.5% relative-error tolerance for that data realization;
   each passes in its own ctest process. ALWAYS validate via ctest, never the monolithic binaries.

## Build recipe (gfx90a, ROCm 7.2.1)
Host deps (apt, standard packages): libopenblas-dev, libgflags-dev (gflags is required by
perf_tests, which BUILD_TESTING pulls in). gtest is FetchContent-fetched by tests/CMakeLists.txt
(no system gtest needed). rocPRIM/rocThrust/hipBLAS/hip::host are in /opt/rocm. FAISS is C++20;
rocPRIM C++17 floor is satisfied (no -std bump).

Configure (Python OFF; C++ + GPU + C API + tests):
```
cmake -S projects/faiss/src -B projects/faiss/src/build \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=ON \
  -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```
Configure runs faiss/gpu/hipify.sh (execute_process): translates faiss/gpu/*.cu -> *.hip and
faiss/c_api/gpu/* in place, backing up originals to faiss/gpu-backup/ and c_api/gpu-backup/.

Build (cap -j to be a good neighbor on a shared box):
```
cmake --build projects/faiss/src/build --target faiss faiss_gpu_objs -j 16
cmake --build projects/faiss/src/build --target \
  TestGpuSelect TestGpuDistance TestGpuIndexFlat TestGpuIndexIVFFlat TestGpuIndexIVFPQ \
  TestGpuIndexIVFScalarQuantizer TestGpuIndexBinaryFlat TestGpuResidualQuantizer \
  TestGpuIcmEncoder TestGpuMemoryException TestCodePacking -j 16
```

Validate on real gfx90a (ctest = canonical gate; per-process; serial -j1):
```
cd projects/faiss/src/build
HIP_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir $(pwd) -j1 -R "TestGpu|TestCodePacking" --output-on-failure
```
TestGpuSelect alone (the de-risking gate, run twice for determinism):
```
HIP_VISIBLE_DEVICES=1 ./faiss/gpu/test/TestGpuSelect
```

## hipify is NOT idempotent -- re-hipify only from PRISTINE source
hipify.sh does `cp -r ./gpu ./gpu-backup` at the START, then hipifies in place. Re-running it on
an ALREADY-hipified tree compounds (<device_functions.h> -> hip/.. -> hip/hip/.. -> hip/hip/hip/..)
AND overwrites gpu-backup with the already-translated state. To re-hipify by hand: first restore
pristine (`git checkout -- faiss/gpu c_api/gpu`), remove artifacts (`find faiss/gpu c_api/gpu
-name '*.hip' -delete; rm -rf faiss/gpu-backup faiss/gpu-tmp c_api/gpu-backup c_api/gpu-tmp`),
THEN run hipify.sh once. A clean `cmake` reconfigure from a pristine tree does this on its own.

## DO NOT COMMIT the generated artifacts
hipify edits TRACKED files in place (the .cuh/.h/.cpp under faiss/gpu get their CUDA includes
rewritten) and creates untracked .hip files + gpu-backup/. NONE of these belong in the commit --
they regenerate at configure. The fork commit must contain ONLY the hipify.sh edit. Before
committing, restore them (`git checkout -- faiss/ c_api/`, preserving just hipify.sh) so the
diff is one file. The build dir keeps the hipified/tested artifacts untouched.

## Install as a dependency (raft, cuvs, and other FAISS-GPU consumers)
FAISS is a base library: raft (neighbors/detail/faiss_select) and cuvs vendor a CUDA-only COPY
of FAISS's gpu/utils select files. GPU-validating FAISS-ROCm proper (TestGpuSelect green on
gfx90a) is the canonical confirmation that the kWarpSize warp/block-select is wave64-correct on
CDNA; it directly de-risks the raft and cuvs select ports.

To build + install the ROCm fork for a consumer (into _deps/faiss/install at the repo root):
```
git clone https://github.com/jeffdaily/faiss.git _deps/faiss
cd _deps/faiss && git checkout moat-port
cmake -S . -B build \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install
cmake --build build --target faiss -j 16 && cmake --install build
```
Then point the consumer at it: `-DCMAKE_PREFIX_PATH=.../_deps/faiss/install` (exports the
`faiss` CMake package via faiss-targets; headers under include/faiss, lib under lib/). For a
different AMD target, change ONLY `-DCMAKE_HIP_ARCHITECTURES` (e.g. gfx1100). NOTE most consumers
vendor the select SOURCE rather than linking libfaiss; for those, the reference is
faiss/gpu/utils/{Select,WarpShuffles,PtxUtils,DeviceDefs}.* on the moat-port branch.

## Out of scope (by design, not regressions)
- cuVS/CAGRA: FAISS_ENABLE_CUVS=ON pulls NVIDIA cuVS/raft (CUDA-only), mutually exclusive with
  ROCm. TestGpuIndexCagra / TestGpuIndexBinaryCagra / TestGpuFilterConvert are CUVS-gated and
  absent on the ROCm build. Not a regression.
- bfloat16 GPU distance subtests (TestGpuDistance.*_BF16): the test self-skips ("no bfloat16
  support on AMD") -- the test's own gate, not a port failure.
- Python bindings (swig) and benchs/demos: not built (Python OFF; gflags satisfies perf_tests).
</content>

## Validation 2026-06-01 (validator, linux-gfx90a) -> completed

Device: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=1, ROCm 7.2.1
Build: reused porter build at a5c47343e73cb528bcc620e9d51cf948206383cb (intact, binaries from 2026-05-31)

TestGpuSelect (de-risking gate, run twice for determinism):
- Run 1: 6/6 PASSED (7203 ms)
- Run 2: 6/6 PASSED (6645 ms) -- deterministic

GPU index suite via ctest (per-process, serial, -j1, OPENBLAS_NUM_THREADS=1):
- ctest -R "TestGpu|TestCodePacking": 108/108 PASSED (407.97 s total)
  Includes: TestGpuIndexFlat (18, incl LargeIndex+UnifiedMemory), TestGpuIndexIVFFlat (21),
  TestGpuIndexIVFPQ (13), TestGpuIndexBinaryFlat (4), TestGpuMemoryException (1),
  TestGpuIndexIVFScalarQuantizer (12), TestGpuResidualQuantizer (1),
  TestGpuDistance (28, BF16 subtests self-skip as documented), TestGpuSelect (6),
  TestCodePacking (4)
- TestGpuIcmEncoder (7/7, run direct -- parameterized test names not matched by ctest -R)

State: completed. validated_sha = a5c47343e73cb528bcc620e9d51cf948206383cb
Followers auto-unblocked: linux-gfx1100, windows-gfx1151 -> port-ready

## Validation 2026-06-01 (validator, linux-gfx1100) -> completed

Device: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wavefront=32), HIP_VISIBLE_DEVICES=0, ROCm 7.2.1
Build: fresh clone of jeffdaily/faiss@moat-port (a5c47343e73c), cmake configure + build on this host
GPU arch in libfaiss.so: hipv4-amdgcn-amd-amdhsa--gfx1100 (168 code objects, confirmed via llvm-objdump --offloading)

Configure command (gfx1100):
```
cmake -S projects/faiss/src -B projects/faiss/src/build \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=ON \
  -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```
hipify ran cleanly at configure (the hipify.sh device_functions.h doubled-prefix fix applied).

Warp-size resolution on gfx1100 (THE wave32 question):
- DeviceDefs.cuh (ROCm 7+): `constexpr __device__ int kWarpSize = rocprim::arch::wavefront::max_size()`
- rocprim::arch::wavefront::max_size() -> min_size() when __HIP_DEVICE_COMPILE__ -> 32u when ROCPRIM_NAVI=1
- ROCPRIM_NAVI=1 when __GFX11__ (gfx1100 kernel compile). Therefore kWarpSize=32 on gfx1100.
- WarpShuffles.cuh USE_AMD_ROCM branch: maskless __shfl/__shfl_xor (no 0xffffffff mask); all shfl widths default to kWarpSize=32.
- MergeNetworkWarp.cuh: BitonicMergeStep uses `if constexpr (kWarpSize == 32)` -> takes the 32-lane branch natively.
  BitonicSortStep: `if constexpr (kWarpSize == 64)` -> skipped on gfx1100 (correct for 32-lane sort).
- No wave64-only path exists. No __GFX9__-gated code. The kWarpSize-parameterized select code handles
  both wave widths via constexpr branches -- gfx1100 natively matches NVIDIA warp=32, no adaptation needed.

TestGpuSelect (warp-select de-risking gate, run twice for determinism):
- Run 1: 6/6 PASSED (6959 ms) -- testWarp + test1Warp + testExactWarp (WarpSelectKernel), test + test1 + testExact (BlockSelectKernel)
- Run 2: 6/6 PASSED (7105 ms) -- deterministic, no HSA faults

GPU index suite via ctest (per-process, serial, -j1, HIP_VISIBLE_DEVICES=0, OPENBLAS_NUM_THREADS=1):
- ctest -R "TestGpu|TestCodePacking": 108/108 PASSED (412.40 s total)
  Includes: TestGpuIndexFlat (18, incl LargeIndex+UnifiedMemory), TestGpuIndexIVFFlat (21),
  TestGpuIndexIVFPQ (13), TestGpuIndexBinaryFlat (4), TestGpuMemoryException (1),
  TestGpuIndexIVFScalarQuantizer (12), TestGpuResidualQuantizer (1),
  TestGpuDistance (28, BF16 subtests self-skip as documented), TestGpuSelect (6),
  TestCodePacking (4)
- TestGpuIcmEncoder: 7/7 PASSED (direct run, parameterized names not matched by ctest -R)

Verdict: warp-select CORRECT on wave32, no HSA 0x1016, no NaN, no wrong neighbors/distances.
No fork changes: gfx1100 validated at a5c47343e73c with zero source delta (cmake -DCMAKE_HIP_ARCHITECTURES=gfx1100 only).
Result: 108/108 matches gfx90a (108/108) exactly.

State: completed. validated_sha = a5c47343e73cb528bcc620e9d51cf948206383cb

## Review 2026-06-01 (reviewer, linux-gfx90a) -> review-passed
Verdict: review-passed. The code change is sound; one notes-only accuracy fix to record (does not block validation, does not touch the fork HEAD).

Verified clean (no action): the diff is exactly +2 lines in faiss/gpu/hipify.sh (a comment + one sed), the only tracked change vs upstream 0c72755; the .hip files and gpu/gpu-backup in the worktree are untracked configure artifacts, not committed. The sed `s@#include <hip/hip/device_functions.h>@#include <hip/device_functions.h>@` (hipify.sh:87) is correctly anchored on the full doubled-prefix string (no over-match of hip/hip_runtime.h), idempotent on an already-fixed line, and sits inside the .tmp rewrite loop beside the existing hipblas/hiprand corrections -- the project's own translation layer, so it survives a re-hipify. It targets the real symptom: PtxUtils.cuh:12 includes <device_functions.h> under USE_AMD_ROCM, which hipify-perl rewrites with the doubled hip/ prefix. All "no fix needed" claims confirmed against source: kWarpSize = rocprim::arch::wavefront::max_size() (DeviceDefs.cuh:31, =64 on gfx90a, not a hardcoded 32); WarpShuffles.cuh USE_AMD_ROCM branch (lines 105-118) uses maskless __shfl/__shfl_xor (the _sync/0xffffffff lines are CUDA_VERSION>=9000-gated, dead on HIP); MatrixMult-inl.cuh guards HIPBLAS_COMPUTE_32F vs HIPBLAS_R_32F on hipBLAS version (pre-existing, untouched); no set_target_properties(... HIP_ARCHITECTURES) in source (faiss/gpu/CMakeLists.txt:277 sets only PIC + WINDOWS_EXPORT), so -DCMAKE_HIP_ARCHITECTURES propagates to --offload-arch; no literal gfx arch anywhere (followers are cache-var-only); no --offload-compress. ROCm enablement is gated behind FAISS_ENABLE_ROCM (default OFF) with the CUDA path untouched -- additive and BC-clean. TestGpuSelect de-risking is structurally valid: it drives WarpSelectKernel/BlockSelectKernel -> Select.cuh/WarpShuffles.cuh/MergeNetworkWarp.cuh, the exact files raft vendors verbatim under neighbors/detail/faiss_select, so FAISS-proper passing on wave64 cross-validates the raft vendored copy. Commit hygiene clean: [ROCm] title 67 chars, Claude disclosed, no noreply/Co-Authored-By trailer, Test Plan present, ASCII, no AMD-internal account refs; fork main is a clean upstream mirror (HEAD = 0c72755); moat-port is one curated commit over base.

Problem to fix (notes accuracy, not code; no re-validation needed):
- notes.md "## Install as a dependency" (line 141) and line 45 state that BOTH raft AND cuvs "vendor a CUDA-only COPY" of faiss/gpu/utils/select. Only raft vendors (projects/raft/src/cpp/include/raft/neighbors/detail/faiss_select/, no find_package(faiss)/link). cuvs does NOT vendor a faiss_select copy; it FETCHES and LINKS libfaiss via cpp/cmake/thirdparty/get_faiss.cmake (find_and_configure_faiss() at line 117, exporting faiss::faiss / faiss::faiss_gpu_objs), i.e. a genuine build dependency on the CUDA/NVIDIA-cuVS path. The closing caveat (line 165, "most consumers vendor the select SOURCE rather than linking libfaiss") generalizes the raft pattern onto cuvs. Reframe: for raft, FAISS-proper is a reference/cross-check + upstreaming candidate (vendor-only); for cuvs, libfaiss is an actual link dependency (and the build-and-install instructions in this section apply to that case). The TestGpuSelect-as-de-risking claim for the vendored raft path stands either way.
