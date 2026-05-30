# 3DUNDERWORLD-SLS-GPU_CPU notes

ROCm/HIP port. Lead: linux-gfx90a (CDNA2, wave64), validated on GPU ordinal 2
(MI250X). Strategy A (compat header + `enable_language(HIP)` + `.cu` LANGUAGE HIP).

## Build classification + strategy

Pure CMake (no Torch). Legacy `find_package(CUDA)` + `cuda_add_library` /
`cuda_add_executable`, gated on `ENABLE_CUDA` (default OFF). -> Strategy A.

CUDA surface is tiny and clean: cudaMalloc/Free/Memcpy/Memset, H2D/D2H,
cudaPeekAtLastError, cudaGetErrorString, cudaError_t/cudaSuccess, cudaEvent_*
(profiling only), atomicInc (bucket insert), __threadfence + a trap. No
__shfl/__ballot/warpSize, no textures/surfaces, no curand/cublas/cufft/thrust.

## Existing AMD support

None (no HIP path, no OpenCL/Vulkan/SYCL backend, no stale ROCm branch). Genuine
CUDA->HIP port.

## Changes (all USE_HIP-guarded; CUDA path byte-for-byte unchanged)

1. `src/lib/ReconstructorCUDA/cuda_to_hip.h` (new) -- the single HIP-aware file.
   On USE_HIP: includes `<cstring>`/`<cstdlib>` then `<hip/hip_runtime.h>` and
   aliases the cuda* surface to hip*. Else plain `<cuda_runtime.h>`.
   Force-included (CMake `-include`) on the HIP GPU targets so it precedes GLM.
2. `CUDA_Error.cuh` -- include `cuda_to_hip.h` instead of bare `<cuda_runtime.h>`.
3. `DynamicBits.cuh` -- `asm("trap;")` (NVPTX, illegal on amdgcn) ->
   `__builtin_trap()` (portable). Never-taken overflow guard; no output effect.
4. `ReconstructorCUDA.cu` / `FileReaderCUDA.cu` -- guard the NVIDIA-only
   `<device_launch_parameters.h>` include behind `!USE_HIP` (HIP provides those
   builtins intrinsically).
5. CMake: root adds `USE_HIP` option (+ unified `USE_GPU` flag); on HIP
   `enable_language(HIP)`, arch from `${CMAKE_HIP_ARCHITECTURES}` (default
   gfx90a ONLY when unset -- never a literal). `ReconstructorCUDA/CMakeLists.txt`
   and `src/app/CMakeLists.txt` mark the existing `.cu` `LANGUAGE HIP`, set
   `HIP_ARCHITECTURES`, `cxx_std_17`, and force-include the compat header. CUDA
   branch keeps `cuda_add_*` unchanged.
6. Bit-rot repair (see Quirks) -- host-side, required for the GPU path to build
   on ANY backend: repoint `FileReaderCUDA` from the deleted `FileReader` /
   `core/FileReader.h` to the current `ImageFileProcessor`; make
   `ReconstructorCUDA` a standalone class (owns cameras_/projector_) instead of
   deriving the refactored pure-interface `Reconstructor`.

## Fault classes

- wave64 / warp size: NONE apply. No warp primitives, no hardcoded 32, no
  warp-sized shared arrays. Grid-stride, per-thread-independent kernels. (Implies
  gfx1100/gfx1151 RDNA wave32 should pass with no delta.)
- NVPTX inline asm: `asm("trap;")` -> `__builtin_trap()` (fix #3).
- OOB reads: audited, already safe (add2Bucket clamps bktIdx; buildBuckets
  checks projector bounds + mask; color idx are in-bounds pixel indices;
  atomicInc wraps at bucket capacity so the row write stays in-bounds). No clamp
  fix needed; confirmed at runtime -- AMD would have faulted on a stray read and
  the run completed clean.
- atomicInc on managed memory dropped-RMW class: N/A (plain cudaMalloc device
  memory; only int/uint atomicMin/atomicMax are in that class, not atomicInc).

## Quirks (also see PORTING_GUIDE changelog)

- GLM (0.9.9.8, libglm-dev) only emits `__host__ __device__` on its math
  functions when it detects the NVIDIA CUDA compiler via `__CUDACC__` +
  `CUDA_VERSION>=7000` (glm/simd/platform.h); hipcc/clang defines neither, so
  every glm:: call (dot/length/normalize/mat*vec, used in the device
  triangulation helpers) is host-only and the kernels fail with "call to
  __host__ function from __device__ function". GLM's qualifier macros
  (GLM_FUNC_QUALIFIER etc.) are redefined UNCONDITIONALLY in detail/setup.hpp, so
  pre-defining them does NOT win. The working fix: in the compat header, AFTER
  `<hip/hip_runtime.h>` is fully parsed, define `__CUDACC__`, `CUDA_VERSION 8000`,
  and `GLM_FORCE_CUDA`, then let GLM be included (transitively) afterwards -- this
  steers GLM into its CUDA path so its qualifiers become `__device__ __host__`,
  matching how a real CUDA build gets device-callable glm. Force-include the
  compat header on the HIP targets so it precedes GLM. (Defining __CUDACC__ after
  the HIP runtime is parsed does not disturb HIP.)
- memcpy/memset in device helpers: inside a .cu compiled as HIP, these resolve to
  HIP device overloads once hip_runtime is in scope; include <cstring>/<cstdlib>
  BEFORE hip_runtime in the compat header so the libc host decls remain available
  (matches the gpuRIR lesson).
- Upstream GPU-path bit-rot: the shipped `FileReaderCUDA`/`ReconstructorCUDA`
  reference a `FileReader` base and `core/FileReader.h` that were renamed
  (`Camera->ImageProcessor`, `FileReader->ImageFileProcessor`, commit 82078e2)
  and a pre-refactor `Reconstructor` base with `cameras_`/`projector_`/`addCamera`
  that was later reduced to a pure `reconstruct(Buckets)` interface (23a06f1). So
  the CUDA GPU build did NOT compile on stock nvcc either. A half-finished
  `ImageFileProcessorCUDA.cuh` (header only, unused) shows the intended rename.
  Fixed minimally on the host side (fix #6) so the GPU pipeline -- which builds
  buckets on-device and is a standalone parallel path -- compiles and runs.

## Build (gfx90a, GPU 2)

```bash
cd projects/3DUNDERWORLD-SLS-GPU_CPU/src
# deps: apt install libglm-dev ; OpenCV (libopencv-dev) already present (4.6.0)
# data: git submodule path `data` is not in the dev HEAD tree; clone directly:
#   git clone --depth 1 https://github.com/theICTlab/3DUNDERWORLD-SLS-DATA.git data
rm -rf build_hip && mkdir build_hip
HIP_VISIBLE_DEVICES=2 cmake -S . -B build_hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DGTEST=ON
HIP_VISIBLE_DEVICES=2 cmake --build build_hip -j$(nproc)
# Outputs: build_hip/bin/{SLS,SLS_GPU}, build_hip/test/runCPUTest
# SLS_GPU links libamdhip64.so.7 and embeds 3 gfx90a code objects (roc-obj-ls).
```

For a follower arch (RDNA wave32), only the configure changes -- no source edit:
`-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1151).

## Validation (real GPU, GPU 2 = MI250X gfx90a) -- PASS

Dataset: shipped `alexander` (1024x768 projector, 42 jpg/cam) -- the README demo
and gtest dataset.

```bash
DATA=projects/3DUNDERWORLD-SLS-GPU_CPU/src/data/alexander
BIN=projects/3DUNDERWORLD-SLS-GPU_CPU/src/build_hip/bin
# GPU run (x2 for determinism), CPU reference run -- same args
HIP_VISIBLE_DEVICES=2 $BIN/SLS_GPU --leftcam=$DATA/leftCam/dataset1 \
  --rightcam=$DATA/rightCam/dataset1 \
  --leftconfig=$DATA/leftCam/calib/output/calib.xml \
  --rightconfig=$DATA/rightCam/calib/output/calib.xml \
  --output=./output.ply --format=jpg --width=1024 --height=768
$BIN/SLS  ... (same args)   # CPU reference
```

Results (comparator: agent_space/sls_val/compare.py, not committed):
- Point counts identical across all runs: GPU run1 = GPU run2 = CPU = 146064.
- GPU vs CPU correspondence: 100% coverage both directions @tol=10 AND @tol=0.5;
  nearest-neighbor mean 3.2e-5, p99.9 1.0e-3, max 1.0e-3 world units. The 1e-3
  ceiling is the ASCII PLY print precision (6 sig figs) -- i.e. the GPU cloud is
  identical to the CPU reference to the file's representable precision. Project's
  own pass tolerance is MAX_DIFF=10, so this is ~4 orders of magnitude inside it.
- Determinism (GPU run1 vs run2): same 146064 points; NN mean 1.8e-5, max 1.0e-3
  (print-precision ceiling). Per-axis sorted distributions agree to 1e-3. The
  only run-to-run variation is last-ASCII-digit (~1e-4) float jitter from the
  bucket-fill order being atomicInc-dependent, which reorders the per-bucket
  avgPoint float sum (float non-associativity). Geometry/topology and point set
  are stable; this is well below output precision and the test tolerance.
- CPU gtest `runCPUTest` (Arch + Alexander vs in-tree reference clouds,
  MAX_DIFF=10): both PASS. No regression in non-GPU tests.

Verdict: HIP GPU reconstruction is numerically equal to the CPU reference (to
print precision) and to the gtest ground-truth, with stable correspondences and
output-precision determinism on gfx90a. Validated.

## Outstanding / follower notes

- gfx1100/gfx1151: expected to pass with only a `-DCMAKE_HIP_ARCHITECTURES`
  change (no wave64 dependence). Validate on that hardware before marking.
- The `data` submodule gitlink is missing from the dev HEAD tree (`.gitmodules`
  references it but `git ls-tree HEAD data` is empty); clone the DATA repo
  directly as above. Not a port issue.
