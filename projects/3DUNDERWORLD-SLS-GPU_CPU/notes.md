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

- gfx1100: VALIDATED 2026-05-30 (see section below).
- gfx1101: VALIDATED 2026-06-05 (see section below).
- gfx1151: host retired; validation superseded by gfx1101/gfx1201.
- gfx1201: pending validation.
- The `data` submodule gitlink is missing from the dev HEAD tree (`.gitmodules`
  references it but `git ls-tree HEAD data` is empty); clone the DATA repo
  directly as above. Not a port issue.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: linux-gfx1100. GPU: 2x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). ROCm 7.2.1, hipcc/clang++ 22.0.0. SHA validated: 3a506a202f999f97bfe93b080c55e188bd7a0e35.

No source or CMake changes were needed; only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` differs from the gfx90a build.

### Build commands

```bash
cd projects/3DUNDERWORLD-SLS-GPU_CPU/src
# dep install (GLM not present on this host)
sudo apt-get install -y libglm-dev
# data: clone directly (submodule not in dev HEAD tree)
git clone --depth 1 https://github.com/theICTlab/3DUNDERWORLD-SLS-DATA.git data
rm -rf build_hip && mkdir build_hip
HIP_VISIBLE_DEVICES=0 cmake -S . -B build_hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DGTEST=ON
# wrapped with timeit:
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU compile -- \
  cmake --build build_hip -j$(nproc)
```

Build result: success, warnings only (nodiscard on cudaEvent aliases, unused variable in CPU path -- pre-existing). 3 HIP translation units compiled for gfx1100.

### Code-object evidence

```
roc-obj-ls build_hip/bin/SLS_GPU
# 3 code objects, all hipv4-amdgcn-amd-amdhsa--gfx1100; no gfx90a.
1  hipv4-amdgcn-amd-amdhsa--gfx1100  (6640 bytes)
2  hipv4-amdgcn-amd-amdhsa--gfx1100  (12392 bytes)
3  hipv4-amdgcn-amd-amdhsa--gfx1100  (27128 bytes)
```

### GPU reconstruction result

```bash
DATA=projects/3DUNDERWORLD-SLS-GPU_CPU/src/data/alexander
BIN=projects/3DUNDERWORLD-SLS-GPU_CPU/src/build_hip/bin
# Run 1
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU test -- bash -c \
  "HIP_VISIBLE_DEVICES=0 $BIN/SLS_GPU --leftcam=$DATA/leftCam/dataset1 \
   --rightcam=$DATA/rightCam/dataset1 \
   --leftconfig=$DATA/leftCam/calib/output/calib.xml \
   --rightconfig=$DATA/rightCam/calib/output/calib.xml \
   --output=/tmp/output_gpu_run1.ply --format=jpg --width=1024 --height=768"
# Run 2 (determinism)
HIP_VISIBLE_DEVICES=0 $BIN/SLS_GPU ... --output=/tmp/output_gpu_run2.ply ...
# CPU reference
$BIN/SLS ... --output=/tmp/output_cpu.ply ...
```

Point counts: GPU run1 = GPU run2 = CPU = 146064 (matches gfx90a reference exactly).

Coordinate stats (GPU run1):
- x: min=-119.898, max=135.822, mean=45.003
- y: min=-117.639, max=208.327, mean=16.143
- z: min=-116.617, max=134.884, mean=-55.124
- No NaN/Inf detected.

Matches gfx90a coordinate stats within print precision.

GPU vs CPU nearest-neighbor correspondence:
- CPU->GPU: mean=3.7e-5, p99.9=1.0e-3, max=2.5e-3; 100% coverage @tol=10.0 and @tol=0.5.
- GPU->CPU: same (symmetric). Reconstruction is numerically identical to CPU reference to file print precision.

Determinism (run1 vs run2, set-based NN):
- count match: True (146064 each)
- run1->run2: mean=4.7e-6, max=1.0e-3; 100% coverage @tol=1.0, 99.96% @tol=1e-3.
- The residual max (1.0e-3) is the ASCII PLY 6-sig-fig print-precision ceiling; identical to gfx90a behavior. The point SET is stable; only bucket-fill ordering (atomicInc) varies, producing last-digit float jitter -- well below the project's MAX_DIFF=10 tolerance.

### CPU gtest suite

```bash
cd build_hip && ctest --output-on-failure
# 3/3 passed: RunCPUTest.Arch, RunCPUTest.Alexander, CPU_TEST (44 sec total)
```

No regression in non-GPU tests.

### Verdict

PASS. gfx1100 (wave32, RDNA3) produces identical reconstruction results to gfx90a and CPU reference. No source changes were needed; the port is wave-size-agnostic. No fork interaction performed.

## Validation 2026-06-05 (windows-gfx1101, ROCm 7.14)

Platform: windows-gfx1101. GPU: Radeon PRO V710 (gfx1101, RDNA3, wave32). ROCm 7.14.0a20260604 (clang++ 23.0.0). SHA validated: 633065b857387209d619468a0f765ca7460c1ccd (commit on top of 3a506a2 adding the WIN32 OpenCV compat path).

### Windows-specific adaptations

1. OpenCV include path: upstream sources use `<opencv4/opencv2/...>` (Linux /usr/include/opencv4 layout). The Windows prebuilt OpenCV puts headers at `build/include/opencv2/` without the `opencv4/` prefix. Fixed by creating a compat directory with a junction `opencv4/ -> build/include/` and adding `-DOPENCV4_COMPAT_DIR=<compat>` to cmake (a new WIN32 conditional in CMakeLists.txt committed as 633065b).

2. cxxabi.h: `cmdline.h` (third-party header in the project) includes `<cxxabi.h>` for `abi::__cxa_demangle`. Not available in MSVC ABI (which clang++ targets on Windows). Provided a Windows stub `cxxabi.h` in the compat directory -- demangle returns the raw mangled name, acceptable for command-line error messages.

3. gtest: the test CMakeLists expects `libgtest.a` (Unix static lib naming); Windows links `gtest.lib`. Built without `-DGTEST=ON` on this host. CPU tests are already validated on Linux; skipping them here does not affect GPU correctness validation.

4. DLL loader order: copied TheRock DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc*.dll) into the exe directory so they win over System32's Adrenalin amdhip64 (exe-dir beats System32 in the Windows DLL search order).

5. OpenCV DLL (opencv_world4110.dll) copied to exe directory.

### Build commands

```bash
SRC="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src"
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
GLM_DIR="B:/develop/moat/agent_space/glm/glm"       # GLM 0.9.9.8 headers extracted here
OPENCV_DIR="B:/develop/opencv-install/extracted/opencv/build"
OPENCV4_COMPAT="B:/develop/moat/agent_space/opencv4_compat"  # opencv4/ junction + cxxabi.h stub
MSVC_VER="14.44.35207"
WINSDK_VER="10.0.26100.0"
MSVC_ROOT="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/$MSVC_VER"
WINSDK_ROOT="C:/Program Files (x86)/Windows Kits/10"

export LIB="$MSVC_ROOT/lib/x64;$WINSDK_ROOT/Lib/$WINSDK_VER/ucrt/x64;$WINSDK_ROOT/Lib/$WINSDK_VER/um/x64"
export INCLUDE="$MSVC_ROOT/include;$WINSDK_ROOT/Include/$WINSDK_VER/ucrt;$WINSDK_ROOT/Include/$WINSDK_VER/um;$WINSDK_ROOT/Include/$WINSDK_VER/shared"
export HIP_DEVICE_LIB_PATH="$ROCM/lib/llvm/amdgcn/bitcode"
export HIP_VISIBLE_DEVICES=0

# data: clone directly (submodule not in dev HEAD tree)
# git clone --depth 1 https://github.com/theICTlab/3DUNDERWORLD-SLS-DATA.git data

cmake -S "$SRC" -B "$SRC/build_gfx1101" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_PREFIX_PATH="$ROCM" \
  -DGLM_INCLUDE_DIR="$GLM_DIR" \
  -DOpenCV_DIR="$OPENCV_DIR" \
  -DOpenCV_ARCH=x64 -DOpenCV_RUNTIME=vc16 \
  -DOPENCV4_COMPAT_DIR="$OPENCV4_COMPAT" \
  -DGTEST=OFF

bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU compile -- cmake --build "$SRC/build_gfx1101" -j64

# Copy runtime DLLs to exe dir
ROCM_CORE="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
for dll in amdhip64_7.dll amd_comgr.dll rocm_kpack.dll hiprtc-builtins0714.dll hiprtc0714.dll; do
    cp "$ROCM_CORE/bin/$dll" "$SRC/build_gfx1101/bin/"
done
cp B:/develop/opencv-install/extracted/opencv/build/x64/vc16/bin/opencv_world4110.dll "$SRC/build_gfx1101/bin/"
```

Build result: success, warnings only (fopen deprecated in DynamicBitset.cpp/Log.cpp -- pre-existing; nodiscard on cudaEvent aliases -- same as Linux). Binary embeds `hipv4-amdgcn-amd-amdhsa--gfx1101` code objects (confirmed via `strings SLS_GPU.exe | grep gfx1101`).

### GPU reconstruction result

```bash
DATA="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src/data/alexander"
BIN="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src/build_gfx1101/bin"
export HIP_VISIBLE_DEVICES=0

# Run 1
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU test -- \
  "$BIN/SLS_GPU.exe" \
    --leftcam="$DATA/leftCam/dataset1" --rightcam="$DATA/rightCam/dataset1" \
    --leftconfig="$DATA/leftCam/calib/output/calib.xml" \
    --rightconfig="$DATA/rightCam/calib/output/calib.xml" \
    --output="C:/Temp/output_gpu_run1.ply" --format=jpg --width=1024 --height=768
# Run 2 (determinism): same with --output=...run2.ply
# CPU reference: SLS.exe with same args
```

Point counts: GPU run1 = GPU run2 = CPU = 146064 (matches gfx90a and gfx1100 reference exactly).

Coordinate stats (GPU run1):
- x: min=-119.898, max=135.822, mean=45.003
- y: min=-117.639, max=208.327, mean=16.143
- z: min=-116.617, max=134.884, mean=-55.124
- No NaN/Inf detected. Matches gfx90a/gfx1100 stats exactly.

GPU vs CPU nearest-neighbor correspondence (set-based NN, compare.py):
- CPU->GPU: mean=3.69e-5, p99.9=1.0e-3, max=2.52e-3; 100% coverage @tol=0.5 and @tol=10.
- GPU->CPU: same (symmetric).

Determinism (run1 vs run2):
- count match: True (146064 each)
- run1->run2: mean=5.23e-6, max=1.0e-3; 100% coverage @tol=0.5 and @tol=10.
- Residual max (1.0e-3) is ASCII PLY 6-sig-fig print-precision ceiling; identical to gfx90a/gfx1100 behavior.

### Verdict

PASS. gfx1101 (wave32, RDNA3) produces numerically identical reconstruction results to gfx90a, gfx1100, and the CPU reference. The only changes from the Linux build are the Windows-specific include-path workarounds (compat dir for opencv4/ layout and cxxabi.h stub) and the DLL copy step. GPU kernels are unchanged.
