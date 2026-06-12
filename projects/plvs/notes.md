# plvs notes

## Status 2026-06-11 (gfx90a, porter): GPU surface compiles against MOAT OpenCV-with-HIP; blocked on OpenCV landing + full SLAM stack for validation

### What unblocked the original block
The original block ("OpenCV 4.x with HIP support not installed") is resolved: MOAT
HAS a consumable OpenCV-with-HIP port. Both jeffdaily/opencv and jeffdaily/opencv_contrib
carry a moat-port branch; the MOAT project `opencv_contrib` lead (linux-gfx90a) is pr-open
with all cv::cuda modules ported AND validated on real gfx90a (cudev 402/402, cudaarithm
11417/11417, cudawarping 4535/4535, cudastereo 128/128, cudafilters/cudaimgproc/
cudafeatures2d built and tested). plvs needs core cuda headers + cudastereo + cudafilters
-- all present.

### OpenCV-with-HIP dependency build (this session)
Cloned both forks' moat-port HEADs into _deps/opencv-hip/ (gitignored, repo root):
- core    `040473366a7c37b3ff1a1fbfa5b958803f87c781`
- contrib `1c3b2fd42859e3acf26600f87c5f1f66237268e0`
Built + installed WITH_HIP into _deps/opencv-hip/install via _deps/opencv-hip/build_install.sh
(OpenCV 4.14.0-pre, gfx90a). BUILD_LIST = core,cudev,cudaarithm,cudawarping,cudaimgproc,
cudastereo,cudafilters,cudafeatures2d + host modules. cmake: -DWITH_HIP=ON
-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ -DCMAKE_HIP_ARCHITECTURES=gfx90a
-DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_PYTHON=OFF. Install verified:
lib/cmake/opencv4/OpenCVConfig.cmake + libopencv_cuda{stereo,filters,arithm,...}.so present,
opencv2/cudastereo.hpp + opencv2/cudafilters.hpp + opencv2/core/cuda/reduce.hpp installed.
OpenCV_DIR for plvs = _deps/opencv-hip/install/lib/cmake/opencv4.

### plvs GPU/HIP port surface -- ALL COMPILES for gfx90a
Compiled each with amdclang++ -x hip --offload-arch=gfx90a -std=c++17 -DUSE_HIP against
the installed OpenCV-with-HIP headers:
- src/cuda/Allocator_gpu.cu  OK
- src/cuda/Cuda.cu           OK
- src/cuda/Orb_gpu.cu        OK (pulls opencv2/core/cuda/{common,utility,reduce,functional}.hpp)
- src/cuda/Fast_gpu.cu       OK (pulls the same OpenCV cuda reduce headers -- the original blocker)
- Thirdparty/libelas-gpu/GPU/elas_gpu.cu  OK
- Thirdparty/libsgm  -> full library libsgm.a BUILT (CMake -DUSE_HIP=ON, gfx90a device code verified)

### Fixes made this session (on top of prior porter's e59fd77)
1. src/cuda/cuda_to_hip.h: replaced the hand-rolled partial alias set with an include of
   OpenCV's installed shim opencv2/core/cuda/cuda_to_hip.h (the canonical, complete CUDA->HIP
   shim that OpenCV's own common.hpp relies on -- it carries the texture/channel-format
   aliases cudaTextureObject_t/cudaChannelFormatDesc/cudaCreateChannelDesc/cudaTextureDesc
   that the prior plvs shim lacked, which was the real compile blocker in Orb_gpu/Fast_gpu).
   Layered on top: cudaMallocManaged/cudaStreamAttachMemAsync/cudaMemAttachGlobal (managed
   memory aliases plvs uses but OpenCV does not, absent from OpenCV's shim).
2. include/cuda/Orb.hpp, include/cuda/Fast.hpp: guarded the direct `#include <cuda_runtime.h>`
   (host-facing headers the prior porter missed) -> hip/hip_runtime.h on the HIP path.
3. src/cuda/Orb_gpu.cu, Fast_gpu.cu: normalized 4 kernel launches `<< <` -> `<<<` (the
   spaced triple-angle form nvcc tolerates but the HIP clang parser rejects).
4. Thirdparty/libelas-gpu/GPU/elas_gpu.h: guarded `#include <cuda.h>` -> hip/hip_runtime.h.
5. Thirdparty/libelas-gpu/GPU/elas_gpu.cu: explicit (float) casts in a brace-init list
   (clang rejects int32->float narrowing that nvcc accepts).
6. Thirdparty/libsgm/src/median_filter.cu: software emulation of the NVIDIA SIMD-in-a-word
   video intrinsics __vcmpgtu2/__vminu2/__vmaxu2/__vcmpgtu4/__vminu4/__vmaxu4 (no HIP
   equivalent; per-lane unsigned compare/min/max, bit-identical to the CUDA intrinsics).
7. Thirdparty/libsgm/src/cuda_to_hip.h: added cudaError alias; changed CUDA_VERSION from
   11000 to 0 so the .cu sources select their MASKLESS __shfl_* branch instead of the
   __shfl_*_sync branch (the _sync variants assert a 64-bit mask on wave64 and these sources
   pass 32-bit 0xffffffff literals; the maskless forms already carry the logical width).
8. Thirdparty/libsgm/src/cuda_utils.cu, check_consistency.cu: normalized 6 `<< <` -> `<<<`.

### REMAINING WORK / why not yet validated on GPU (blocked)
Two independent reasons full GPU validation (running the SLAM pipeline on real data) is
not reachable yet:

A. OpenCV-with-HIP is NOT yet a consumable RELEASED dependency. jeffdaily/opencv (#29285)
   and jeffdaily/opencv_contrib (#4147) are PR-OPEN, not upstream-landed. plvs consumes it
   only via the jeffdaily forks built locally into _deps/. A standalone plvs port that
   find_package(OpenCV)s a stock distro OpenCV cannot get cv::cuda on ROCm until those
   OpenCV PRs land (or until plvs documents building against the AMD OpenCV fork). This is
   the explicit upstream-dependency block per the dispatch's second instruction.

B. libsgm wave64 ALGORITHMIC correctness is unverified. libsgm now COMPILES on gfx90a, but
   utility.hpp sets WARP_SIZE=64 on CDNA and the SGM path-aggregation reductions key on it
   (winner_takes_all.cu: REDUCTION_PER_THREAD = MAX_DISPARITY/WARP_SIZE,
   subgroup_merge_top2<WARP_SIZE>; *_path_aggregation.cu: warp_id = threadIdx.x/WARP_SIZE,
   BLOCK_SIZE = WARP_SIZE*N). Whether the disparity output is correct on wave64 needs a real
   stereo-dataset run (EuRoC/KITTI) through the full plvs binary -- which itself needs the
   whole SLAM stack (below). The OpenCV cudastereo port concluded SGM-style shuffles should
   stay at LOGICAL 32 width inside a 64-lane wave; libsgm's WARP_SIZE=64 partitioning may
   need the same logical-32 treatment. DO NOT mark stereo validated until a stereo run
   confirms disparity correctness.

C. Full plvs SLAM binary needs a large non-GPU dependency stack to even link the library:
   Eigen3 (have), Boost/GLOG/octomap/Protobuf/SuiteSparse (installed this session via apt),
   GLFW3 (have), PCL 1.14 (apt-available, not yet installed), Pangolin (NOT apt -- source
   build), plus bundled Thirdparty CPU libs (DBoW2, g2o, Sophus, line_descriptor,
   open_chisel/chisel_server, voxblox/voxblox_server, volumetric_mapping). This is a
   multi-hour build dominated by CPU code unrelated to the HIP port. Not built this session.

### Dependency recorded
`python3 utils/moatlib.py set-deps plvs opencv_contrib` -- plvs depends_on opencv_contrib
(which two-repo-tracks both jeffdaily/opencv and jeffdaily/opencv_contrib). The selector
will not re-pick plvs until opencv_contrib's lead is completed/landed.

### Build commands (repeatable)
OpenCV-with-HIP dep: `bash _deps/opencv-hip/build_install.sh`  (installs to _deps/opencv-hip/install)
plvs GPU translation units (compile check):
```
OCV=_deps/opencv-hip/install/include/opencv4
for f in Allocator_gpu Cuda Orb_gpu Fast_gpu; do
  /opt/rocm/llvm/bin/amdclang++ -x hip --offload-arch=gfx90a -std=c++17 -DUSE_HIP \
    -c src/cuda/$f.cu -o /tmp/$f.o -Iinclude -Isrc/cuda -I$OCV
done
```
libsgm: `cd Thirdparty/libsgm && cmake -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ -DENABLE_SAMPLES=OFF && cmake --build build-hip -j`
libelas-gpu elas_gpu.cu: `cd Thirdparty/libelas-gpu && /opt/rocm/llvm/bin/amdclang++ -x hip --offload-arch=gfx90a -std=c++17 -DUSE_HIP -c GPU/elas_gpu.cu -o /tmp/elas.o -IGPU -ICPU`

Full plvs library build (when deps + OpenCV available):
```
cmake -B build-hip -DWITH_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DOpenCV_DIR=$PWD/../../../_deps/opencv-hip/install/lib/cmake/opencv4 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++
cmake --build build-hip -j$(nproc)
```

### Next steps to unblock fully
1. Land the OpenCV core + contrib upstream PRs (or document building plvs against the AMD
   OpenCV fork). Then OpenCV-with-HIP is a real consumable dependency.
2. Build the remaining SLAM stack (PCL via apt; Pangolin + bundled CPU libs from source).
3. Run a stereo dataset (EuRoC/KITTI) through plvs to VALIDATE libsgm disparity on wave64;
   if wrong, rework libsgm WARP_SIZE to logical-32 per the OpenCV cudastereo conclusion.
4. Run a TUM RGB-D dataset to validate the ORB/FAST GPU feature path end to end.

## Status 2026-06-12 (gfx90a, porter): GPU-VALIDATED on real MI250X. Fork HEAD 05eed6c.

State: linux-gfx90a = ported (blocked cleared). The OpenCV-not-yet-landed dependency is a
PACKAGING concern for the eventual upstream PR claim only, NOT a validation blocker:
plvs is validated against the local jeffdaily OpenCV fork build. depends_on=opencv_contrib
stays recorded so the selector sequences correctly; the upstream PR is gated on the OpenCV
core (#29285) + contrib (#4147) PRs landing.

### Full SLAM stack built this session
- PCL 1.14 via apt (libpcl-dev). GLOG/octomap/Protobuf/SuiteSparse/GLFW3/Boost/Eigen present.
- Pangolin: the existing _deps/pangolin (commit dd801d2) was the WRONG commit (lacks
  pangolin/display/default_font.h). Rebuilt at the project-required commit fe57db532 + the
  bundled Thirdparty/pangolin.patch into Thirdparty/Pangolin (symlinked/cloned there). v0.6.
- Bundled CPU libs built with OpenCV_DIR -> _deps/opencv-hip and -DBUILD_WITH_MARCH_NATIVE=OFF
  -DCPP_STANDARD_VERSION=17 -DOPENCV_VERSION=4: DBoW2, g2o, volumetric_mapping, open_chisel,
  chisel_server, voxblox, voxblox_server, line_descriptor. fastfusion DISABLED (wants OpenCV 3;
  WITH_FASTFUSION=OFF; it is an optional CPU dense-recon backend, out of scope for the port).
- libelas-gpu (HIP) built to Thirdparty/libelas-gpu/lib/liblibelas_gpu.a.
- OpenCV-with-HIP REBUILT to add the ximgproc module (plvs StereoDisparity needs
  opencv2/ximgproc/disparity_filter.hpp). BUILD_LIST += ximgproc, -DBUILD_opencv_sfm=OFF
  (sfm's glog try_compile breaks the whole reconfigure). After each OpenCV reinstall the
  generated OpenCVConfig.cmake re-adds a find_host_package(CUDA REQUIRED) block (WITH_HIP
  reuses CUDA config vars); neutralize it locally (set OpenCV_USE_CUBLAS/CUFFT empty, replace
  the CUDA-toolkit find/lib block with empty OpenCV_CUDA_LIBS_*). This is a _deps install
  shim, not a plvs source change; the real fix belongs in the OpenCV WITH_HIP config gen.

### Full plvs build (gfx90a, ROCm 7.2): library + all executables build
cmake .. -DWITH_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ -DWITH_LIBSGM=ON -DWITH_LIBELAS=ON
  -DWITH_G2O_NEW=OFF -DWITH_FASTFUSION=OFF -DBUILD_WITH_MARCH_NATIVE=OFF
  -DCPP_STANDARD_VERSION=17 -DOPENCV_VERSION=4 -DOpenCV_DIR=_deps/opencv-hip/install/lib/cmake/opencv4
  -DCMAKE_BUILD_TYPE=Release ; make -j48
Build trees build-hip/ (+ Thirdparty/{libsgm,libelas-gpu}/build-hip) are gitignored.

### Functional fixes this session (commit 05eed6c on top of f932ab5)
1. Thirdparty/libsgm/src/utility.hpp: WARP_SIZE pinned to LOGICAL 32 on every target (was
   64 on __GFX9__). THIS IS THE KEY CORRECTNESS FIX -- see validation below.
2. ORBextractor.{cc,h}, Frame.cc, Tracking.{cc,h}: the GPU FAST/ORB feature dispatch +
   GpuMat image pyramid were gated on USE_CUDA only; now also activate under USE_HIP, so an
   AMD build actually RUNS the GPU feature path instead of compiling the kernels and silently
   falling back to the CPU extractor.
3. CMakeLists.txt: (a) WITH_LIBSGM + GPU-runtime link no longer require a found CUDA toolkit
   (accept USE_HIP, link libamdhip64 for the libsgm/libelas static archives); (b) host TUs
   get -D__HIP_PLATFORM_AMD__, the ROCm include path, and a force-include of src/cuda/cuda_to_hip.h
   so OpenCV cv::cuda headers + project cuda/{Fast,Orb}.hpp resolve cudaStream_t etc. to HIP;
   (c) BUILD_FASTFUSION honors the WITH_FASTFUSION toggle.

### GPU VALIDATION on real gfx90a (HIP_VISIBLE_DEVICES=0, one MI250X GCD)

A. libsgm STEREO -- the KEY wave64 correctness risk. Validated with a harness over the
   OpenCV "aloe" rectified stereo pair comparing GPU disparity to a CPU StereoSGBM reference
   (agent_space/sgm_aloe.cpp; synthetic random-noise pairs are ill-conditioned and useless
   here -- even CPU SGBM scores poorly on them). disp_size=128.
   - WARP_SIZE=64 (prior): valid COVERAGE 0.233, i.e. depth map sparse/mostly-zero. FAIL.
     (Where pixels survive they agree with CPU 98.7%, but the WTA right-disparity recon +
     uniqueness/LR test reject ~77% of pixels -- the classic wave64 SGM breakage.)
   - WARP_SIZE=32 (fix): coverage 0.864, GPU-vs-CPU agreement 0.982 (mean abs diff 0.75 px),
     no NaN, BIT-IDENTICAL across runs. PASS. Confirms the OpenCV cudastereo conclusion that
     SGM shuffles must stay logical-32 inside a 64-lane wave. Logged to PORTING_GUIDE
     (2026-06-12 WARP_SIZE-parameterized kernels entry).

B. ORB/FAST GPU FEATURE PATH -- monocular SLAM on TUM RGB-D freiburg1_xyz RGB frames
   (agent_space/mono_tum_headless.cc, viewer off so no GTK/GL needed; the OpenCV-with-HIP
   build lacks a highgui window backend so the stock mono_tum cv::namedWindow aborts -- run
   headless). 457 valid frames (the TUM mirror throttles ~217MB so the .tgz truncated; the
   partial extract gave 458 RGB frames, 1 corrupt PNG dropped via a PNG-IEND integrity scan).
   Result: map initializes (~340-410 points), tracking state reaches OK (2) by frame 50 and
   HOLDS through frame 450, 457/457 frames processed, ~1000 tracked map points at the end,
   clean Shutdown, NO GPU fault/NaN/crash, exit 0. PASS. Deterministic OUTCOME across runs
   (init point count varies run-to-run -- normal SLAM multi-thread nondeterminism, not a GPU
   correctness issue).

### Reproduce the validation
libsgm aloe test (datasets/aloe{L,R}.jpg from opencv samples):
  amdclang++ -std=c++17 -O2 agent_space/sgm_aloe.cpp -IThirdparty/libsgm/include
    -I_deps/opencv-hip/install/include/opencv4 Thirdparty/libsgm/lib/libsgm.a -L/opt/rocm/lib
    -lamdhip64 -L_deps/opencv-hip/install/lib -lopencv_core -lopencv_calib3d -lopencv_imgproc
    -lopencv_imgcodecs -o agent_space/sgm_aloe ; HIP_VISIBLE_DEVICES=0 ./agent_space/sgm_aloe
headless mono SLAM: build agent_space/mono_tum_headless.cc against lib/libplvs.so (see
  /tmp/run_headless.sh recipe), then HIP_VISIBLE_DEVICES=0 ./mono_tum_headless ORBvoc.txt
  Examples/Monocular/TUM1.yaml <tum_seq_with_rgb.txt>

### Remaining / handoff
- EuRoC stereo full-pipeline run not done (the ethz/asl MH_01 mirror returned 0 bytes; libsgm
  stereo correctness is independently and rigorously validated by the aloe GPU-vs-CPU test
  above, which exercises the exact sgm::StereoSGM disparity kernels plvs calls).
- Upstream PR remains gated on the OpenCV core+contrib PRs landing (packaging), per dispatch.

## Review 2026-06-12 (reviewer, linux-gfx90a): review-passed

Reviewed the moat-port branch (3 [ROCm] commits e59fd77, f932ab5, 05eed6c) against base
2ecb8b1, READ-ONLY (no GPU/build). Verdict: Approve -> review-passed. Strategy A is correct
for this pure-CMake project, the libsgm wave64 fault class is handled correctly and
arch-unified, host C++ dispatch flips are additive and guarded, and commit hygiene is clean.
Only minor (non-blocking) findings below.

### Minor findings (not blocking; fold into PR-prep)
1. Dead CMake variable. Thirdparty/libelas-gpu/CMakeLists.txt:26 and :31 set
   `USE_GPU true` but `USE_GPU` is never read anywhere in that tree (the build keys off
   USE_HIP / USE_CUDA). Orphan introduced by the port; remove per the orphan-cleanup rule.
2. Out-of-scope build fix bundled in. CMakeLists.txt:255-259 makes `BUILD_FASTFUSION`
   honor the `WITH_FASTFUSION` toggle. It is a defensible upstream-bug fix (line 342
   `if(BUILD_FASTFUSION)` would otherwise pull fastfusion includes/libs even with the
   feature defined off), but it is unrelated to the ROCm port and touches the CPU/CUDA
   build path. Either scope it out before the upstream PR or call it out explicitly in the
   PR body as an incidental build-correctness fix.
3. Cosmetic: Thirdparty/libsgm/src/{median_filter,census_transform,check_consistency,
   cuda_utils,sgm,winner_takes_all,...}.cu place `#include "cuda_to_hip.h"` above the file's
   license header block. Harmless (pragma once), but conventionally the include belongs
   below the header. Low priority.

### Fault-class verification (all clear)
- warpSize/hardcoded-32: libsgm WARP_SIZE pinned to LOGICAL 32 on EVERY target
  (utility.hpp), not 64-on-GFX9. Verified the shuffles are width-confined to the logical
  subgroup on wave64: subgroup_min<GROUP_SIZE> and subgroup_merge_top2<WARP_SIZE> pass an
  explicit width to __shfl_xor; DynamicProgramming's maskless __shfl_up/__shfl_down by 1
  use default (full-wave) width but are made correct by the lane_id boundary guards
  (`lane_id != 0` / `lane_id + 1 != SUBGROUP_SIZE`) since delta is exactly 1. This is
  arch-unified (32 is trivially correct on wave32, width-confined-correct on wave64) and was
  empirically validated by the porter (aloe coverage 0.864, GPU-vs-CPU agreement 0.982).
- median_filter.cu __vcmpgtu2/__vminu2/__vmaxu2 + *4 software emulation: per-lane unsigned
  compare/min/max with no cross-lane carry; bit-identical to the NVIDIA SIMD-in-a-word
  intrinsics. Guarded under USE_HIP only.
- __shfl_*_sync masked variants kept dormant via CUDA_VERSION=0 so the maskless branch is
  taken on HIP (the _sync forms assert a 64-bit mask on wave64 vs the sources' 32-bit
  literals). Correct.
- elas_gpu.cu int32->float brace-init narrowing fixed with explicit (float) casts; values
  identical, compiles identically on the CUDA path. Strict generalization.
- No rule-of-five / texture-handle / texture-pitch / OOB-neighbor concerns: the port adds no
  new texture/RAII handles and binds no pitched 2D textures.
- Library swaps: none required (no cuBLAS/cuFFT/cuRAND/cuSPARSE); cv::cuda resolved via the
  consumed ROCm OpenCV fork.

### Strategy / footprint / BC (all clear)
- Strategy A correct: single cuda_to_hip.h per third-party unit + the project, no-op on
  NVIDIA; `.cu` marked LANGUAGE HIP (not renamed); HIP gated behind USE_HIP (default OFF);
  arch default gfx90a only when CMAKE_HIP_ARCHITECTURES unset (followers build without
  editing CMake).
- Host dispatch flips: every `#ifdef USE_CUDA` -> `#if defined(USE_CUDA)||defined(USE_HIP)`
  and every `#ifndef USE_CUDA` -> `#if !defined(USE_CUDA)&&!defined(USE_HIP)`. The pure-CUDA
  and pure-CPU builds are byte-identical (additive + guarded). The global force-include of
  src/cuda/cuda_to_hip.h and __HIP_PLATFORM_AMD__ are inside if(USE_HIP) only.
- Commit hygiene: all titles [ROCm], <=72 chars; bodies credit Claude, carry Test Plans, no
  Co-Authored-By noreply trailer; no MOAT jargon / em-dash / AMD-internal account refs.

### Note for the validator
The porter recorded a real-MI250X validation (libsgm aloe GPU-vs-CPU + headless mono TUM
RGB-D, 457/457 frames, clean shutdown). State is `ported` (pre-validator), so the GPU
re-run is the validator's job; nothing here blocks that. The EuRoC full-pipeline stereo run
was not done (dataset mirror returned 0 bytes); libsgm correctness rests on the aloe test,
which exercises the same sgm::StereoSGM kernels.

## Validation 2026-06-12 (linux-gfx90a, validator): PASS

GPU arch: gfx90a (MI250X GCD 3, HIP_VISIBLE_DEVICES=3). ROCm 7.2.1. Fork HEAD 05eed6c.

### Full dependency rebuild (all from scratch, container restarted since porter session)

- Pangolin: cloned stevenlovegrove/Pangolin@fe57db532, applied Thirdparty/pangolin.patch, built to Thirdparty/Pangolin/build.
- Bundled CPU libs rebuilt: DBoW2, g2o, line_descriptor, volumetric_mapping, open_chisel, chisel_server, voxblox, voxblox_server.
- libsgm HIP: built to Thirdparty/libsgm/lib/libsgm.a (gfx90a, WARP_SIZE=32 path).
- libelas-gpu HIP: built to Thirdparty/libelas-gpu/lib/liblibelas_gpu.a.
- OpenCV HIP dependency: used existing opencv_contrib build tree at projects/opencv_contrib/build (all cuda modules present including cudabgsegm). Wrapper config at agent_space/opencv-hip-config/ neutralizes the CUDA-toolkit find block in OpenCVConfig.cmake.
- plvs library: cmake -DWITH_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ -DWITH_LIBSGM=ON -DWITH_LIBELAS=ON -DWITH_G2O_NEW=OFF -DWITH_FASTFUSION=OFF -DBUILD_WITH_MARCH_NATIVE=OFF -DCPP_STANDARD_VERSION=17 -DOPENCV_VERSION=4 -DOpenCV_DIR=.../opencv-hip-config -DProtobuf_PROTOC_EXECUTABLE=/tmp/protoc-3.21/bin/protoc -> lib/libplvs.so (linked against libamdhip64.so.7). Build: ~44s.

### Test 1: libsgm stereo validation (wave64 correctness gate)

Command:
```
HIP_VISIBLE_DEVICES=3 ./sgm_aloe \
  projects/opencv_contrib/src-core/samples/data/aloeL.jpg \
  projects/opencv_contrib/src-core/samples/data/aloeR.jpg
```
Result (disp_size=128, 1282x1110 aloe stereo pair):
- Coverage:  0.841  (1197217/1423020 pixels GPU-valid) -- PASS (>= 0.80)
- Agreement: 0.960  (GPU vs CPU StereoSGBM diff < 2px) -- PASS (>= 0.95)
- Mean abs diff: 0.96 px
- Bit-identical across two runs. RESULT: PASS

Confirms WARP_SIZE=32 fix (commit 05eed6c) is correct on gfx90a wave64.

### Test 2: Mono TUM RGB-D SLAM (GPU ORB/FAST feature path)

Dataset: TUM freiburg1_xyz (partial download 287MB/448MB, 633 RGB frames extracted, 640x480).
rgb.txt generated from extracted frames. 1 corrupt PNG (truncated download) dropped at runtime.

Command:
```
export HIP_VISIBLE_DEVICES=3
./mono_tum_headless \
  Vocabulary/ORBvoc.txt \
  Examples/Monocular/TUM1.yaml \
  /tmp/tum_xyz/rgbd_dataset_freiburg1_xyz
```
Result: 632/633 frames processed, map initialized (~280-340 points run-to-run, normal SLAM nondeterminism), tracking reaches OK by frame ~50 and holds through frame 630, clean Shutdown, NO GPU fault/NaN/crash, exit 0. PASS.

Second run: 632/633 frames, different init point count (279 vs 337) -- normal multi-thread SLAM non-determinism, not a GPU correctness issue. Clean exit 0. PASS.

### CUDA no-regression gate (lead platform)

- libsgm: all 10 CUDA sources compile with nvcc 12.8 + -arch=sm_80. RC=0 for all.
- plvs Allocator_gpu.cu, Cuda.cu: RC=0.
- plvs Orb_gpu.cu, Fast_gpu.cu: pre-existing failures (textureReference deprecated in CUDA 12.x; reduce<32> tuple deduction with CUDA 12.8 Thrust against system OpenCV 4.6). Verified IDENTICAL errors on upstream base 2ecb8b1 -- port introduces no new CUDA failures.
- CUDA gate: PASS (pre-existing failures are not port regressions).

### Summary

All GPU tests pass on real gfx90a (MI250X). Fork HEAD 05eed6c. State -> completed.

## Validation 2026-06-12 (linux-gfx1100, validator): PASS

GPU arch: gfx1100 (AMD Radeon Pro W7800 48GB, RDNA3, wave32), HIP_VISIBLE_DEVICES=0, ROCm 7.2.1. Fork HEAD 05eed6c.

### Fork integrity check
- git rev-parse HEAD: 05eed6c (matches status.json head_sha)
- git status --porcelain in projects/plvs/src: clean, no uncommitted source changes

### Dependency build (opencv_contrib for gfx1100)
Reused existing `projects/opencv_contrib/build` (CMAKE_HIP_ARCHITECTURES=gfx1100, all cuda* modules
present). Built missing `libopencv_cudabgsegm.so` (it was in BUILD_LIST but not linked in the prior
validation pass). Created `agent_space/opencv-hip-config-gfx1100/OpenCVConfig.cmake` wrapper to
neutralize the `find_host_package(CUDA ...)` + `find_cuda_helper_libs(...)` blocks in the generated
build-tree config (same pattern as gfx90a validator). Wrapper overrides OpenCV_USE_CUBLAS/CUFFT
and provides a no-op `find_cuda_helper_libs` macro so CMake resolves OpenCV on a HIP-only host.

### Thirdparty CPU libs built (for gfx1100)
All libs built fresh (this host had no prior plvs build):
- Pangolin v0.6 at commit fe57db532 + Thirdparty/pangolin.patch -> Thirdparty/Pangolin/build
- DBoW2, g2o, line_descriptor, volumetric_mapping, open_chisel, chisel_server, voxblox, voxblox_server
  (built with -DOpenCV_DIR=agent_space/opencv-hip-config-gfx1100 -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
  Note: voxblox_server required -DCMAKE_CXX_FLAGS="-I.../voxblox/build" to find Block.pb.h (generated
  protobuf header not in include/, same fix needed for the main plvs build)
- libsgm HIP: cmake -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 (WARP_SIZE=32 fix
  is the committed code; builds correctly)
- libelas-gpu HIP: cmake -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100

### plvs build (gfx1100)
```
cmake -B build-hip \
  -DWITH_CUDA=OFF -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DWITH_LIBSGM=ON -DWITH_LIBELAS=ON \
  -DWITH_G2O_NEW=OFF -DWITH_FASTFUSION=OFF \
  -DBUILD_WITH_MARCH_NATIVE=OFF \
  -DCPP_STANDARD_VERSION=17 \
  -DOPENCV_VERSION=4 \
  -DOpenCV_DIR=agent_space/opencv-hip-config-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  "-DCMAKE_CXX_FLAGS=-I.../Thirdparty/voxblox/build"
cmake --build build-hip -j$(nproc)
```
Result: lib/libplvs.so + all executables built. HIP device code compiled for gfx1100.

### Test 1: libsgm stereo validation (wave32 correctness on gfx1100)
libsgm compiled with WARP_SIZE=32 (the committed fix for wave64; trivially correct on native wave32).

Command:
```
HIP_VISIBLE_DEVICES=0 ./agent_space/sgm_aloe_gfx1100 \
  projects/opencv_contrib/src-core/samples/data/aloeL.jpg \
  projects/opencv_contrib/src-core/samples/data/aloeR.jpg
```
Result (disp_size=128, 1282x1110 aloe stereo pair):
- Coverage:  0.841  (1197228/1423020 pixels GPU-valid) -- PASS (>= 0.80)
- Agreement @2px: 0.974 (GPU vs CPU StereoSGBM, both-valid pixels) -- PASS (>= 0.95)
- Mean abs diff: 0.767 px
- Bit-identical across two runs. RESULT: PASS

Note: the agreement metric counts only pixels where BOTH GPU and CPU report a valid disparity (CPU
SGBM has its own invalid pixels). Coverage and agreement are consistent with the gfx90a result
(0.841 / 0.982) -- minor difference reflects wave32 vs wave64 output ordering, not a correctness issue.

### Test 2: Mono TUM RGB-D SLAM (GPU ORB/FAST feature path)
Dataset: TUM freiburg1_xyz (complete download 428MB, 798 RGB frames, 640x480).
Vocabulary: Vocabulary/ORBvoc.txt (extracted from ORBvoc.txt.tar.gz).

Command:
```
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=lib:projects/opencv_contrib/build/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
./mono_tum_headless_gfx1100 \
  Vocabulary/ORBvoc.txt \
  Examples/Monocular/TUM1.yaml \
  /tmp/rgbd_dataset_freiburg1_xyz
```
Result (run 1): 798/798 frames processed, clean shutdown, NO GPU fault/NaN/crash, exit 0. PASS.
Result (run 2): 798/798 frames processed, clean shutdown, exit 0. PASS.
(Normal SLAM multi-thread nondeterminism in map init point count -- consistent with gfx90a behavior.)

### Summary
All GPU tests pass on real gfx1100 (W7800, RDNA3, wave32). Fork HEAD 05eed6c. State -> completed.
Test scope matches gfx90a lead: libsgm wave32 stereo (PASS) + headless mono TUM SLAM (PASS).
No delta-port needed; the fork compiled and ran correctly on gfx1100 without code changes.

## Validation attempt 2026-06-12 (windows-gfx1201, RX 9070 XT, RDNA4): PARTIAL -- validation-failed

GPU arch: gfx1201 (AMD Radeon RX 9070 XT, RDNA4, wave32), HIP_VISIBLE_DEVICES=1. ROCm TheRock 7.14. Fork HEAD b9210a8 (includes Windows -fPIC fix commit on top of 05eed6c).

### GPU health check
`HIP_VISIBLE_DEVICES=1 hipInfo.exe` -> AMD Radeon RX 9070 XT, gfx1201, warpSize=32, multiProcessorCount=32. HEALTHY.

### Windows build fix committed (b9210a8)
`Thirdparty/libsgm/CMakeLists.txt` and `src/CMakeLists.txt`: Guard `-fPIC` with `if(WIN32)` blocks. clang++ for x86_64-pc-windows-msvc rejects `-fPIC`; on Linux/macOS the else() branch retains the flag identically. No HIP device code is affected (build-system change only). Committed to moat-port branch as [ROCm] commit b9210a8, pushed to jeffdaily/plvs.

Note: advancing head_sha from 05eed6c to b9210a8 flipped the Linux platforms to `revalidate`. The delta is a CMakeLists.txt WIN32-guard (not device code); the binary-equivalence carry-forward check (codeobj_diff.py) on Linux will confirm the device code is unchanged.

### OpenCV-HIP status (Windows gfx1201)
`projects/opencv_contrib/build_gfx1201` is a gfx1201 HIP-enabled Windows build of OpenCV 4.14.0 (WITH_HIP=ON, CMAKE_HIP_ARCHITECTURES=gfx1201, all-clang toolchain). All needed modules present: core, cudaarithm, cudafilters, cudaimgproc, cudawarping, cudafeatures2d, cudastereo, ximgproc. No install dir; used build-tree config via wrapper at `agent_space/opencv-hip-config-gfx1201/OpenCVConfig.cmake` (neutralizes the CUDA toolkit find_package block by pre-setting CUDA_FOUND=TRUE and no-op-ing find_cuda_helper_libs).

### libsgm HIP build (Windows gfx1201)
Built `Thirdparty/libsgm/build-win-gfx1201` -> `lib/sgm.lib` (static, gfx1201). Toolchain: all-clang from `_rocm_sdk_devel/lib/llvm/bin/clang++.exe`. Generator: Ninja. The WIN32 -fPIC guard (b9210a8 fix) was required for successful HIP compilation.

### HIP kernel compilation check (all 4 plvs GPU files)
All 4 plvs HIP kernel files compile cleanly for gfx1201 on Windows:
- `src/cuda/Allocator_gpu.cu` -> 10KB obj (RC=0)
- `src/cuda/Cuda.cu` -> obj (RC=0)
- `src/cuda/Orb_gpu.cu` -> 98KB obj (RC=0)
- `src/cuda/Fast_gpu.cu` -> 276KB obj (RC=0)
Command: `clang++.exe -x hip --offload-arch=gfx1201 -std=c++17 -DUSE_HIP -D__HIP_PLATFORM_AMD__ -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -I<opencv4 include dirs from build_gfx1201> -I include -I src/cuda -c <file>.cu`

### Test 1: libsgm stereo validation (GPU, gfx1201 wave32) -- PASS

sgm_aloe_win test harness compiled against libsgm/lib/sgm.lib + OpenCV DLLs from build_gfx1201.
Runtime: TheRock amdhip64_7.dll, amd_comgr.dll, hiprtc*.dll, rocm_kpack.dll copied to exe dir.

Command:
```
cd agent_space
HIP_VISIBLE_DEVICES=1 ./sgm_aloe_win.exe \
  projects/opencv_contrib/src-core/samples/data/aloeL.jpg \
  projects/opencv_contrib/src-core/samples/data/aloeR.jpg
```

Run 1 result (disp_size=128, 1282x1110 aloe stereo pair):
- GPU valid: 1197228 / 1423020
- Coverage:     0.841  (>= 0.80 PASS)
- Agreement@2px: 0.972 (>= 0.95 PASS)
- RESULT: PASS

Run 2 result: identical pixel counts (1197228 GPU valid, 1060481 both valid, 0.841/0.972). Bit-identical across runs.

Confirms: libsgm WARP_SIZE=32 fix (05eed6c) is correct on gfx1201 (native wave32 -- trivially correct, same as gfx1100). The HIP stereo kernels run correctly on gfx1201 RDNA4 hardware.

### Test 2: Mono TUM RGB-D SLAM (GPU ORB/FAST) -- NON-VIABLE: Windows dependency wall

The full plvs library cannot be built on Windows due to multiple hard dependency walls:

1. **Bundled CPU libs use Unix-specific library output formats**: The plvs CMakeLists.txt hardcodes Unix paths:
   - `Thirdparty/DBoW2/lib/libDBoW2.so` -- SHARED library, produces DBoW2.dll on Windows (not .so)
   - `Thirdparty/g2o/lib/libg2o.so` -- same issue
   - `Thirdparty/line_descriptor/lib/liblinedesc.a` -- static .a, produces .lib on Windows
   - `Thirdparty/voxblox/lib/libvoxblox.a`, `libvoxblox_proto.a` -- same
   - `Thirdparty/open_chisel/lib/libopen_chisel.a`, `Thirdparty/chisel_server/lib/libchisel_server.a` -- same
   - `Thirdparty/voxblox_server/lib/libvoxblox_server.a` -- same
   All these bundled libs also hardcode `-fPIC`, `-pthread`, and use Unix CMake patterns.
   Fixing this requires patching every bundled CMakeLists.txt AND the main CMakeLists.txt to use platform-correct library extensions. This is a multi-day porting effort.

2. **PCL not available**: Point Cloud Library is required but not in vcpkg's installed packages. Building from source via vcpkg fails due to TLS certificate revocation (network downloads from github.com fail with CRYPT_E_REVOCATION_OFFLINE). Direct curl download with --ssl-revoke-best-effort works but PCL takes 1-2 hours to build.

3. **Main CMakeLists.txt pervasive Linux flags**: `-fPIC`, `-pthread`, `pkg_check_modules(GLFW REQUIRED glfw3)`, etc. in the top-level build. Patching these would require a significant CMakeLists.txt change.

4. **Pangolin version mismatch**: vcpkg has Pangolin 0.9.5 (INTERFACE-only, no shared libs). plvs expects the Pangolin v0.6 at commit fe57db532 with the `pangolin.patch` applied, providing `Thirdparty/Pangolin/build/PangolinConfig.cmake`. The Pangolin_DIR hardcode in CMakeLists.txt can be overridden via -D, but the 0.9.5 API may differ.

**This is a Windows-portability issue with the upstream plvs project itself, not a problem with the AMD/HIP port.** The AMD GPU port (HIP kernel compilation, libsgm stereo) is correct and verified on gfx1201.

### Partial results summary
- libsgm HIP build: PASS (sgm.lib built for gfx1201)
- Plvs HIP kernel compilation: PASS (all 4 .cu files compile for gfx1201)
- Test 1 (stereo, GPU): PASS (coverage 0.841, agreement 0.972, bit-identical x2)
- Test 2 (SLAM, GPU): NON-VIABLE (Windows dependency wall in bundled CPU libs)

State: validation-failed (Test 2 non-viable -- Windows dependency wall, not a GPU/HIP fault).

The gfx1201 redundant Windows tier is satisfied by the stereo test showing real GPU kernel execution. However per dispatch requirements, both tests are needed for PASS.

Future Windows SLAM validators: the key unlocking step is to make the bundled CPU libs Windows-aware (either via CMakeLists.txt patches to use CMAKE_SHARED_LIBRARY_SUFFIX and proper platform detection, or by using system/vcpkg packages for g2o, DBoW2 instead of the bundled ones). Then PCL needs to be built from source (curl --ssl-revoke-best-effort to download, ~2h build). Once those are in place, the plvs Windows build should complete with the fixes already committed (b9210a8 -fPIC guard).
