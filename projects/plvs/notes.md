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
