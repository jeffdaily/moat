# opencv_contrib notes

## Research provenance + decision state 2026-06-09

Full porting plan in plan.md (synthesis of 3 deep-research agents + prior investigation).
- Agent 1: module/feature/dependency inventory (12 cuda modules, source-verified).
- Agent 2: proprietary-dep replacement plans (NPP->RPP+kernels; Video Codec SDK->rocDecode/AMF; Optical Flow SDK->none; cuDNN->MIOpen out-of-scope).
- Agent 3: build system + HIP-port strategy + warp-layer wave64 risk + RPP-not-a-backend + phased order.
- Prior: no existing OpenCV cv::cuda ROCm port anywhere; AMD CV-on-ROCm = OpenCL/T-API for OpenCV + native RPP/rocCV/rocAL/MIVisionX [[moat-opencv-cv-cuda-rocm-gap]].

3 GENUINE feature losses (must never be passed off as "complete"): cudalegacy graphcut (no ROCm maxflow), cudaoptflow NvidiaOpticalFlow_1_0/2_0 (no AMD HW OF engine), cudacodec codec deltas (rocDecode lacks MPEG/VC1/VP8/MJPEG; HW encode pending AMF).

STATE: lead blocked pending jeff's strategic proceed/defer decision (plan.md open-decision-1).
This is AMD-owned domain (like RAPIDS) + a multi-month 2-repo effort (opencv core + opencv_contrib). Registered + 100% scoped; NOT auto-progressing until jeff decides. Unblock to proceed.

## Decisions 2026-06-10 (jeff: PROCEED)

Jeff unblocked the port ("emboldened by the new Claude Fable model; let's see what you can do").
Open decisions resolved per plan.md recommendations:
1. Proceed (not defer). Showcase port.
2. NPP: reimplement missing entry points as direct HIP kernels (no RPP backend in pass 1).
3. Video encode: gate native HW encode off; delegate to FFmpeg AMF encoders via videoio. Decode via rocDecode.
4. graphcut: scope OUT of pass 1 (register in deferred.json when gated); revisit a maxflow kernel later.
5. Scope: cuda* contrib only. opencv dnn CUDA backend (cuDNN->MIOpen) stays out of scope; register as deferred feature-port.

Phase order per plan.md: 0=opencv core (warp layer, wave64) -> 1=cudev -> 2=pure-custom modules -> 3=NPP-light -> 4=NPP-heavy -> 5=SDK-gated.

Key sources: opencv/opencv + opencv/opencv_contrib (CMake + cuda* modules + core warp headers); ROCm/rpp, ROCm/rocDecode, ROCm/MIOpen, GPUOpen AMF; HIP kernel-language ref (warpSize/__shfl_sync 64-bit mask/__ballot 64-bit); ROCm gpu-arch-specs (wavefront per arch). Full URLs in the agent reports / [[moat-opencv-cv-cuda-rocm-gap]].

## Phase 0 + Phase 1 PORTED (2026-06-11, gfx90a lead)

opencv core (GpuMat infra + warp-layer headers) + opencv_contrib cudev module ported to ROCm/HIP and VALIDATED on real gfx90a: opencv_test_cudev 402/402 PASS, deterministic across runs (ROCm 7.2.1, wave64). Multi-arch fat binary (gfx90a;gfx1100) emits both device code objects.

### Two-repo layout
- `src/`      = jeffdaily/opencv_contrib @ moat-port (this project's head_sha tracks THIS)
- `src-core/` = jeffdaily/opencv        @ moat-port
- Base upstream SHAs: opencv_contrib `2a5154a4479e841aa1282ef83d139c4870d17b8f`, opencv `aed41fdabe3c5381395e2a7a8272c7519f3c289f`
- Fork HEADs after Phase 0/1: contrib `0e750c36725790885efe299b969479fca472d8bf`, core `f90ef85504cf3e364720ec9f8863acb87caee937`
- Two upstream PRs expected (core->opencv, modules->opencv_contrib). Both forks have Actions disabled.

### Build (## Build section)
Repeatable: `projects/opencv_contrib/build_hip.sh` (configures core with WITH_HIP, EXTRA_MODULES_PATH=../src/modules, BUILD_LIST trimmed to core+cudev+ts deps, builds opencv_test_cudev).
Key cmake invocation (build dir = `projects/opencv_contrib/build`, configure against `src-core`):
```
cmake -G Ninja ../src-core -DWITH_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DOPENCV_EXTRA_MODULES_PATH=../src/modules \
  -DBUILD_LIST="core,cudev,imgproc,imgcodecs,videoio,highgui,ts" \
  -DBUILD_TESTS=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_PYTHON=OFF
cmake --build . --target opencv_test_cudev -j$(nproc)
./bin/opencv_test_cudev   # 402/402 on gfx90a
```
GOTCHAS:
- CMake 4.0 REJECTS hipcc as CMAKE_HIP_COMPILER ("Use Clang directly"); must pass `/opt/rocm/llvm/bin/amdclang++`.
- Need CMAKE_HIP_STANDARD 17 (rocPRIM requires C++17); set in OpenCVDetectHIP.cmake.
- `hip::device` interface target injects `-x hip --offload-arch` into ALL target sources incl. plain .cpp -> link only `hip::host`; device .cu compile via per-source LANGUAGE HIP.
- OpenCVPackaging.cmake does `string(REPLACE . - ... ${CUDA_VERSION})` under HAVE_CUDA -> set CUDA_VERSION on HIP path or it errors.

### Build-system design (core CMake)
- WITH_HIP option + enable_language(HIP) block (CMakeLists.txt) + cmake/OpenCVDetectHIP.cmake (sets HAVE_HIP + HAVE_CUDA=1, CMAKE_HIP_ARCHITECTURES autodetect via amdgpu-arch defaulting gfx90a;gfx1100;gfx1101;gfx1201, force-includes the shim).
- OpenCVFindLibsPerf.cmake includes DetectHIP when WITH_HIP; OpenCVModule.cmake sets LANGUAGE HIP on globbed src/cuda/*.cu and links hip::host; the FindCUDA fallback paths (ocv_add_library, ocv_create_module link, cudev test CMakeLists) excluded when HAVE_HIP.
- Shim: modules/core/include/opencv2/core/cuda/cuda_to_hip.h (force-included via CMAKE_HIP_FLAGS -include; host cpp pulls it via private.cuda.hpp). Defines __CUDA_ARCH__=350 in device pass so cudev CV_CUDEV_ARCH>=300 shuffle fast paths light up.

### Warp-layer decisions (THE wave64 surface)
- cudev WARP_SIZE kept at LOGICAL 32 on all arches; width-32 shuffles stay inside a 32-lane subgroup of the 64-lane wavefront (PORTING_GUIDE l.145) -> wave64-correct.
- Warp::laneId(): HIP computes linearized-tid % 32 (logical lane), replacing NVIDIA `mov %%laneid` PTX. warpId unchanged.
- block/detail/reduce.hpp GenericOptimized32: loopShfl width changed `warpSize`->WARP_SIZE (literal warpSize=64 on gfx90a broke the per-32 partition).
- grid/detail/integral.hpp horisontal_pass_8u_shfl_kernel: lane_id/warp_id/leader-test changed `warpSize`->literal 32 (kLogicalWarp). This was the LAST failure fixed (Integral._8u_opt). All shfl widths already explicit 32 -> arch-agnostic.
- CV_CUDEV_ARCH=350 on HIP DEVICE pass lights up the shfl reduce/scan/integral paths; simd_functions.hpp PTX SIMD forced to software `#else` via per-file push/pop of CV_CUDEV_ARCH=0 (vadd2 PTX doesn't assemble). __CUDACC_VER_MAJOR__ undefined on HIP already routes scan.hpp/block-scan to the maskless smem branches.

### Numeric fixes (0.0-tolerance accuracy tests)
- saturate_cast.hpp (BOTH cudev util/ and core cuda/): PTX cvt.sat integer specializations replaced with clamp (min/max) on HIP; INCLUDE the from-float/double specs (round-to-nearest via __float2int_rn then clamp) -- missing them let the generic T(v) template wrap (Cvt uchar 255->0 bug).
- cast_fp16: HIP must REINTERPRET the 16-bit half bit-pattern (`*(__half*)&v` / `*(short*)&h`) like the CUDA>=9 path, NOT `__half2float(short)`/`(short)__float2half_rn` which value-convert (CvFp16 bug).
- cuda_info.cpp CudaArch: on HIP seed bin/ptx/features with level 90 so deviceSupports()/builtWith() answer true (Histogram SHARED_ATOMICS assert). gfx90a prop.major.minor = 9.0 -> version 90 >= all FeatureSets.
- vec_math.hpp: add `char abs_(char)` overload on HIP (charN::x is plain char, ambiguous between schar/short overloads).

### RESUME POINT -> Phase 2 (pure-custom modules)
Next porter dispatch: Phase 2 = cudabgsegm, cudafeatures2d, cudaobjdetect, cudastereo, cudalegacy (NCV/staging/BM, NOT graphcut). These pull the SAME core warp headers + cudev (now ported) so the foundation is done. Build by adding each module to BUILD_LIST and OPENCV_EXTRA_MODULES_PATH; expect: (1) more PTX asm in module .cu (same cvt.sat/laneid/shfl-mask classes -- apply the established patterns), (2) Thrust->rocThrust (cudafeatures2d/cudalegacy fgd), (3) per-module test binaries opencv_test_<module> are the wave64 gates (stereo/features2d exercise reductions/ballot heavily). graphcut path in cudalegacy: scope OUT + register deferred.json. NPP host helpers already HIP-guarded in private.cuda.hpp/cuda_info.cpp; NPP-using modules are Phase 3/4.

## Review findings (phase 0+1)

### Review 2026-06-11 (reviewer, linux-gfx90a): PASS, no changes requested

Reviewed both fork branches base..HEAD via /pr-review: core (jeffdaily/opencv f90ef85) and contrib (jeffdaily/opencv_contrib 0e750c3). Verdict review-passed. No blocking defects. The items below are confirmations and one forward-looking note, not change requests.

Wave64 warp layer (the #1 risk) verified CORRECT:
- block/detail/reduce.hpp GenericOptimized32 and warp_reduce_detail::WarpReductor pass an EXPLICIT width 32 down to shfl_down (loopShfl(...,WARP_SIZE=32) -> mergeShfl(...,width=32)), so the per-32 reduction is wave64-correct.
- warpScanInclusive / blockScanInclusive take the legacy smem branch on HIP (__CUDACC_VER_MAJOR__ undefined -> the `>=9` block is skipped) and inside it the CV_CUDEV_ARCH>=300 shfl path runs shfl_up(data,i) with the DEFAULT width = warpSize = 64 on gfx90a. This is still correct because the gate `if (laneId >= i)` uses laneId = tid&31: for any thread, the add only fires when tid_within_32 >= i, which forces the shfl_up source physical lane (tid - i) to stay inside the thread's own 32-lane half of the 64-wide wavefront. No cross-32-boundary contribution. Confirmed by lane arithmetic and by 402/402 passing the block_size = 40/64/70/128/256/1024 BlockScan tests (1D <<<1,block_size>>> launches that put 2 logical 32-warps in one physical wavefront).
- warp.hpp keeps WARP_SIZE logical 32 on all arches (PORTING_GUIDE-sanctioned width-32-shuffle-in-64-lane-wavefront pattern); laneId() = tid & (WARP_SIZE-1) replaces the NVIDIA `mov %%laneid` PTX. warpId()/warpFill/warpCopy all derive from WARP_SIZE=32 consistently.
- grid/detail/integral.hpp horisontal pass partitions on kLogicalWarp=32 (lane_id/warp_id/leader test) instead of physical warpSize -> wave64-correct; the integral_sums path uses blockDim.x=32 with the same laneId-gated scan.

CUDA path preservation verified: every change is HIP-guarded (WITH_HIP/HAVE_HIP/__HIP_PLATFORM_AMD__/__HIPCC__/__HIP_DEVICE_COMPILE__). CMake guards reduce to the originals when HAVE_HIP is unset (OpenCVModule.cmake `NOT HAVE_HIP AND ...`, OpenCVUtils.cmake, cudev test CMakeLists). saturate_cast (core + cudev): the new HIP specializations sit in `#if __HIP... #else <original CUDA> #endif`; the HIP and CUDA specialization SETS are identical (33 each in core, matching set in cudev) -- no missing specialization that would silently fall through to the wrapping generic T(v). fast_math.hpp / cuda_types.hpp only widen `__CUDACC__` guards to also match `__HIPCC__`; no effect on nvcc or the CPU build.

Shim correctness: __CUDA_ARCH__=350 is set only under __HIP_DEVICE_COMPILE__, lighting up the cudev CV_CUDEV_ARCH>=300 shfl/scan/integral fast paths; simd_functions.hpp push/pop-macros CV_CUDEV_ARCH to 0 for just that header so the PTX vadd2 SIMD routes to software, then restores it. cuda_compat.hpp keys on CUDA_VERSION (device macro, undefined on HIP) and safely takes the non-CUDA-13 `::double4`/`::ulonglong4` branch provided by hip_runtime.

Numeric fixes correct: integer clamps match PTX cvt.sat ranges (incl. saturate_cast<int>(uint) -> INT_MAX, short(ushort) -> 32767); float/double specs use __float2int_rn/__double2int_rn (round-nearest-even) then clamp, matching cvt.rni.sat. cast_fp16 reinterprets the 16-bit half bit-pattern (matches CUDA>=9), not a value-convert. vec_math abs_(char) overload resolves the HIP plain-char ambiguity; CUDA-guarded out.

Build system: WITH_HIP defaults OFF; add_definitions(-D__HIP_PLATFORM_AMD__) is confined to OpenCVDetectHIP.cmake (only included under WITH_HIP), never leaking to the CUDA build; modules link hip::host only (not hip::device, which would inject -x hip into plain .cpp); HAVE_CUDA=1 lights up only HIP-safe paths (NPP error tables + CUDA driver error table + nppSafeSetStream guarded out; printCudaDeviceInfo compiles because hipDeviceProp_t carries the maxTexture*/asyncEngineCount/tccDriver/surfaceAlignment fields). CUDA_VERSION fake (= hip_VERSION) is set so OpenCVPackaging's string(REPLACE) on CUDA_VERSION stays well-formed.

House style / upstreamability: AMD copyright + `\author Jeff Daily` on both substantially-new files (cmake/OpenCVDetectHIP.cmake, cuda_to_hip.h). Both commits are single, `[ROCm]`-prefixed, titles <= 72 chars, mention Claude, carry a Test Plan, no noreply trailer. Whole-diff grep: no MOAT jargon (moat/follower/Strategy/head_sha/validated_sha/revalidate/gfx1151), ASCII-only, no em-dash.

Forward-looking note (NOT a defect this phase, already captured at notes.md "RESUME POINT"): the CORE deprecated warp headers (modules/core/include/opencv2/core/cuda/{warp.hpp, warp_shuffle.hpp, emulation.hpp, scan.hpp, detail/reduce.hpp}) still carry unported NVIDIA PTX -- `asm("mov.u32 %0,%%laneid")` in warp.hpp:64, the 32-bit-mask `__shfl*_sync(0xFFFFFFFFU,...)` redefines in warp_shuffle.hpp (guarded by `__CUDACC_VER_MAJOR__>=9`, so dormant under HIP), and the __CUDA_ARCH__>=300 shfl bodies. These are UNREACHABLE from the Phase 0+1 build (cudev pulls only core/cuda/cuda_compat.hpp from core's cuda headers; gpu_mat.cu/gpu_mat_nd.cu include no warp header), so they are correctly left dormant and did not need porting here. But the shim's __CUDA_ARCH__=350 WILL activate their `>=300` PTX paths the moment a Phase 2+ module includes them, and `mov %%laneid` does not hipify -- Phase 2 must apply the same laneId/shfl-mask/ballot fixes (as already noted at the resume point) to these core headers before those modules build.

## Phase 2 progress (2026-06-11, gfx90a lead) -- pure-custom modules

Fork HEADs after the cudalegacy commit: contrib `911fca397e59f13c7b38ab631eeaa111610f99e1`, core `0e402ed`.
Test data: partial sparse clone of opencv_extra at `projects/opencv_contrib/opencv_extra/`
(`git clone --filter=blob:none --sparse`, `sparse-checkout set testdata/gpu testdata/cv`;
44M gpu + 328M cv). Set `OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata`. GPU: run on
`HIP_VISIBLE_DEVICES=0` (device 2 is the other session's). build_hip.sh now takes
`BUILD_LIST` and `BUILD_TARGET` from the environment.

### STEP 1 done -- core deprecated warp headers (commit core 5319c3d)
- warp.hpp laneId(): HIP derives logical lane = linearized-tid & (WARP_SIZE-1) (no %%laneid PTX), keyed to logical WARP_SIZE=32. HIP-guarded.
- warp_shuffle.hpp: default shuffle width pinned to logical 32 on HIP (`OPENCV_CUDA_SHFL_WIDTH`); builtin warpSize=64 on wave64 would cross the per-32 partition. CUDA keeps warpSize(==32). The `__CUDACC_VER_MAJOR__>=9` sync-mask redefines stay dormant on HIP; the non-sync `__shfl`/`__shfl_up`/`__shfl_down` HIP builtins honor width.
- detail/reduce.hpp GenericOptimized32: `loopShfl(...,warpSize)` -> `loopShfl(...,32)` under HIP guard (same per-32 reason). reduce_key_val.hpp's identical line is under `#if 0`, untouched.
- scan.hpp / emulation.hpp: NO change needed. scan uses literal `OPENCV_CUDA_WARP_SIZE`(=1<<5, already 32) and the laneId&31-gated default-width shfl_up (wave64-correct, established pattern). emulation's `__CUDA_ARCH__=350` routes syncthreadsOr/smem::atomicMin to HIP builtins; `Emulation::Ballot` (the one 64-bit-ballot-truncation risk) is UNUSED by any Phase 2 module so left alone for CUDA byte-identity.

### Shim additions (cuda_to_hip.h, commits core 5319c3d + 0e402ed)
- cudaFuncSetCacheConfig -> typed helper `cv::cuda::detail::hipFuncSetCacheConfigT` (a function-like macro can't wrap it: template kernel ids carry commas; hipFuncSetCacheConfig wants const void* and a __global__ template id doesn't implicitly convert). cudaFuncCachePreferShared added.
- constant memory: cudaMemcpyToSymbol{,Async}, cudaMemcpyFromSymbol{,Async}, cudaGetSymbolAddress. cudaMemset{,Async}, cudaMallocHost.
- texture: cudaFilterModeLinear, cudaReadModeNormalizedFloat, cudaAddressModeWrap/Mirror/Border.
- OpenCVModule.cmake: per-module accuracy-TEST executables now link hip::host when HAVE_HIP (NCV.hpp pulls hip/hip_runtime.h; test .cpp are plain g++ and need the ROCm include path).

### Modules ported + GPU test results (literal: `HIP_VISIBLE_DEVICES=0 OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata ./bin/opencv_test_<m>`)
- cudabgsegm (contrib 264d339): builds + links. MOG/MOG2 accuracy tests are gated behind `HAVE_VIDEO_INPUT`, a macro this OpenCV 4.14-pre tree NEVER defines (confirmed: absent from core, contrib, samples, apps) -> 0 tests run on CUDA too. Not a port gap; kernels compile for gfx90a. Fix: HIP-guard mog2.hpp's `typedef ... cudaStream_t` forward-decl (collides with the shim's hipStream_t mapping).
- cudastereo (contrib 264d339): **128/128 PASS**. Fixes: drop gratuitous `<cuda.h>` under HIP; software `__vminu2/__vmaxu2/__vcmpgtu2` + byte-wise `*4` (NVIDIA SIMD-in-a-word, absent on HIP) for StereoSGM's median sort; **census-transform border-zero** (NEW FAULT CLASS, see below). StereoSGM shuffle helpers already use logical cudev::WARP_SIZE=32 widths + non-sync __shfl on HIP -> wave64-correct.
- cudalegacy (contrib 911fca3): **14/14 PASS** (NPPST staging: Integral/SquaredIntegral/RectStdDev/Resize/VectorOperations/Transpose; NCV: VectorOperations/HaarCascadeLoader/HaarCascadeApplication/HypothesesFiltration/Visualization; Calib3D: TransformPoints/ProjectPoints/SolvePnPRansac). Fixes: widen `#ifdef __CUDACC__` -> `|| __HIPCC__` in NCV.hpp/NCVHaarObjectDetection.hpp/fgd.hpp; cuda_runtime.h -> hip/hip_runtime.h + shim in NCV.hpp/NPP_staging.cu/NCVPyramid.cu/NCVTest.hpp; graphcuts.cpp `#if ...||__HIP_PLATFORM_AMD__` so HIP takes the modern-CUDART(>=8000) throw_no_cuda stub (graphcut + connectivityMask + labelComponents all stubbed, same as modern CUDA); solvePnPRansac's `cuda::minMaxLoc` (cudaarithm, optional dep) gets a host argmax fallback over the 1xN scores row when cudaarithm absent. The NCV Haar cascade working here is what cudaobjdetect needs.
- graphcut DEFERRED: `python3 utils/deferred.py list` id=opencv-cudalegacy-graphcut.

### NEW FAULT CLASS: ROCm allocator does not zero-init device memory
cudastereo census_transform: the kernel writes only interior pixels and leaves the
half-window border untouched, relying on the destination GpuMat already being 0.
CUDA's device allocator happens to hand back zeroed pages so the border stays 0 and
the zero-tolerance accuracy test passes; the ROCm allocator returns REUSED (non-zero)
memory, so the border is garbage and the test fails -- but only after a prior test has
dirtied a similarly-sized buffer (passes in isolation, fails in-suite; a classic
allocator-reuse tell). Fix: `dest.setTo(Scalar::all(0), stream)` before the launch,
HIP-guarded to keep CUDA byte-identical. This is a latent UPSTREAM bug (kernel relies
on implicit zero-init); worth an upstream note. GENERAL LESSON: never assume fresh
device memory is zero on ROCm; any kernel that writes a sub-region and leaves a border
must explicitly zero it.

### RESUME POINT -> cudafeatures2d + cudaobjdetect must wait for Phase 3/4
DONE + passing this dispatch: cudabgsegm (builds; tests gated upstream), cudastereo
(128/128), cudalegacy (14/14). These are the 3 modules with NO NPP-module dependency.

cudafeatures2d and cudaobjdetect are NOT pure-custom in practice: they have HARD
(non-OPTIONAL) CMake deps AND real algorithmic runtime calls into the Phase 3/4 NPP
modules, so they cannot build or be validated until those land:
- cudafeatures2d `ocv_add_module(... opencv_cudafilters opencv_cudawarping)`; ORB
  (orb.cpp) calls `cuda::resize` (cudawarping) for the image pyramid and holds a
  `cuda::Filter` Gaussian blur (cudafilters). Stubbing them makes ORB produce wrong
  keypoints -> tests fail. Also Thrust->rocThrust still needed in orb.cu/bf_*.
- cudaobjdetect `ocv_define_module(... opencv_cudaarithm opencv_cudawarping ...)`;
  cascadeclassifier.cpp calls `cuda::resize` (cudawarping) + `cuda::integral`
  (cudaarithm) per pyramid level. Its NCV/Haar device half is ALREADY proven via
  cudalegacy's HaarCascadeApplication 14/14, but the host cascade driver needs
  resize+integral to run.

cudawarping NPP surface is confined to warp.cpp (rotate/warpAffine/warpPerspective,
gated by USE_NPP_STREAM_CTX; the remap/resize/pyr*/warp.cu kernels are custom HIP).
cudaarithm is the heaviest NPP module. So the next dispatch should be Phase 3
(cudawarping -> cudafilters) and Phase 4 (cudaimgproc -> cudaarithm), and fold
cudafeatures2d + cudaobjdetect in once cudawarping/cudafilters/cudaarithm exist.
This is a phase-sequencing boundary, not a blocker.

## Orchestrator state note 2026-06-11

Phase 0+1 foundation: ported -> review-passed (clean, see "Review findings (phase 0+1)").
State then walked back review-passed -> porting by the ORCHESTRATOR (deliberate
override of the transition table): this is a phased multi-month port and the table
has no review-passed -> porting edge. Going through completed would prematurely
unblock followers onto a 2-of-6-phase port; recording validation-failed would be
false. Lead stays in the porting loop (porter -> reviewer per phase) until all
phases land; the full reviewer -> validator -> completed gate runs at the END.
Reviewer's forward-looking note for Phase 2: the CORE deprecated warp headers
(core/cuda/warp.hpp PTX laneid, warp_shuffle.hpp 32-bit masks, emulation.hpp /
scan.hpp / detail/reduce.hpp PTX bodies) are dormant in Phase 0+1 but ACTIVATE
when Phase 2 modules include them (shim defines __CUDA_ARCH__=350); they need the
same laneId/shfl-mask/ballot fixes as cudev got.
