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
