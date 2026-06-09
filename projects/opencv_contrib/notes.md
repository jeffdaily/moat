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

Key sources: opencv/opencv + opencv/opencv_contrib (CMake + cuda* modules + core warp headers); ROCm/rpp, ROCm/rocDecode, ROCm/MIOpen, GPUOpen AMF; HIP kernel-language ref (warpSize/__shfl_sync 64-bit mask/__ballot 64-bit); ROCm gpu-arch-specs (wavefront per arch). Full URLs in the agent reports / [[moat-opencv-cv-cuda-rocm-gap]].
