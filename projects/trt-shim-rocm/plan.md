# trt-shim-rocm port plan

Greenfield project (not a CUDA fork): a TensorRT-API-compatible shim for ROCm,
backed by MIGraphX. Full design and rationale: `~/.claude/plans/trt-shim-rocm-migraphx.md`.

## Why

TensorRT's core `libnvinfer` is closed-source, so it cannot be ported; but its
API (headers, ONNX parser, samples, trtexec) is Apache-2.0. A shim that
re-implements the `nvinfer1` interfaces over MIGraphX lets unmodified TensorRT
apps build and run on AMD. This unblocks MOAT projects currently `cant-port`
solely on TensorRT: DynOSAM, SAM3-TensorRT, DDN-SLAM Stage 2, oneflow.

## Key insight

Real apps treat `INetworkDefinition` as a write-only token between
`nvonnxparser::parse` and `buildSerializedNetwork`. The shim's parser captures
the raw ONNX bytes and feeds `migraphx::parse_onnx_buffer -> compile(gpu) ->
run_async`, so the 57+ layer-builder methods are not needed for the common path.
The three `extern "C"` factory functions (`createInferBuilder_INTERNAL`,
`createInferRuntime_INTERNAL`, `createNvOnnxParser_INTERNAL`) are the only
symbols a consumer pulls from `libnvinfer`, so shipping `libtrtshim.so` aliased
as `libnvinfer.so` gives link-level drop-in -- which is what lets stock
`trtexec` run against it.

## Phases (each ends in a real-GPU gate)

- **Phase 0 (DONE):** repo + MIGraphX install + `migraphx_smoke` C++ tool.
  Gate: parse+compile+eval an ONNX model on gfx90a, argmax matches the ONNX
  reference golden. PASSED on gfx90a / ROCm 7.2.1 / MIGraphX 2.15.0.
- **Phase 1:** vendor TRT 10.7 headers + `NvOnnxParser.h` + `compat/cuda_runtime_api.h`;
  implement Builder/Network(token)/Config/OnnxParser/HostMemory/Runtime/
  CudaEngine/ExecutionContext + factories; backend build()/run(); enqueueV3 ->
  run_async. Gate: stock `sampleOnnxMNIST` linked against our `libnvinfer.so`
  classifies correctly on gfx90a.
- **Phase 2:** vendor + build `trtexec`; fp16/int8 (IInt8Calibrator driver +
  Q/DQ path); magic-header serialize/deserialize (reject NVIDIA `.engine`);
  ResNet-50 + YOLO. Gate: unmodified `trtexec --onnx ... --fp16/--int8/
  --save+loadEngine` on gfx90a + gfx1100.
- **Phase 3 (final committed):** single-axis dynamic shapes (IOptimizationProfile);
  stand up the ONNX backend-test backend -> per-operator coverage scoreboard.
  Gate: `sampleDynamicReshape` + `sampleNamedDimensions` pass; conformance
  matrix captured.

## Cut from committed scope (clear unsupported error; MIGraphX feedback instead)

Custom plugins (IPluginV3), control-flow ops (Loop/If/Scan), multi-axis dynamic
shapes. Each is a MIGraphX gap better fixed upstream; collected into a single
consolidated feedback report (`findings/migraphx-trt-shim-gaps/report.md`) for
jeff to forward to the MIGraphX team. Hard non-goals: NVIDIA `.engine` blobs,
CUDA plugin binaries, DLA.

## Platforms

- linux-gfx90a: LEAD, the active gate.
- linux-gfx1100: follower (validate after gfx90a).
- windows-gfx1101/1201/1151: blocked -- MIGraphX has no Windows build
  (TheRock #1912).
