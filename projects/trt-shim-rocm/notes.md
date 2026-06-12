# trt-shim-rocm notes

## gfx1100 validation hand-off

The lead (linux-gfx90a) is validated: ctest 9/9 plus stock sampleOnnxMNIST and
stock trtexec passing. gfx1100 is the follower; it reuses the SAME source
(jeffdaily/trt-shim-rocm @ main -- greenfield, no per-arch branch) and just
re-runs the gates on its own GPU.

Key fact that makes this easy: the shim is host-only C++ over migraphx::c +
hip::host. It has NO device kernels and NO wavefront-size assumptions -- every
GPU codegen decision lives inside MIGraphX. So shim correctness is
arch-independent; what gfx1100 actually exercises is MIGraphX on RDNA3/wave32 +
the shim. A test that passes on gfx90a but fails on gfx1100 is almost certainly a
MIGraphX RDNA3 codegen/accuracy issue, NOT a shim bug (precedent: the ffpa-attn
Triton-AMD wave32 bug, and the MIGraphX dynamic-concat codegen bug we already
filed). Capture any such failure under findings/ and treat it as a MIGraphX
finding.

Prereqs on the gfx1100 host:
- ROCm + `migraphx` + `migraphx-dev` for the host's ROCm, and confirm the
  installed MIGraphX has gfx1100 in its target list (ROCm packages normally do).
- python3 is REQUIRED for the build (CMake runs tools/gen_stubs.py).
- onnx / onnxruntime / numpy are OPTIONAL -- only for regenerating test assets,
  the model sweep, and the ONNX scoreboard. The committed ctest needs none of
  them (models/goldens are committed).

Build + validate (the gate):
```
cd projects/trt-shim-rocm/src           # or clone jeffdaily/trt-shim-rocm
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/rocm   # use the host's ROCm path
cmake --build build -j
ctest --test-dir build --output-on-failure          # expect 9/9
```
The trtexec ctest sets LD_LIBRARY_PATH=build itself (for the runtime dlopen of
libnvinfer.so.10). trtexec will print DIFFERENT device info on gfx1100 (name,
compute capability ~11.0, fewer CUs, less VRAM) -- that is expected; the compat
aliasing is identical.

Optional broader validation (network-dependent, downloads models, ~30-60 min):
`bash test/run_gpu_tests.sh`, `bash tools/sweep_models.sh`, and
`python3 tools/onnx_backend_scoreboard.py build/trtshim_infer`. Re-running the
scoreboard on gfx1100 is the best way to catch RDNA3-specific op differences vs
the gfx90a result (test/onnx_scoreboard.md).

Record the result:
- All green -> set projects/trt-shim-rocm/status.json linux-gfx1100 to
  `completed` (validated on a real gfx1100 GPU) and commit-project.
- A gate fails -> capture it; if it is a MIGraphX RDNA3 issue, write a
  findings/<slug>/ note + add a deferred.py rocm-bug-report, and set gfx1100
  `validation-failed` (or `blocked` with the reason). NOTE: linux-gfx1100 is a
  PR-REQUIRED platform, so an unresolved gfx1100 failure blocks the upstream PR.

## Host / toolchain (verified 2026-06-12)

- gfx90a, ROCm 7.2.1, MIGraphX 2.15.0 (`apt install migraphx migraphx-dev`).
- MIGraphX C++ header: `/opt/rocm/include/migraphx/migraphx.hpp` (wraps the C
  API in `migraphx.h`). CMake: `find_package(migraphx)`, link `migraphx::c`.
- Compilers: hipcc / amdclang++ present; the Phase 0 smoke builds fine with
  plain g++ since it only uses the MIGraphX C++ wrapper (no device code yet).
  The shim's HIP TUs (vendored TRT headers + compat shim) will need amdclang++.
- `onnx` 1.21 + numpy present; `onnxruntime` is NOT installed. Goldens use
  `onnx.reference.ReferenceEvaluator` (the spec reference backend) -- no extra
  dependency and more rigorous than the ORT CPU EP for conformance.

## Build / test recipe

```
cd projects/trt-shim-rocm/src
ROCM_PATH=/opt/rocm bash test/run_gpu_tests.sh     # regen golden, build, ctest
# or manually:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## MIGraphX C++ API facts confirmed from the installed header

Resolves the plan's "verify" items; trust these over the earlier web research.

- `migraphx::parse_onnx_buffer(const void* data, size_t size [, onnx_options])`
  -> `program`. Confirms the ONNX-capture design: the shim's parser keeps the
  raw bytes and parses here at build time.
- `program::compile(target [, compile_options])`. `target("gpu")` selects the
  GPU target.
- `compile_options::set_offload_copy(bool)` is the IO-ownership knob:
  - `true`  -> MIGraphX inserts host<->device copies; bind HOST buffers in the
    parameter_map. Used by the Phase 0 smoke (simplest correctness proof).
  - `false` -> caller supplies DEVICE pointers; zero-copy. This is what the
    shim's `enqueueV3` path uses to honor TensorRT `setTensorAddress`.
- `argument(shape, void* buffer)` wraps a caller-provided buffer (host or
  device) -- the direct analogue of `setTensorAddress(name, ptr)`.
- `program_parameters` is the name->argument map (`add(name, argument)`); the
  analogue of TensorRT's named IO bindings.
- `program::eval(program_parameters)` runs synchronously and returns
  `arguments`. `program::run_async(program_parameters, Stream* s)` is a template
  taking a stream pointer (pass a `hipStream_t`); maps to `enqueueV3(stream)`.
  Phase 0 uses `eval`; the shim uses `run_async`.
- `program::get_parameter_shapes()` / `get_output_shapes()` supply the IO
  metadata `ICudaEngine::getIOTensorName/getTensorShape/getTensorIOMode` need.
- `shape`: `lengths()`, `strides()`, `type()`, `elements()`, `bytes()`.
- Serialization: `migraphx::save(program, file)` / `load(file)` (msgpack). The
  shim wraps this with a magic header so NVIDIA `.engine` blobs are rejected
  cleanly rather than crashing.
- `onnx_options` exposes input-shape overrides, `default_dim_value`, and loop
  iteration limits (`set_default_loop_iterations`, `set_limit_loop_iterations`)
  -- MIGraphX unrolls ONNX loops at parse time up to a bound. Minor; control
  flow is out of committed scope regardless.

## Phase 0 result

`migraphx_smoke` on gfx90a: GPU argmax matches the ONNX reference golden
(argmax=9), logits agree to ~6 decimals. The backend primitives the shim builds
on are confirmed on real hardware.

## Phase 1a: shim mechanism (DONE)

End-to-end TensorRT API path through the shim runs on gfx90a: a minimal driver
(tools/driver_smoke.cpp) does createInferBuilder -> createNetworkV2 ->
createParser -> parseFromFile -> buildSerializedNetwork -> createInferRuntime ->
deserializeCudaEngine -> createExecutionContext -> setTensorAddress -> executeV2
and gets the correct argmax. It links the shim via the libnvinfer.so /
libnvonnxparser.so aliases (link-level drop-in confirmed: `ldd driver_smoke`
shows libtrtshim.so; `nm -D` exports createInferBuilder_INTERNAL etc.).

Key implementation facts (TensorRT 10.7 headers):
- The public nvinfer1::IXxx classes forward to a protected `apiv::VXxx* mImpl`
  (the V-classes are the pure-virtual interfaces, in NvInferImpl.h). The shim
  derives from BOTH (`class ShimEngine : public ICudaEngine, public
  apiv::VCudaEngine`) and sets `mImpl = this` in the ctor. INoCopy has a
  protected default ctor, so this is legal from a derived class.
- Only VRuntime, VCudaEngine, VExecutionContext declare `getPImpl()`; the others
  do NOT -- do not add a getPImpl override where the V-class lacks it.
- The shim classes must live in `namespace nvinfer1` so the generated stub
  signatures (which use unqualified `Dims`, `ITensor`, ... because the V-classes
  are in nvinfer1) resolve.
- nvonnxparser::IParser is a plain pure-virtual interface (no apiv). Its 19
  methods are implemented directly. `SubGraphCollection_t` is in the GLOBAL
  namespace (declared before `namespace nvonnxparser`).
- ~280 trivial overrides are generated by tools/gen_stubs.py from NvInferImpl.h
  (reference-returning stubs abort via trtshim_die; the rest log+return a
  default). The network (INetworkDefinition) is a write-only token: its 75
  methods are all stubs except getInput/getOutput/getNb*, populated by the
  parser from MIGraphX introspection.

MIGraphX integration decisions:
- offload_copy=TRUE everywhere for now. With offload_copy=false,
  get_parameter_shapes() exposes internal scratch params (size()==2 for a
  1-input model) and shape introspection is messier. With true,
  get_parameter_shapes() is exactly the model inputs. eval() then takes/returns
  HOST buffers, so the shim stages caller DEVICE pointers through host buffers
  in run() (device->host before eval, host->device after). True zero-copy
  (offload_copy=false + caller device pointers) is a deferred optimization.
- Output names: MIGraphX get_output_shapes() is positional (no names), so for a
  single output the shim synthesizes "output". The real ONNX output-name
  extractor (needed for the stock sample's "Plus214_Output_0") is Milestone 1b.

Bug found and fixed (worth remembering): `v.assign(s.lengths().begin(),
s.lengths().end())` calls lengths() twice, so begin()/end() index DIFFERENT
temporary vectors -> garbage range -> std::length_error. Always bind the
container to a local first.

## Phase 1b: stock sampleOnnxMNIST runs (DONE)

The UNMODIFIED NVIDIA sampleOnnxMNIST (TensorRT v10.7.0), built against the shim
(linked as libnvinfer.so / libnvonnxparser.so), runs on gfx90a and prints
`&&&& PASSED TensorRT.sample_onnx_mnist`, classifying the digit correctly. This
exercises the real common/ helpers (buffers.h, safeCommon.h, timingCache, ...),
whose CUDA-runtime calls resolve to HIP via include/compat. ctest: 3/3.

What it took beyond 1a:
- ONNX output-name extractor (src/backend.cpp PbReader): MIGraphX has no output
  names, so a minimal protobuf walk recovers GraphProto.output[].name (field
  7 -> 12 -> 1). The engine now reports the real names (Input3,
  Plus214_Output_0) that the sample hardcodes and looks up buffers by.
- CUDA-runtime->HIP compat headers (include/compat/cuda_runtime_api.h, cuda.h,
  cuda_runtime.h, cuda_fp16.h): the executed runtime calls (malloc/free/memcpy,
  streams, events, device queries) forward to HIP; CUDA graph-capture symbols
  (used only by helper code the sample never calls) are stubbed with
  CUDA-shaped signatures to avoid HIP graph-API signature coupling. The QNX/safe
  path (cuda_runtime_api_safe_ex.h, cudaSafeEx*) is behind `#if IS_QNX_SAFE`,
  undefined, so it is skipped.
- Vendored the sample + common + utils unmodified; compiled logger.cpp,
  timingCache.cpp, fileLock.cpp alongside the sample. `createInferRefitter_INTERNAL`
  added as a null-returning stub (the sample's DEFINE_TRT_ENTRYPOINTS references
  it even though refit is unused).
- Real MNIST data: tools/gen_mnist_pgms.py picks the best-classified test image
  per digit and writes the inverted PGM the sample expects (input = 1 - pgm/255).

The common/ closure solved here (compat headers + utils sources) is the same one
trtexec needs, so Phase 2 reuses it.

## Phase 2a: fp16 + serialize + real model (DONE)

Validated on gfx90a via tools/trtshim_run.cpp (flexible driver: --fp16, --save,
--load):
- fp16: kFP16 flag -> migraphx::quantize_fp16; correct argmax on mnist_cnn and
  on ResNet-50 (resnet50-v2-7, 102MB; argmax 556 matches onnxruntime CPU in both
  fp32 and fp16).
- Cross-process serialize round-trip: --save writes the engine blob to disk,
  --load in a fresh process deserializes and runs correctly (ResNet-50 and
  mnist_cnn). The magic header cleanly rejects a non-shim blob ("not a
  trt-shim-rocm engine").
- ResNet-50 needs default_dim_value=1 (dynamic batch); set via migraphx::
  onnx_options in backend introspect/build.

ctest is 6/6 (migraphx_smoke, driver_smoke, sampleOnnxMNIST, trtshim_run_fp16,
trtshim_run_save, trtshim_run_load). ResNet-50 is validated locally but not committed
(too large); tools/make_resnet_golden.py regenerates its golden via onnxruntime.

## Phase 2b/3: breadth sweep + trtexec deferral

Model breadth sweep (tools/sweep_models.sh): four distinct CNN architectures
built and run through the shim on gfx90a, each matching the onnxruntime CPU
argmax -- proving op/architecture coverage:
- ResNet-50 (residual) argmax 556
- SqueezeNet 1.1 (fire modules) argmax 111
- MobileNet v2 (depthwise-separable conv) argmax 78
- GoogLeNet/Inception (multi-branch concat) argmax 885
4/4 match. Models are downloaded to /tmp, not committed.

trtexec is DEFERRED (data/deferred.py: trt-shim-rocm-trtexec). It requires
porting the full TensorRT sample inference framework (sampleDevice/
sampleInference/sampleEngines), which needs a large CUDA-runtime surface
(cudaDeviceProp with CUDA struct layout, cudaMemGetInfo, cudaMallocManaged/Host,
cudaLaunchHostFunc, real cudaGraph capture, cudaProfiler) plus broad nvinfer1
optimization-profile/inference API the shim currently stubs. That is a
multi-day harness port, not a shim capability gap -- shim correctness is already
demonstrated by the stock sampleOnnxMNIST and the four-model sweep. Resume by
expanding include/compat for the device/graph surface and implementing real
IOptimizationProfile + enqueueV3 paths.

int8 calibration is not yet wired (BuilderConfig::setInt8Calibrator is stubbed);
it needs an IInt8Calibrator->migraphx::quantize_int8 bridge. Deferred follow-up.
(Note: an earlier draft cited MIGraphX issue #3585 as an int8 GPU-calibration
blocker. That was a mischaracterization from planning research -- #3585 is closed
as not-a-bug, a user misreading rocm-smi during compile-dominated runtime, not a
defect. quantize_int8 on the gpu target works; the real shim work is the
calibrator bridge.)

## Phase 3+: int8, dynamic shapes, conformance scoreboard (DONE)

- **int8 calibrator bridge:** BuilderConfig::setInt8Calibrator stored; build
  drives IInt8Calibrator::getBatch (app fills device pointers) -> host staging ->
  migraphx::quantize_int8_options::add_calibration_data -> quantize_int8 on the
  uncompiled program. backend.h's abstract CalibrationSource keeps backend.* free
  of TRT types; shim's ShimCalibSource implements it. Validated (trtshim_run --int8)
  on mnist_cnn and ResNet-50, argmax preserved.
- **Single-axis dynamic shapes:** createOptimizationProfile/setDimensions/
  addOptimizationProfile -> ShimOptimizationProfile derives the one dynamic axis
  (kMIN != kMAX) -> backend parses with set_default_dyn_dim_value and compiles a
  dynamic program. setInputShape threads the concrete shape into Engine::run
  (which builds the migraphx argument at that shape); engine reports -1 for
  dynamic dims. Validated (dyn_run) one engine at batch 1 and 2.
  - MIGraphX dynamic-shape codegen BUG on ResNet-50 (fused concat kernel arity)
    filed: findings/migraphx-trt-shim-gaps/ + deferred.py
    migraphx-dynamic-concat-codegen. Shim path itself is correct (concat-free
    dynamic model passes).
- **ONNX backend-test scoreboard** (tools/trt_infer.cpp + onnx_backend_scoreboard.py):
  drives all 1678 standardized onnx.backend.test node cases through the shim vs
  ONNX expected. Result: 895 pass, 757 fail, 26 skip; 96 operators fully green
  (Conv, MatMul, Gemm, Gather/Scatter, Layer/Group/Instance norm, activations,
  RNN/LSTM/GRU, QLinearConv, MatMulInteger). Failures are mostly
  MIGraphX-unsupported ops (Loop/Scan/sequence/training/bitwise) and shape ops
  whose node tests use runtime shape inputs (constant-shape works). Full table:
  src/test/onnx_scoreboard.md.

ctest is 8/8 (the small committed gates); the scoreboard + ResNet/sweep are
larger local validations (models downloaded, not committed).

MIGraphX feedback collected (findings/migraphx-trt-shim-gaps/report.md): (1)
dynamic-concat codegen bug, (2) no ONNX output-name accessor, (3) no buffer-based
program serialize/load. int8 confirmed working (not a gap).

## trtexec (DONE)

The stock unmodified NVIDIA trtexec (v10.7.0) runs on gfx90a through the shim:
--onnx, --fp16, --int8, --saveEngine, --loadEngine all PASS with normal
throughput/latency output. ctest is 9/9. (deferred.py trt-shim-rocm-trtexec ->
done.)

Three things beyond the sampleOnnxMNIST path:
- Larger include/compat surface (cudaDeviceProp->hipDeviceProp_t,
  getDeviceProperties/MemGetInfo, managed/host alloc, launchHostFunc, pointer
  attributes, stream-capture status, cuda_profiler_api.h stub).
- Runtime dlopen drop-in: trtexec dlopens libnvinfer.so.10 / libnvonnxparser.so.10
  and dlsyms the factories, so the build makes those VERSIONED aliases plus a
  no-op libnvinfer_plugin.so.10 (src/plugin_stub.cpp).
- More nvinfer1 methods: getNbOptimizationProfiles=1, *V2 format queries,
  setOptimizationProfileAsync, IExecutionContext getEngine/getTensorShape/
  getTensorStrides, and IStreamReader/IStreamReaderV2 deserialize (drain stream
  -> blob deserialize) for --loadEngine. TRTSHIM_DEBUG=1 surfaced each one.

Docs scrubbed of internal jargon (no "Strategy A"/MOAT) so the repo stands alone.
Layer-builder API: MIGraphX CAN back it (migraphx::module + operation), so it is
tractable-but-large, deferred until a code-built-network consumer needs it (the
ONNX path, which trtexec and all targets use, never touches it).

## Install as a dependency

Not applicable yet (no MOAT project depends on this). When the shim ships, this
section will document linking apps against the installed `libnvinfer.so` alias.
