# trt-shim-rocm notes

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

## Install as a dependency

Not applicable yet (no MOAT project depends on this). When the shim ships, this
section will document linking apps against the installed `libnvinfer.so` alias.
