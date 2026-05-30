# cudf jitify -> hipRTC porting plan (linux-gfx90a, MI250X, ROCm 7.2.1)

Scope: replace cudf's NVIDIA jitify (NVRTC-based runtime JIT) with ROCm hipRTC for
the binaryop / transform / rolling / stream_compaction-filter kernels. This plan is
to be executed AFTER libcudf core compiles (the core blockers -- cuCollections, the
`__CUDACC__`/`CUDF_HOST_DEVICE` macro cascade, the C++20 monolith scale -- are tracked
separately in notes.md and are NOT in scope here). All paths are at tag `v25.08.00`.

Primary finding up front: a standalone hipRTC proof-of-concept compiled a cudf-style
templated kernel from a source STRING at runtime, including one that pulls in
`<cuda/std/type_traits>` from the rmm-bundled libhipcxx, loaded the code object with
the HIP driver API, launched it on a gfx90a GCD, and verified the result. The
runtime-compile path is viable on this ROCm. Recommendation: port jitify to hipRTC
(not AOT) for the CUDA-source UDF path; the Numba-PTX UDF path cannot be carried to
AMD and must be disabled. See sections 3 and 4.

---

## 1. Map of cudf's JIT subsystem at v25.08.00

### 1.1 The pieces and how they fit

cudf has a runtime-JIT path used by exactly four algorithms, each of which accepts a
user-defined function (UDF) or PTX and must specialize a templated kernel for the
concrete column types at call time:

- binaryop (generic binary op from PTX),
- transform (row-wise UDF, CUDA or PTX),
- rolling (windowed UDF aggregation, CUDA or PTX),
- stream_compaction/filter (predicate UDF, CUDA or PTX).

The mechanism has three layers:

1. Build-time stringification (jitify_preprocess). Each JIT kernel `.cu` is run through
   the `jitify_preprocess` tool, which recursively resolves every `#include`, strips and
   stringifies the whole header closure, and emits a generated C++ header
   `<file>.cu.jit.hpp` that embeds the program text + all dependent headers as string
   literals (a `jitify2::PreprocessedProgramData`). These are baked into libcudf.so.
2. Runtime compile + cache. At call time cudf instantiates a template name (via
   `jitify2::reflection`), hands the preprocessed program + the instantiation + the
   user's UDF source override to a `jitify2::ProgramCache`, which calls NVRTC to compile,
   caches the result in-process and on disk, and returns a launchable kernel handle.
3. Launch. The handle is configured (`configure_1d_max_occupancy`) and launched
   (`->launch(...)`), which under the hood is `cuModuleLoadData` + `cuModuleGetFunction`
   + `cuLaunchKernel` on the NVRTC output.

### 1.2 Exact files and call sites

JIT core (`cpp/src/jit/`):
- `cache.hpp` / `cache.cpp` -- the program cache. `cache.hpp:30-43` declares
  `cudf::jit::program_cache` holding `std::unordered_map<std::string,
  std::unique_ptr<jitify2::ProgramCache<>>>`. `cache.cpp:115-139`
  (`program_cache::get`) lazily constructs a per-kernel `jitify2::ProgramCache<>` with
  process/disk cache limits from env vars (`LIBCUDF_KERNEL_CACHE_LIMIT_PER_PROCESS`,
  `..._DISK`). `cache.cpp:141-144` (`jit::get_program_cache`) is the entry point every
  call site uses; it routes through `cudf::get_context().program_cache()`. Disk cache
  dir logic: `cache.cpp:63-106` (`get_cache_dir`), keyed by `$HOME/.cudf/$CUDF_VERSION/<cc>`
  where `<cc>` is the CUDA compute capability from `cudaDeviceGetAttribute`.
- `helpers.hpp` -- reflection + device-marshalling helpers shared by transform/filter.
  `helpers.hpp:35-48` (`input_column_reflection`) builds an instantiation string via
  `jitify2::reflection::Template("cudf::jit::column_accessor").instantiate(type_name,
  index)`. `helpers.hpp:50-82` (`build_jit_template_params`) assembles the full template
  parameter list. `helpers.hpp:84-105` (`build_ptx_params`) maps params for the PTX path.
  `helpers.hpp:24` includes the generated `jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp`.
- `parser.hpp` / `parser.cpp` -- the PTX-to-CUDA translator. `parser.cpp:413`
  (`parse_single_function_ptx`) turns Numba-emitted PTX into a `__device__ __inline__`
  function whose body is inline `asm volatile("...ptx...")` (see `parse_instruction`,
  `parser.cpp` around 120-330, and the register-type maps `register_type_to_cpp_type`
  `parser.cpp:102` / `register_type_to_contraint` `parser.cpp:84`). `parser.cpp:423`
  (`parse_single_function_cuda`) is the much simpler CUDA-source path (it just renames
  the user's `__device__` function to the expected `GENERIC_*_OP` name).
- `util.cpp` / `util.hpp` -- `get_data_ptr(column_view/scalar)` (`util.cpp:58-66`), a
  type-dispatched raw-pointer extractor used to build launch argument lists.
- `accessors.cuh`, `span.cuh` -- device-side templated accessors
  (`column_accessor`, `span_accessor`, `scalar_accessor`, `device_optional_span`) that
  the generated kernels instantiate. These live in the stringified header closure.

Where NVRTC is actually invoked: cudf never calls NVRTC directly. The single call site
is inside jitify2 (`ProgramCache::get_kernel` -> `Program::compile` -> `nvrtcCompileProgram`),
in the header `${JITIFY_INCLUDE_DIR}/jitify2.hpp` fetched by
`cpp/cmake/thirdparty/get_jitify.cmake` (`rapidsai/jitify` branch `jitify2`). cudf's
contract with it is entirely through `jitify2::ProgramCache<>::get_kernel(name, opts,
header_overrides, compile_flags)` returning a `jitify2::Kernel`, then `->configure_*`
-> `->launch(...)`.

The four launcher call sites (the get_kernel/configure/launch chain):
- binaryop (PTX only): `cpp/src/binaryop/binaryop.cpp:160-166`
  `get_program_cache(*binaryop_jit_kernel_cu_jit).get_kernel(kernel_name, {},
  {{"binaryop/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})
  ->configure_1d_max_occupancy(0,0,nullptr,stream)->launch(out.size(), out_ptr, lhs_ptr,
  rhs_ptr)`. The instantiated name is built at
  `binaryop.cpp:154-159` via `reflection::Template("cudf::binops::jit::kernel_v_v").instantiate(...)`.
- transform: `cpp/src/transform/transform.cpp:42-49` (`get_kernel`) with flags
  `{"-arch=sm_.", "--device-int128"}` and header override
  `{{"cudf/detail/operation-udf.hpp", cuda_source}}`; built + configured at
  `transform.cpp:71-78` and `99-106`; launched at `transform.cpp:128` and `155`
  (`kernel->launch(args.data())` with a `std::array<void*,3>` arg pack).
- filter: `cpp/src/stream_compaction/filter/filter.cu:208-215` (`get_kernel`, same flags
  as transform), configured at `filter.cu:235-242`, launched at `filter.cu:178`.
- rolling: `cpp/src/rolling/detail/rolling.cuh:1286-1298`
  `get_program_cache(*rolling_jit_kernel_cu_jit).get_kernel(kernel_name, {},
  {{"rolling/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})
  ->configure_1d_max_occupancy(...)->launch(input.size(), ...)`. Name at
  `rolling.cuh:1278-1284`.

The JIT kernel source files (stringified at build time):
- `cpp/src/binaryop/jit/kernel.cu` (`kernel_v_v`, `kernel_v_v_with_validity`),
- `cpp/src/transform/jit/kernel.cu` (`kernel`, `fixed_point_kernel`, `span_kernel`),
- `cpp/src/rolling/jit/kernel.cu`,
- `cpp/src/stream_compaction/filter/jit/kernel.cu`.
Plus the runtime-overridable placeholders `cpp/src/binaryop/jit/operation-udf.hpp`,
`cpp/src/rolling/jit/operation-udf.hpp`, and `cpp/include/cudf/detail/operation-udf.hpp`
(empty stubs; the UDF source is injected as a header override at runtime).

The cmake jit-preprocessing step:
- `cpp/cmake/thirdparty/get_jitify.cmake` -- downloads jitify (`DOWNLOAD_ONLY`), sets
  `JITIFY_INCLUDE_DIR`.
- `cpp/cmake/Modules/JitifyPreprocessKernels.cmake` -- builds the `jitify_preprocess`
  executable from `${JITIFY_INCLUDE_DIR}/jitify2_preprocess.cpp` (line 16), defines
  `jit_preprocess_files()` (lines 23-58) which for each kernel runs `jitify_preprocess`
  with flags `-i -std=c++20 -remove-unused-globals -D_FILE_OFFSET_BITS=64
  -D__CUDACC_RTC__ -DCUDF_RUNTIME_JIT -I<cudf include> -I<cudf src> -I<libcudacxx>
  -I<CUDAToolkit> --no-preinclude-workarounds --no-replace-pragma-once`
  (lines 44-50), emitting `${CUDF_GENERATED_INCLUDE_DIR}/include/jit_preprocessed_files/<file>.cu.jit.hpp`.
  The list of preprocessed kernels is at lines 64-67. `add_custom_target(jitify_preprocess_run ...)`
  (lines 69-73) and `add_dependencies(cudf jitify_preprocess_run)` (`cpp/CMakeLists.txt:967`)
  wire it into the build. Include dirs for the build target: `cpp/CMakeLists.txt:904-916`
  (note `JITIFY_INCLUDE_DIR` and the generated dir are PUBLIC). Logging is silenced via
  `JITIFY_PRINT_LOG=0` (`cpp/CMakeLists.txt:924`); the cache option is `JITIFY_USE_CACHE`
  (`cpp/CMakeLists.txt:53`, applied at `926-935`).

### 1.3 The compile-flag set passed to NVRTC

Per call site, the flags handed to NVRTC are tiny: `{"-arch=sm_."}` (binaryop, rolling)
or `{"-arch=sm_.", "--device-int128"}` (transform, filter). `-arch=sm_.` is a jitify
idiom meaning "fill in the running device's compute capability". Everything else (the
`-std=c++20`, the `-I` include dirs, `-D__CUDACC_RTC__`, `-DCUDF_RUNTIME_JIT`,
`-remove-unused-globals`) is applied at the build-time PREPROCESS step, not at runtime:
jitify embeds the fully-resolved, header-injected program text into the `.jit.hpp` so the
runtime NVRTC compile sees a self-contained string and needs almost no flags or include
paths. This is the crux of the porting difference (section 2).

---

## 2. The hipRTC mapping

### 2.1 Call-for-call API map (NVRTC/jitify -> hipRTC)

| jitify2 / NVRTC operation                              | hipRTC / HIP driver equivalent |
|--------------------------------------------------------|--------------------------------|
| `nvrtcCreateProgram(&p, src, name, nHdr, hdrs, names)` | `hiprtcCreateProgram(&p, src, name, nHdr, hdrs, names)` (identical signature, `hiprtc.h:271`) |
| jitify reflection: `Template(name).instantiate(args)` (string build) + `nvrtcAddNameExpression(p, expr)` | build the same instantiation string + `hiprtcAddNameExpression(p, expr)` (`hiprtc.h`, symbol present) |
| `nvrtcCompileProgram(p, nOpts, opts)`                  | `hiprtcCompileProgram(p, nOpts, opts)` (`hiprtc.h:249`) |
| `nvrtcGetProgramLogSize` / `nvrtcGetProgramLog`        | `hiprtcGetProgramLogSize` / `hiprtcGetProgramLog` (`hiprtc.h:316,327`) |
| `nvrtcGetLoweredName(p, expr, &mangled)`               | `hiprtcGetLoweredName(p, expr, &mangled)` (`hiprtc.h:304`) |
| `nvrtcGetPTXSize` / `nvrtcGetPTX` (PTX text)           | `hiprtcGetCodeSize` / `hiprtcGetCode` (`hiprtc.h:338,349`) -- returns a gfx code object (ELF), NOT text |
| `cuModuleLoadData(&m, ptx)`                            | `hipModuleLoadData(&m, code)` (`hip_runtime_api.h:6531`) |
| `cuModuleGetFunction(&f, m, mangled)`                  | `hipModuleGetFunction(&f, m, mangled)` (`hip_runtime_api.h:6351`) |
| `cuLaunchKernel(...)` (what `jitify2::Kernel::launch` calls) | `hipModuleLaunchKernel(f, gx,gy,gz, bx,by,bz, shmem, stream, args, extra)` (`hip_runtime_api.h:6654`) |
| `configure_1d_max_occupancy` (jitify computes block/grid via occupancy API) | `hipModuleOccupancyMaxPotentialBlockSize` + `hipFuncGetAttributes`, or a fixed 256-thread block with grid-stride (cudf kernels are all grid-stride loops, so launch config only affects perf) |
| disk/in-proc program cache (`jitify2::ProgramCache`) | reimplement a small cache keyed on (preprocessed-program-name, instantiation, UDF hash); the code-object blob from `hiprtcGetCode` is what gets cached |

All required hipRTC and hipModule symbols are present in this ROCm (verified: `nm -D
/opt/rocm/lib/libhiprtc.so` lists `hiprtcCreateProgram`, `hiprtcAddNameExpression`,
`hiprtcCompileProgram`, `hiprtcGetCode`, `hiprtcGetLoweredName`, `hiprtcGetBitcode`;
`hip_runtime_api.h` declares the module/launch API).

### 2.2 Header dependency satisfaction: the central difference

jitify's model: at BUILD time, `jitify_preprocess` walks the kernel's `#include` closure
and stringifies every header into the `.jit.hpp`. At RUNTIME, NVRTC compiles a string
that already contains all its headers inline, plus a couple of runtime header overrides
(the UDF placeholder). So the runtime compile needs essentially no include path.

hipRTC's model: `hiprtcCreateProgram` takes `(numHeaders, headers, includeNames)` arrays
exactly like NVRTC -- so the SAME header-injection trick works. There are two viable ways
to feed the header closure to hipRTC:

- (a) Keep jitify_preprocess. It is a host-only C++ tool that stringifies headers; it does
  not run NVRTC and is not NVIDIA-specific in its core job. It CAN be retargeted to walk
  the libhipcxx/ROCm header closure instead of the libcudacxx/CUDAToolkit one (change the
  `-I` set in `JitifyPreprocessKernels.cmake:44-50` to libhipcxx + `/opt/rocm/include`,
  and drop `-D__CUDACC_RTC__` in favor of the HIP RTC define). Then continue to embed the
  closure into `.jit.hpp` and pass it to `hiprtcCreateProgram` via the header arrays. This
  keeps the diff localized to the cache/launcher shim plus the preprocess flags.
- (b) Drop preprocessing and pass `-I` include dirs to `hiprtcCompileProgram` directly.
  The PoC (section 3) shows this works: hipRTC's underlying comgr/clang resolves
  `#include <cuda/std/...>` and `#include <hip/hip_runtime.h>` from `-I` flags at runtime.
  This is simpler to stand up but ties the runtime to on-disk header trees (libhipcxx +
  ROCm headers must be present at the install location and discoverable), which is exactly
  what jitify's embedding was designed to avoid. It also re-parses the full header closure
  on every cold compile (slower first-call latency; the program cache hides warm calls).

Recommended: (a) for parity with upstream behavior and to keep headers self-contained in
the .so, but (b) is an acceptable first cut to get GPU validation fastest, since the PoC
proves it compiles. Either way the runtime MUST be given libhipcxx and `/opt/rocm/include`
(see PoC finding: libhipcxx's `cuda/std/detail/__config` `#include <hip/hip_runtime.h>`).

### 2.3 Name-expression / mangled-name handling

cudf builds the instantiation as a string (`jitify2::reflection::Template(name).instantiate(...)`),
where `type_to_name(data_type)` maps a cudf type to its C++ spelling. This string is a
valid C++ type-id and is passed verbatim to the name-expression API. hipRTC accepts it
through `hiprtcAddNameExpression` and returns the Itanium-mangled symbol via
`hiprtcGetLoweredName`. The PoC confirms this end to end: name expr
`kernel_v_v<float, int, float>` -> lowered name `_Z10kernel_v_vIfifEviPT_PKT0_PKT1_`,
which `hipModuleGetFunction` resolved. So the reflection layer
(`jitify2::reflection::Template`/`reflect`, used in only 9 + 2 spots across the codebase --
`grep` count) maps directly; only the thin string-building helper in `jitify2::reflection`
needs a replacement (a `cudf::jit` namespaced equivalent), and the mangled-name lookup is
a direct `hiprtcGetLoweredName` call.

### 2.4 The device-arch flag

NVRTC uses `-arch=compute_XX` (jitify's `-arch=sm_.` placeholder). hipRTC uses
`--offload-arch=gfx90a` (confirmed working in the PoC). The placeholder logic that fills
in the running device's capability should be replaced by querying
`hipGetDeviceProperties(...).gcnArchName` (PoC printed `gfx90a:sramecc+:xnack-`) and
passing `--offload-arch=<that>`. The per-device disk-cache key in `cache.cpp:78-86`
(which uses CUDA compute-capability ints) should likewise key on `gcnArchName`.

### 2.5 The genuinely hard parts (called out honestly)

1. The Numba-PTX UDF path is unportable. `parse_single_function_ptx` (`parser.cpp:413`,
   reached from binaryop.cpp:146, transform.cpp:63/92, filter.cu:228, rolling.cuh:1261)
   emits NVIDIA inline `asm volatile("...ptx...")`. PTX is NVIDIA ISA; AMD has no PTX and
   hipRTC will not assemble it. There is NO mechanical fix. The PTX path must be DISABLED
   on ROCm: the public `binary_operation(..., std::string const& ptx, ...)` overload and
   the `is_ptx==true` branches of transform/filter/rolling must throw a clear
   "PTX UDFs are not supported on ROCm" error (`CUDF_FAIL`). The CUDA-source UDF path
   (`parse_single_function_cuda`, `parser.cpp:423`) is portable and is what survives. This
   is the single biggest functional reduction: cudf-python's `@cuda.jit`/Numba-PTX UDFs
   will not run; string/CUDA-C++ UDFs will. (For most C++ libcudf callers and the gtest
   suite, the CUDA-source path is the one exercised; PTX is primarily the Python-Numba
   bridge.)

2. jitify's reflection + header-embedding model. The reflection part is easy (string
   building; PoC-proven). The header-embedding part is the real work: jitify_preprocess is
   tightly coupled to walking a CUDA/libcudacxx header tree with `__CUDACC_RTC__`. It must
   either be retargeted at the libhipcxx/ROCm tree (option (a) above) or bypassed in favor
   of runtime `-I` (option (b)). The kernel `.cu` files themselves use `CUDF_KERNEL`
   (cudf's `__global__` macro) and pull in `cudf/types.hpp`,
   `cudf/column/column_device_view*.cuh`, `cuda/std/*` -- so the whole
   `__CUDACC__`/`CUDF_HOST_DEVICE` HIP-awareness work (notes.md item, 46 guard sites) must
   already be done in the core for these headers to compile under hipRTC at all. This plan
   DEPENDS on that core work.

3. std / libhipcxx header availability under hipRTC. PoC finding: it works, but ONLY when
   both libhipcxx's include dir AND `/opt/rocm/include` are on the compile include set
   (libhipcxx pulls `<hip/hip_runtime.h>`). hipRTC does not auto-add `/opt/rocm/include`.
   This must be supplied explicitly (embedded headers in option (a), or `-I` in option (b)).
   libhipcxx is the rmm-bundled copy at `agent_space/libhipcxx/include` (and is installed
   into the rmm dependency prefix); the cudf build already consumes it.

4. Kernel name mangling: not hard. hipRTC's `hiprtcGetLoweredName` returns the correct
   Itanium mangling and `hipModuleGetFunction` consumes it (PoC-proven). No special handling
   beyond holding the lowered-name string for the lifetime of the lookup (it is owned by the
   program; copy it before destroying the program, as the PoC does).

5. cuco / thrust headers from hipRTC. The four JIT kernels do NOT include cuCollections and
   do NOT instantiate thrust algorithms inside the device kernel -- they include
   `cudf/detail/utilities/grid_1d.cuh`, `cudf/types.hpp`, `cuda/std/*`, the cudf accessors,
   and `cudf/strings/string_view.cuh` (transform/span). thrust appears only in the HOST
   launcher code (e.g. `thrust::counting_iterator` in helpers.hpp, `thrust::copy_if` in
   filter.cu), which is compiled by the normal HIP/host toolchain, NOT by hipRTC. So
   rocThrust/hipCUB usability under hipRTC is NOT on the critical path for these kernels.
   `cuda/std` (libhipcxx) IS, and the PoC shows it compiles. (If a future JIT kernel needed
   rocThrust device-side, that would be a separate and harder question; it does not arise
   for binaryop/transform/rolling/filter.)

---

## 3. Standalone hipRTC proof-of-concept

Location: `agent_space/cudf_hiprtc_poc/` (`poc.cpp`, `build_and_run.sh`).

What it does: mirrors cudf's binaryop JIT. It takes a templated grid-stride kernel as a
SOURCE STRING, registers a concrete instantiation as a name expression, compiles at runtime
with hipRTC for `--offload-arch=gfx90a`, retrieves the mangled (lowered) name and the code
object, loads it with `hipModuleLoadData`, resolves the function with `hipModuleGetFunction`,
launches with `hipModuleLaunchKernel`, and verifies the output on the host. Two programs are
compiled: [A] a self-contained kernel (no external headers), and [B] the same kernel with
`#include <cuda/std/type_traits>` + `cuda::std::common_type` to exercise libhipcxx under
hipRTC (the load-bearing case for real cudf kernels).

Build + run commands (exactly as executed):

```
cd /var/lib/jenkins/moat/agent_space/cudf_hiprtc_poc
/opt/rocm/bin/hipcc -O2 -std=c++17 poc.cpp -o poc -lhiprtc

# Stage A only (self-contained templated kernel):
HIP_VISIBLE_DEVICES=0 ./poc

# Stage A + B (B pulls in cuda/std from libhipcxx; include order matters --
# /opt/rocm/include first so libhipcxx's <hip/hip_runtime.h> resolves):
HIP_VISIBLE_DEVICES=0 ./poc /opt/rocm/include /var/lib/jenkins/moat/agent_space/libhipcxx/include
```

Captured output (Stage A + B run, the conclusive one):

```
hiprtc version 9.0
device 0: AMD Instinct MI250X / MI250 (gcnArch gfx90a:sramecc+:xnack-)
[A] compiled OK: code_object=5496 bytes, lowered_name=_Z10kernel_v_vIfifEviPT_PKT0_PKT1_
[A] launch + verify OK: out[1023]=1534.500000 (expected 1534.500000)
[B] compiled OK: code_object=5496 bytes, lowered_name=_Z10kernel_v_vIfifEviPT_PKT0_PKT1_
[B] launch + verify OK: out[1023]=1534.500000 (expected 1534.500000)
RESULT: PASS
```

Intermediate finding worth keeping (the first Stage B attempt, before adding
`/opt/rocm/include`):

```
/var/lib/jenkins/moat/agent_space/libhipcxx/include/cuda/std/detail/__config:36:10:
  fatal error: 'hip/hip_runtime.h' file not found
1 error generated when compiling for gfx90a.
[B] hiprtcCompileProgram FAILED: 6 (HIPRTC_ERROR_COMPILATION)
```

This is the concrete, early-surfaced header issue the porter must handle: hipRTC does not
implicitly know `/opt/rocm/include`; libhipcxx's config header needs the HIP runtime header,
so the runtime compile MUST be given `/opt/rocm/include` (and libhipcxx's include dir) either
as embedded headers (option (a) in 2.2) or as `-I` flags (option (b)). With that, it compiles
and runs.

Environment: ROCm 7.2.1, hiprtc version 9.0, AMD clang 22.0.0 (roc-7.2.1), 4x MI250X
(8 GCDs, all idle), gfx90a. PoC pinned to `HIP_VISIBLE_DEVICES=0`.

Conclusion: the runtime-compile -> load -> launch -> verify path is fully functional on this
ROCm, including the templated-kernel instantiation by name expression and the libhipcxx
header dependency. hipRTC is a viable jitify replacement for the CUDA-source UDF path.

---

## 4. Execution checklist (file-by-file) + hipRTC-vs-AOT assessment

### 4.1 hipRTC replacement, file by file

Prereq (NOT in this plan, tracked in notes.md): libcudf core compiles under HIP, and the JIT
kernels' header closure (`cudf/types.hpp`, `column_device_view*.cuh`, `grid_1d.cuh`,
`string_view.cuh`, `cuda/std/*`) is HIP-aware (`CUDF_HOST_DEVICE`/`CUDF_KERNEL` and the 46
`__CUDACC__`/`__CUDA_ARCH__` guard sites). hipRTC cannot compile what hipcc cannot.

1. `cpp/cmake/thirdparty/get_jitify.cmake` -- on `USE_HIP`, stop fetching NVIDIA jitify.
   Either remove the dependency (option (b)) or, if reusing jitify_preprocess (option (a)),
   keep the download but expect to retarget its include set.

2. `cpp/cmake/Modules/JitifyPreprocessKernels.cmake` -- option (a): change the preprocess
   `-I` set (lines 44-50) from libcudacxx + CUDAToolkit to libhipcxx + `/opt/rocm/include`,
   replace `-D__CUDACC_RTC__` with the HIP RTC define, and keep emitting `.cu.jit.hpp`.
   Option (b): replace this whole module with a trivial CMake step that stringifies each
   kernel `.cu` as plain text (no header walk) into a `<file>.cu.jit.hpp` exposing just the
   program name + source; headers come from `-I` at runtime. The build wiring
   (`add_dependencies(cudf jitify_preprocess_run)`, `cpp/CMakeLists.txt:967`) stays.

3. New file `cpp/src/jit/hiprtc_cache.cpp` (+ `.hpp`) OR rewrite `cpp/src/jit/cache.cpp` --
   provide `cudf::jit::get_program_cache(...)` returning an object exposing the SAME surface
   the call sites use: `.get_kernel(name, opts, header_overrides, flags)` -> handle with
   `->configure_1d_max_occupancy(...)->launch(...)`. Internally: `hiprtcCreateProgram`
   (with embedded headers in (a), or empty header arrays + `-I` flags in (b)),
   `hiprtcAddNameExpression(name)`, `hiprtcCompileProgram(--offload-arch=<gcnArchName>,
   -std=c++20, [-I...], -DCUDF_RUNTIME_JIT)`, `hiprtcGetLoweredName`, `hiprtcGetCode`,
   `hipModuleLoadData`, `hipModuleGetFunction`, `hipModuleLaunchKernel`. Keep an in-proc
   (and optional on-disk) cache keyed on (program name, instantiation, UDF source hash,
   gcnArchName). The PoC's `compile()` + `run_and_verify()` are a direct template for this.
   Replace the CUDA-compute-capability cache key (`cache.cpp:78-86`) with `gcnArchName`.

4. New file `cpp/src/jit/reflection.hpp` (or inline in the cache) -- replace the
   `jitify2::reflection::Template(...).instantiate(...)` and `reflect<T>()` calls (9 + 2
   sites) with a small `cudf::jit` string builder producing the identical type-id text.
   PoC shows the resulting string is accepted verbatim by `hiprtcAddNameExpression`.

5. `cpp/src/jit/cache.hpp` -- drop `#include <jitify2.hpp>`; change `program_cache` to hold
   the new hipRTC cache type instead of `jitify2::ProgramCache<>`.

6. Call sites -- update the four launchers to the new handle type (the chain shape is
   unchanged):
   - `cpp/src/binaryop/binaryop.cpp:136-167` (`binops::jit::binary_operation`). This is
     PTX-only -> on ROCm make it `CUDF_FAIL("PTX UDF binaryop not supported on ROCm")`, OR
     keep it only if the upstream Python path is converted to CUDA-source (it is not at
     v25.08). The public `binary_operation(...ptx...)` overload
     (`binaryop.cpp:357,442`) -> throw on ROCm.
   - `cpp/src/transform/transform.cpp:42-49,71-78,99-106,128,155` -- swap handle type;
     keep the CUDA-source branch (`transform.cpp:69,97`), throw on the `is_ptx` branch
     (`transform.cpp:63,92`).
   - `cpp/src/stream_compaction/filter/filter.cu:208-215,235-242,178` -- same: keep
     `parse_single_function_cuda` (`filter.cu:233`), throw on the PTX branch
     (`filter.cu:228`).
   - `cpp/src/rolling/detail/rolling.cuh:1255-1306` -- keep the `aggregation::Kind::CUDA`
     branch (`rolling.cuh:1266-1267`), throw on `aggregation::Kind::PTX`
     (`rolling.cuh:1259-1265`).

7. `cpp/src/jit/parser.cpp` / `parser.hpp` -- `parse_single_function_ptx`
   (`parser.cpp:413`) and its helpers become dead on ROCm. Leave the file (it is
   host-only, compiles fine), but gate the PTX entry points so they are unreachable / throw
   on ROCm. `parse_single_function_cuda` (`parser.cpp:423`) stays live and is needed.
   (Per CLAUDE.md orphan-cleanup: if the PTX helpers become entirely unreferenced on ROCm,
   `#ifndef USE_HIP`-guard them rather than delete, to keep the CUDA build intact.)

8. The kernel `.cu` files (`binaryop/jit/kernel.cu`, `transform/jit/kernel.cu`,
   `rolling/jit/kernel.cu`, `stream_compaction/filter/jit/kernel.cu`) and the device
   headers they pull in -- these compile under hipRTC unchanged IF the core HIP-awareness
   work is done. `CUDF_KERNEL` must expand to `__global__` (with launch-bounds) under HIP.
   The `operation-udf.hpp` placeholders (`binaryop/jit/operation-udf.hpp`,
   `rolling/jit/operation-udf.hpp`, `include/cudf/detail/operation-udf.hpp`) stay as-is
   (still overridden at runtime via the header-override mechanism).

9. `cpp/CMakeLists.txt` -- adjust the JIT include/define wiring
   (`904-916`, `923-935`, `967`): drop `JITIFY_INCLUDE_DIR` if option (b); ensure libhipcxx
   and `/opt/rocm/include` are visible to the runtime compile path; link `cudf` against
   `hiprtc` (e.g. `target_link_libraries(cudf PRIVATE hiprtc)`). The
   `_FILE_OFFSET_BITS=64`/`JITIFY_PRINT_LOG=0` defines become irrelevant if jitify is gone.

10. Tests -- `cpp/tests/transform/*`, `cpp/tests/binaryop/*`, `cpp/tests/rolling/*`,
    `cpp/tests/stream_compaction/*`, and `cpp/tests/jit/parse_ptx_function.cpp`. The
    PTX-input tests must be expected-skip/expected-throw on ROCm; the CUDA-source UDF and
    the compiled (non-JIT) binaryop tests are the real GPU validation. Validation = run
    these on a gfx90a GCD (`HIP_VISIBLE_DEVICES=0`) and confirm pass, plus a transform/filter
    UDF test that actually triggers a hipRTC compile (to exercise this subsystem, not just
    the compiled path).

### 4.2 FALLBACK: AOT instantiation instead of JIT

The alternative is to NOT compile at runtime at all: instantiate the kernel templates
ahead of time for the common type combinations and link them into libcudf.so, dispatching
by runtime type.

Feasibility of AOT here:
- binaryop generic/PTX path: AOT-able only for the BUILT-IN ops, but that already exists --
  cudf has a separate fully-compiled binaryop (`binops::compiled`, `binaryop.cpp:170-231`,
  the `cpp/src/binaryop/compiled/` tree) used for all non-UDF ops. The JIT binaryop is
  SOLELY the user-supplied-PTX path. AOT cannot instantiate an unknown user function, so
  AOT does not replace the JIT binaryop -- it just means the UDF binaryop is dropped (which
  is the same outcome as disabling the unportable PTX path).
- transform / filter / rolling: the WHOLE point is a user-supplied UDF
  (`GENERIC_TRANSFORM_OP`/`GENERIC_FILTER_OP`/the rolling operator) injected at runtime.
  There is no finite "common type combination" set to instantiate because the UDF BODY is
  unknown until call time. AOT fundamentally cannot express "run the user's arbitrary
  device function". AOT could only support a small built-in set of operations, which is not
  what these APIs are.

Honest weighing:
- hipRTC pros: preserves the actual feature (arbitrary CUDA-C++ UDFs), PoC-proven to work on
  this ROCm, localized diff (cache/launcher shim + flag/arch changes + PTX disable), matches
  upstream behavior and tests for the CUDA-source path. Cons: adds a runtime dependency on
  libhiprtc + on-disk headers (option (b)) or a retargeted preprocess step (option (a));
  cold-compile latency (mitigated by the program cache); PTX-UDFs still lost.
- AOT pros: no runtime compiler, no header-shipping, simplest deploy. Cons: it does NOT
  implement the feature -- it can only cover built-ins, so transform/filter/rolling UDFs and
  the JIT binaryop would all effectively be removed, not ported. That is a strictly larger
  functional regression than hipRTC (which keeps CUDA-source UDFs) for MORE engineering on
  the dispatch tables.

Recommendation: use hipRTC. It is the only option that actually ports the feature, the PoC
proves it runs on gfx90a, and the change is contained. Pair it with disabling the
Numba-PTX path (unportable regardless of hipRTC-vs-AOT). AOT is not a real substitute here
because the kernels exist precisely to run runtime-unknown user code; "AOT" would mean
deleting the UDF capability. Reserve AOT only as the explanation for why the JIT binaryop's
loss is acceptable (the compiled binaryop already covers all built-in ops).

### 4.3 Top risks for the porter

1. Core prerequisite not done. hipRTC compiles the kernel header closure
   (`cudf/types.hpp`, `column_device_view*.cuh`, `cuda/std/*`) the SAME way hipcc does. If
   the `__CUDACC__`/`CUDF_HOST_DEVICE`/`CUDF_KERNEL` HIP-awareness (notes.md, 46 guard
   sites) is incomplete, the runtime compile fails with the identical macro-cascade errors
   seen in the core probe. This plan is blocked on that; do it first.
2. Runtime header availability. hipRTC does not auto-add `/opt/rocm/include`; libhipcxx's
   `__config` needs `<hip/hip_runtime.h>` (PoC-confirmed failure, then fix). The chosen
   approach (embed via retargeted jitify_preprocess, or `-I` at runtime) must reliably
   locate libhipcxx + ROCm headers at the INSTALLED location, not just in the build tree,
   or JIT calls break for installed consumers (cugraph/cuml).
3. Silent loss of PTX UDFs. Disabling `parse_single_function_ptx` removes Numba/cudf-python
   PTX-UDF support across binaryop/transform/rolling/filter. This must be a clear,
   documented `CUDF_FAIL` (not a crash or a wrong result), and the PTX-input tests
   converted to expected-throw, so callers get a precise message rather than a mysterious
   gfx90a assembler failure.
