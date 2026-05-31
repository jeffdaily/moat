# arrayfire notes

ArrayFire 3.10.0, pure-CMake multi-backend array library. Port shape: a NEW sibling
backend `src/backend/hip` cloned from `src/backend/cuda` (additive; the CUDA/NVIDIA path
stays byte-for-byte). The HIP backend reports `AF_BACKEND_CUDA` so the unified dispatcher and
the 139 gtest binaries treat it as the "cuda" backend (no public ABI/enum change). Lead
platform linux-gfx90a (MI250X, ROCm 7.2.1).

Fork: https://github.com/jeffdaily/arrayfire (branch `moat-port`; `master` stays a clean
upstream mirror). Actions disabled on the fork. base_sha 492718b.

## Install as a dependency
N/A -- arrayfire is a leaf (no other MOAT project depends on it). It vendors only
`extern/half` and finds host/GPU libraries externally; no inter-project MOAT deps.

## Phase 0 (GO/NO-GO): hipRTC runtime-JIT -- RESULT: GO

The make-or-break risk was arrayfire's bespoke NVRTC + CUDA-Driver-API runtime-JIT engine
(`src/backend/cuda/compile_module.cpp`). On NVIDIA it does
`nvrtcCompileProgram(--gpu-architecture=compute_XX)` -> `nvrtcGetPTX` ->
`cuLinkCreate`/`cuLinkAddData(CU_JIT_INPUT_PTX)`/`cuLinkComplete` -> `cuModuleLoadData`. The
question was whether the hipRTC analogue (direct code object, no PTX, no cuLink, `--offload-arch`)
works for arrayfire's TWO JIT paths.

Standalone repro `agent_space/af_hiprtc_poc/af_poc.cpp` (built with
`/opt/rocm/bin/hipcc -O2 -std=c++17 af_poc.cpp -o af_poc -lhiprtc`, run
`HIP_VISIBLE_DEVICES=0 ./af_poc`) reproduces both arrayfire JIT paths with arrayfire's actual
constructs and runs them end to end on gfx90a. RESULT: PASS.

- JIT path (`sourceIsJIT=true`): an `extern "C" __global__` element-wise kernel carrying the
  `kernel/jit.cuh` preamble -- `typedef float2 cuFloatComplex;` / `double2`, `#include <cuda_fp16.h>`,
  `__device__` complex helpers (`__caddf`/`__cmulf`...), and an integer-round intrinsic
  (`__double2ll_rn`). hipRTC compiles it to a 5848-byte code object; `hipModuleLoadData` +
  `hipModuleGetFunction("af_jit_kernel")` + `hipModuleLaunchKernel` + host verify all pass.
- Templated path (`sourceIsJIT=false`): a templated kernel instantiated by name expression
  via `hiprtcAddNameExpression` / `hiprtcGetLoweredName` (lowered name
  `_Z15af_templ_kernelIfLi3EEviPT_PKS0_S3_`), loaded and launched, host verify passes. (This
  matches the already-proven cudf hipRTC PoC, re-confirmed in arrayfire's shape.)

The flow `hiprtcCompileProgram(--offload-arch=gfx90a)` -> `hiprtcGetCodeSize`/`hiprtcGetCode`
-> `hipModuleLoadData` -> `hipModuleGetFunction` -> `hipModuleLaunchKernel` is fully functional.
The arch flag is derived from `hipGetDeviceProperties(...).gcnArchName` with the
`:sramecc+:xnack-` feature suffix stripped (`gfx90a`).

### Three concrete arrayfire-specific hipRTC deltas (all fixed in the repro; none a blocker)

1. **Empty-string ("") header source is REJECTED by `hiprtcCreateProgram` (INVALID_INPUT)**,
   where NVRTC accepts it. `compile_module.cpp` passes several `string("")` dummy header
   bodies ("DUMMY ENTRY TO SATISFY ..." for math.h/stdbool.h/stdlib.h/vector_types.h/utility).
   On HIP each dummy header source must be NON-EMPTY (a `"/* empty */\n"` placeholder works;
   `min4.cpp` mode 1 vs mode 2 isolates this exactly). A NULL header pointer aborts in the
   std::string ctor.
2. **`--device-as-default-execution-space` is REJECTED by hipRTC** ("unknown argument").
   arrayfire passes it unconditionally (`compile_module.cpp:275`). Drop it on HIP -- HIP makes
   device the default execution space for `__global__`/`__device__` via `__HIP_DEVICE_COMPILE__`
   anyway. Same for `--gpu-architecture=compute_XX` -> `--offload-arch=gfx90a`, and the
   `--device-debug`/`--generate-line-info` NVRTC debug flags (drop or map).
3. **An embedded header whose body does `#include <hip/...>` needs `-I/opt/rocm/include`** as a
   runtime compile option (hipRTC does not implicitly know the ROCm include path). This is the
   cudf lesson; the cleanest model is to inject `cuda_fp16.h` (the name arrayfire's JIT source
   `#include`s) as a body of `#include <hip/hip_fp16.h>` and pass `-I/opt/rocm/include`. NOTE:
   hipRTC also provides `__half`/`__float2half`/`__half2float` AND `float2`/`double2` as
   builtins with NO header at all (`min2.cpp` modes 2,3), so the fp16/vector-type headers may
   even be droppable; the `-I/opt/rocm/include` + injected-header model is the conservative
   choice that keeps the source strings unchanged.

Repro files: `agent_space/af_hiprtc_poc/{af_poc.cpp, RESULT.log}` (full repro) and the
isolation probes `min.cpp`/`min2.cpp`/`min3.cpp`/`min4.cpp`. ROCm 7.2.1, hiprtc version 9.0,
AMD clang (roc-7.2.1), 4x MI250X (GCDs 0-3), pinned to GCD 0.

## JIT engine restructure (compile_module.cpp on HIP)

Given Phase 0 GO, the port of `compile_module.cpp` is:
- `nvrtc.h` -> `hip/hiprtc.h`; `nvrtc*` -> `hiprtc*`; `NVRTC_SUCCESS` -> `HIPRTC_SUCCESS`.
- Drop the entire PTX + cuLink block (`nvrtcGetPTXSize`/`nvrtcGetPTX`, `cuLinkCreate`/
  `cuLinkAddData(CU_JIT_INPUT_PTX)`/`cuLinkComplete`, `cuLinkDestroy`). Replace with
  `hiprtcGetCodeSize`/`hiprtcGetCode` into a `vector<char>` and `hipModuleLoadData(&mod, code)`.
- Arch: replace `--gpu-architecture=compute_%d%d` with `--offload-arch=<gcnArchName-stripped>`.
- Drop `--device-as-default-execution-space` and the NVRTC `--device-debug`/`--generate-line-info`.
- Empty dummy header bodies -> non-empty placeholder; add `-I/opt/rocm/include` to the JIT
  compile options (or to the non-JIT options) so injected `hip/...` includes resolve.
- The disk cache stores the code-object blob instead of cubin (key on gcnArchName instead of
  compute capability). Keep the deterministicHash integrity check.
- `cuModuleGetFunction`/`cuLaunchKernel` -> `hipModuleGetFunction`/`hipModuleLaunchKernel` (these
  are in Kernel.hpp/jit.cpp, the runtime side).

## Build commands (gfx90a, headless)
See plan.md "Build commands". Cap `-j 16`. CPU backend stays ON as the in-process reference for
the gtest CPU-vs-GPU diffs.

## Validatable core (first `ported`)
arith/JIT, reduce/scan/ireduce, sort/sort_by_key/set, blas/dot/matmul, fft/fft_real,
random/rng_quality, transpose/reorder/join/moddims, index/assign/lookup. Deferred (documented):
cudnn-gated convolveNN (MIOpen), graphics/imageio (Forge/FreeImage), nonfree sift/gloh, the CV
kernels (fast/harris/orb/sift).

## Session 1 progress (porter) -- what landed, what remains

Fork HEAD pushed: jeffdaily/arrayfire @ moat-port, commit 260285364a006 ([ROCm] Add
HIP backend scaffold; port runtime-JIT engine to hipRTC). 372 files (the cloned backend +
two CMake edits), 996 lines of genuine port diff on top of the byte-identical clone.

LANDED and VERIFIED (GPU on gfx90a, GCD 0):
- Phase 0 hipRTC go/no-go (above) -- GO. Both arrayfire JIT paths proven end to end.
- compile_module.cpp engine restructure (NVRTC+PTX+cuLink -> hipRTC+code-object). The exact
  ported call sequence re-verified standalone (agent_space/af_hiprtc_poc/engine_check.cpp:
  hiprtcCreateProgram with non-empty headers -> --offload-arch=<stripped gcnArchName> ->
  hiprtcGetCode -> hipModuleLoadData, with the ROCM_PATH fallback). PASS.
- kernel/shfl_intrinsics.hpp wave64 fix (64-bit FULL_MASK + _sync intrinsics). Compiled and
  launched on gfx90a (agent_space/af_hiprtc_poc/shfl_check.hip). PASS.
- nvrtc_shims/ (cuComplex.h/cuda_fp16.h(.hpp)/math_constants.h/vector_types.h/
  vector_functions.h HIP substitutes); hip_compat.h (driver/runtime/complex aliasing);
  Module.hpp include swap; kernel/config.hpp kWarpSize abstraction.
- Top-level CMake AF_BUILD_HIP wiring + test harness "cuda"-tag; src/backend/hip/CMakeLists.txt
  wires the JIT-header codegen with the HIP shims.

NOT YET DONE (the remaining work before the backend compiles + the gtest core GPU-validates).
The honest next-wall, in dependency order:

1. device_manager.cpp (~60 CUDA refs) -- the heaviest single file. NVML device enumeration ->
   rocm-smi / hipDeviceGetAttribute; the cublas/cusolver/cufft VERSION queries and the
   compute-capability->cores table (compute2cores) need a gfx-arch analogue or stub; cudnn
   refs are AF_WITH_CUDNN-gated (OFF). platform.cpp (~20 refs) getComputeCapability /
   getDeviceProp(cudaDeviceProp->hipDeviceProp_t, has .gcnArchName/.warpSize/.major/.minor).
2. The CUDA-library swap files (~26), each a focused per-file translation with enum/signature
   deltas (PORTING_GUIDE library-swap lessons apply directly):
   - blas.cu/cublas.cpp/cublas.hpp -> hipBLAS (+ hipBLASLt for the Lt path; watch hipBLAS v2
     enums). solve.cu/cusolverDn.cpp/.hpp + cholesky/lu/qr/svd/inverse.cpp -> hipSOLVER.
   - cufft.cu/cufft.hpp/fft.cu/fftconvolve -> hipFFT (MPPI: status-enum non-1:1, guard orphan
     case labels in any cufftGetErrorString-style switch).
   - sparse*.cu/cusparse*.cpp/cusparse_descriptor_helpers.hpp -> hipSPARSE generic api (amgcl:
     prefer hipSPARSE over rocSPARSE for the persistent-descriptor generic api; watch the
     several-void*-typedef descriptor aliasing gotcha in cusparse_descriptor_helpers.hpp).
   - sort*.cu/set.cu/ThrustAllocator.cuh/ThrustArrayFirePolicy.hpp -> rocThrust/hipCUB. rmm:
     swap <thrust/system/cuda/...> -> <thrust/system/hip/...>, thrust::cuda::par ->
     thrust::hip::par, thrust::cuda_cub -> thrust::hip_rocprim in the custom policy.
   - random_engine.cu is mostly own device-code (Philox/Threefry/MT), not a curand swap.
3. Finish src/backend/hip/CMakeLists.txt: the afhip add_library with the full source list
   (mirror src/backend/cuda minus cuDNN/Forge/static-CUDA), set_source_files_properties(
   <.cu/.cuh kernel TUs> LANGUAGE HIP), HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}",
   force-include hip_compat.h on every HIP TU (target_compile_options -include
   .../hip_compat.h), link hip::host/hipblas/hipfft/hipsolver/hipsparse + ${HIPRTC_LIBRARY},
   and install/export as afcuda. Build cmd in plan.md; cap -j 16.
4. reduce.hpp / reduce_by_key.hpp wave64 staging (DEFERRED ON PURPOSE -- do with GPU
   validation, not blind). The kernels use cub::WarpReduce (hipCUB's is wave-width-aware) but
   surround it with hardcoded `tidx<32`, `s_ptr[tidx+32]`, `nwarps=THREADS/32`, `tid/32`,
   `tid%32`, `tid<32`, `==31`. Rework to kWarpSize (now in config.hpp) so the staging is
   correct on wave64; the s_val[nwarps] sizing and the second-stage `if(tid<32)` fold are the
   hazards. Validate with the `reduce`/`scan`/`ireduce` gtests AND a fixed-seed run-to-run
   determinism check (MPPI lesson) once the backend builds -- applying it unvalidated risks a
   per-arch hack that ping-pongs platforms.
5. textures: LookupTable1D.hpp + interp paths -- rule-of-five on hipTextureObject_t, and the
   popsift/gpuRIR float-element-read linear-filter rejection (manual lerp) if any bind uses
   cudaFilterModeLinear. Enumerate exactly when reaching the index/lookup/approx ops.
6. Deferred features (documented, not core): convolveNN/cudnn -> MIOpen (AF_WITH_CUDNN=OFF for
   the first build), graphics/imageio (Forge/FreeImage), nonfree sift/gloh, CV kernels
   (fast/harris/orb/sift/regions/homography/nearest_neighbour).

Then: build for gfx90a headless (plan.md build cmd, -j 16), iterate compile errors file by
file, and GPU-validate the core gtest subset (jit/arith/reduce/scan/sort/set/blas/fft/random/
transpose/index) on one isolated GCD via `ctest -R '^CUDA\.' --output-on-failure` run serially.
Only then -> `ported`.

## Session 2 FINAL (porter) -- afhip BUILDS + LINKS; core ops GPU-validated; JIT-half wall

Fork HEAD: jeffdaily/arrayfire @ moat-port = 96edd3a. The full afcuda shared
library AND all 132 CUDA-tagged gtest binaries BUILD and LINK for gfx90a. Build:
`bash agent_space/af_build-hip.sh` (config + -j16, copy it into src/ first; it is
intentionally NOT committed to the fork). State stays `porting` (the core is not
fully green yet -- transpose/scan/fft and some blas fail; see the wall below).

GPU-validated on gfx90a (GCD 0, CUDA.* prefix), PASSING:
- jit 1779/1781 (the hipRTC runtime-JIT engine -- the make-or-break component).
- reduce sum/min/max/all/any 120/120 (after the wave64 reduce.hpp rework).
- random 513/513, set 37/37, dot 51/51, moddims 39/51, reorder 233/334,
  sort 71/133, join 43/120, assign 179/397 (partial; the rest hit the JIT-half
  wall or other edge cases).

NOT yet passing (the precise remaining walls, in priority order):
1. (BIGGEST) JIT ops whose runtime source pulls common/half.hpp's half<->int
   CONVERSIONS fail to hipRTC-compile -> AF_ERR_INTERNAL at runtime. Affects
   transpose (25/140), scan (0/50), fft (5/103), and the half paths of many
   ops. Root: common/half.hpp's half2int / int2half device branch is gated on
   __CUDA_ARCH__ and calls __half2short_rn / __half2ll_rn intrinsics. The JIT
   compile now defines __CUDACC_RTC__ + __CUDA_ARCH__=900 + __HIP_RTC__ and a
   minimal std shim (is_integral/is_signed/integral_constant/numeric_limits<int
   types>+float was ADDED to the RTC branch), which got past the
   af/defines.h->compilers.h cascade, stddef.h, AF_CONSTEXPR-constexpr, and the
   missing-traits errors. The CURRENT error is "call to __device__ function
   __half2short_rn from __host__ function": half2int is AF_CONSTEXPR (=__host__
   __device__ under non-RTC, but under hipRTC it is host+device) and clang's
   hipRTC host parse cannot call the __device__-only intrinsic. Likely fix:
   under __HIP_RTC__ make half2int / int2half / the half ctors strictly
   __device__ (hipRTC compiles only device code, so host attributes are the
   problem), or provide __device__ __host__ constexpr software conversions on
   the hipRTC path instead of the intrinsics. This is the gateway to
   transpose/scan/fft and the bulk of the remaining core.
2. blas matmul: the FullOut gemm cases pass but a SubarrayOut (strided) gemm
   case GPU-faults (core dump). Investigate the strided/sub-array hipBLAS path
   (leading dims / the POD-complex reinterpret at the gemm boundary).
3. reduce-by-key / ragged-reduce: still wrong after the maxResPerWarp=kWarpSize
   fix (segmented-scan correctness on wave64 needs more work; the basic
   sum/min/max/all/any are correct). These are a narrower advanced feature.
4. The 2 JIT failures (JIT.CPP_common_node, evaluateBothArrayAndItsTranspose):
   a JIT source that #includes <stdlib.h> still hits the real /usr/include one
   under hipRTC (the dummy header is not intercepting the <> include).
5. Deferred (stubbed/off, documented): sparse (hipSPARSE generic-api port),
   nearest_neighbour + CV kernels (fast/harris/orb/sift), convolveNN (MIOpen).

Key landed fixes this session (all in the curated commit):

## Session 2 progress (porter) -- afhip now compiles ~all core TUs; 2 walls remain

Built on the Session 1 uncommitted work (the ~26-file library swaps + device_manager/platform
were done but never committed). This session committed them + the cuSOLVER macro-body fix +
the full afhip add_library + the compile-error iteration. The backend now configures and
compiles the large majority of the core TUs (the JIT engine, thrust-sort TUs, hipBLAS/
hipSOLVER/hipFFT swaps, reductions, device_manager, platform). Build: bash
projects/arrayfire/src/build-hip.sh (config + -j16); CPU backend ON, FORGE/CUDNN/IMAGEIO OFF.

LANDED this session (all on the moat-port branch):
- cuSOLVER: hipSOLVER's cuSOLVER-compatible hipsolverDn* API matches cuSOLVER signature-for-
  signature (verified in /opt/rocm/include/hipsolver/internal/hipsolver-dense.h: Dnpotrf/
  getrf/geqrf/gesvd/ormqr/getrs). The Session 1 typedef renames left the macro BODIES calling
  cusolverDn##X##...; fixed to hipsolverDn##X## in cholesky/lu/qr/svd/solve.
- afhip CMakeLists.txt: full add_library(afcuda ...) mirroring src/backend/cuda (minus
  cudnn(gated)/sparse-plugin), include(scan_by_key + thrust_sort_by_key sub-CMakeLists),
  .cu LANGUAGE HIP, HIP_ARCHITECTURES from ${CMAKE_HIP_ARCHITECTURES}, hip::host + roc::hipblas
  /hipsolver/hipsparse + hip::hipfft + ${HIPRTC_LIBRARY} links, install/alias as afcuda.
- The detail-namespace collision (THE rocThrust gotcha, below): backend.hpp now makes `detail`
  a REAL namespace re-exporting arrayfire::cuda on HIP (merges with rocprim's global
  `namespace detail`) instead of a colliding alias.
- hipcub include + namespace mapping for the 5 cub:: consumers (reduce/reduce_by_key/topk/
  fast/reduce_impl).
- hip_compat.h greatly extended: device-management (cudaDeviceProp->hipDeviceProp_t,
  Get/SetDevice, peer access, version), cuGetErrorName/String -> hipDrvGetError* (two-arg
  form), textures, GL-interop, cudaPeekAtLastError, etc. Force-included on HIP AND CXX TUs
  (the host .cpp call the CUDA runtime/driver API directly, as on CUDA via nvcc).
- nvrtc_shims/: added cuda_runtime_api.h, cuda_gl_interop.h forwarding shims.
- rocThrust policy: ThrustArrayFirePolicy.hpp -> thrust::hip::execution_policy + thrust::
  hip_rocprim (was thrust::cuda::/cuda_cub) + __HIP_DEVICE_COMPILE__ guards; thrust_utils.hpp
  and regions.hpp <thrust/system/cuda/...> -> <thrust/system/hip/...>.
- math.hpp: the __CUDA_ARCH__ host/device split (minval/maxval, __half min/max/abs, is_nan)
  now also keys on __HIP_DEVICE_COMPILE__ (cudaKDTree/gsplat fault class -- __CUDA_ARCH__ is
  NOT defined in the HIP device pass, so the host-only versions were being pulled into device
  code); the cfloat/cdouble is_nan specializations got the missing __DH__ (gsplat target-attr
  rule). types.hpp: `using half = common::half;` inside namespace cuda (HIP's hip_fp16.h does
  `using half=__half;` at GLOBAL scope, making bare `half` ambiguous).
- blas.cu: __half gemm/gemmBatched function-pointer reinterpret_cast (hipBLAS spells the half
  element hipblasHalf=uint16_t, not __half). fft.cu: cufftExec##/CUFFT_## macro-paste ->
  hipfftExec##/HIPFFT_##.
- reduce_by_key.hpp: full wave64 staging via kWarpSize (was hardcoded 32: laneid%32, nWarps=
  DIMX/32, ==31, the single-key fast-path butterfly shfl 1..16, the per-warp-size warp-scan
  with a 32-bit activemask + raw __shfl_up_sync). AutoDock-GPU axis (native 64-lane warp,
  partials recombine via a per-warp-count scan). MUST be GPU-validated (reduceByKey/scanByKey
  + determinism) before trusting -- applied but NOT yet run.
- sparse: DOCUMENTED DEFERRAL. sparse.cu/sparse_arith.cu/sparse_blas.cu replaced with
  AF_ERR_NOT_SUPPORTED stubs (keeping the exact public signatures + instantiations so the
  monolithic afcuda links); cusparse.cpp/cusparseModule.* + cusparse_descriptor_helpers.hpp
  dropped from the build; platform.cpp's sparseHandle() uses a direct tag-keyed hipsparse
  handle (no dlopen plugin). The cuSPARSE generic-api + dlopen-module port (amgcl lessons) is
  the next follow-on after the core.

TWO REMAINING COMPILE WALLS (precise):
1. (CORE) Complex operator* / operator/ ambiguity. cfloat = cuFloatComplex -> (hip_compat
   alias) hipFloatComplex = HIP_vector_type<float,2>, which ships FRIEND operator* and
   operator/ (componentwise) that TIE with arrayfire's namespace-scope complex operator*/
   operator/ (math.hpp BINOP) at every cfloat*cfloat / cfloat/cfloat site. CUDA's cuComplex is
   a plain struct with no operators, so only arrayfire's exist there. Surfaces in
   common/Binary.hpp (lhs*rhs for Binary<cfloat,af_mul_t>) -> product.cu, and BINOP_SCALAR.
   The += / -= are member-only on HIP (no binary friend), so operator+ / operator- do NOT
   collide; only * and /. There is NO opt-out macro for HIP's vector operators
   (amd_hip_vector_types.h, friends are unconditional), and the semantics differ (complex vs
   componentwise) so arrayfire's cannot be dropped. The clean fix is to make cfloat/cdouble
   NOT be HIP_vector_type on HIP -- either a plain POD {x,y} struct (then re-provide the cuC*
   helpers + hipBLAS/hipFFT conversions, broad ripple) or a struct DERIVED from hipFloatComplex
   (friends are not inherited, so only arrayfire's operators apply, and it slices to the base
   for cuC*/library calls -- verify aggregate-init {x,y} and hipBLAS pointer ABI). Affects
   complex product-reduce and complex element-wise mul/div in the compiled path (blas/fft go
   through hipBLAS/hipFFT, unaffected).
2. (DEFERRED, CV) kernel/nearest_neighbour.hpp maxval<To>() no-match + kernel/convolve.hpp-class
   issues -- CV/feature kernels (fast/harris/orb/sift/nearest_neighbour), already an explicit
   follow-on. Exclude these .cu from the core source list (or stub) if they block the core
   link; they are not in the validatable core set.

Also seen and fixed: hip::device INTERFACE_COMPILE_OPTIONS inject `-x hip --offload-arch`
scoped to $<COMPILE_LANGUAGE:CXX>, which forces EVERY host .cpp of a target linking hip::device
through the HIP compiler -- link hip::host ONLY (the .cu get HIP via LANGUAGE HIP +
CMAKE_HIP_COMPILER). And: CMake `target_compile_options` `-include X` must use the SHELL:
prefix ("SHELL:-include ${path}") or the two tokens get merged/dropped.

Next session: resolve wall #1 (complex type), then iterate the remaining compile errors to a
full link, then GPU-validate the core gtest subset (the hip backend runs under the CUDA.* test
prefix) on one isolated GCD. Only then -> ported. reduce_by_key wave64 staging needs explicit
GPU determinism validation.

## Gotchas log
- Porting a runtime NVRTC JIT engine to hipRTC: the runtime COMPILE OPTIONS need four things
  beyond the symbol swap. (1) Per-token split each option string: NVRTC accepts a flag+arg
  joined by a space (" -D NAME=val", the common DefineValue macro form) as ONE option, but
  clang/hipRTC reads "-D NAME=val" as a filename -> "cannot specify -o when generating multiple
  output files". Tokenize on whitespace into separate argv entries. (2) hipRTC does NOT add
  clang's builtin-header dir, so a JIT source that transitively pulls a libc header (e.g.
  af/defines.h -> <stdlib.h> -> <stddef.h>) fails "'stddef.h' file not found"; add
  -isystem <rocm>/lib/llvm/lib/clang/<ver>/include. (3) Define -D__CUDACC_RTC__: project headers
  gate their host-only #includes behind `#ifndef __CUDACC_RTC__` (NVRTC auto-defines it; hipRTC
  does not), so without it the JIT compile drags in host headers. (4) Define -D__CUDA_ARCH__=NNN
  (>=530, e.g. 900 for gfx9) so embedded device headers take their intrinsic path; HIP provides
  the __half2short_rn / __hlt etc. intrinsics. CAUTION: defining __CUDA_ARCH__ can then surface
  host/device-attribute mismatches in __host__ __device__ helpers that call __device__-only
  intrinsics under hipRTC's host parse -- those helpers may need to be strictly __device__ on the
  hipRTC path -- arrayfire
- hipRTC's bundled std is SMALLER than NVRTC's: a project's NVRTC `namespace std` shim (often
  just is_same/enable_if) is missing is_integral / is_signed / integral_constant / numeric_limits
  that the device code uses; add them to the RTC-path shim (numeric_limits needs per-type
  min()/max() literals since <climits> is also absent) -- arrayfire
- rocThrust's rocprim opens a GLOBAL `namespace detail` (rocprim/common.hpp), which collides
  with a project-level `namespace detail = X;` ALIAS (a namespace alias and a real namespace
  cannot share a name at the same scope: "redefinition of 'detail' as different kind of
  symbol"). If the project needs the `detail` indirection in TUs that also pull rocThrust
  (e.g. reductions: Array.hpp -> common/jit/Node.hpp uses detail::, AND reduce_impl pulls
  hipcub -> rocprim), make `detail` a REAL namespace re-exporting the target via a
  using-directive (`namespace detail { using namespace arrayfire::cuda; }`) instead of an
  alias -- two real namespace definitions MERGE, the using-directive still resolves detail::X.
- HIP's <hip/hip_fp16.h> does `using half = __half;` at GLOBAL scope; a project that uses bare
  `half` to mean its own half type hits "reference to 'half' is ambiguous" in every TU pulling
  fp16. Re-declare the name inside the project's namespace (`using half = myns::half;`) so the
  nearer scope wins.
- HIP_vector_type (hipFloatComplex/hipDoubleComplex = HIP_vector_type<float|double,2>) ships
  FRIEND operator* and operator/ (componentwise) with NO opt-out macro. A project that aliases
  its complex type to hipComplex AND defines its own complex operator*/operator/ gets an
  ambiguity (both exact). operator+/- do NOT collide (HIP has only member +=/-=). Fix: do not
  make the complex type HIP_vector_type (plain POD or a struct derived from hipComplex so the
  friends are not inherited).
- hipSOLVER ships a cuSOLVER-COMPATIBLE hipsolverDn* API (internal/hipsolver-dense.h) that is
  signature-for-signature identical to cusolverDn* (potrf/getrf/geqrf/gesvd-with-rwork/ormqr/
  unmqr/getrs), so cuSOLVER code ports by a literal cusolverDn -> hipsolverDn rename (handle
  type cusolverDnHandle_t -> hipsolverHandle_t, status cusolverStatus_t -> hipsolverStatus_t,
  CUSOLVER_ -> HIPSOLVER_). Distinct from the native hipsolver* API (internal/hipsolver-
  functions.h). hipSOLVER has no CUSOLVER_STATUS_INVALID_LICENSE (orphan enum); it adds
  HANDLE_IS_NULLPTR / INVALID_ENUM.
- hipBLAS spells the half element type hipblasHalf (= uint16_t), NOT __half as cuBLAS does, so
  &hipblasHgemm does not match a std::function/function-pointer typedef built on __half*;
  reinterpret_cast the function pointer (the 16-bit types are layout-compatible).
- hip::device's INTERFACE_COMPILE_OPTIONS add "-x hip --offload-arch" scoped to
  $<COMPILE_LANGUAGE:CXX>, so linking hip::device forces EVERY host .cpp of the target through
  the HIP compiler (breaks the colmap host/device isolation). For a target whose .cu are
  already LANGUAGE HIP, link hip::host ONLY.
- CMake target_compile_options "-include /path/hdr.h" must be wrapped as
  "$<...:SHELL:-include /path/hdr.h>" or CMake de-duplicates/merges the two tokens and the
  second -include silently loses its flag.
- hipRTC hiprtcCreateProgram REJECTS an empty-string ("") header source body (INVALID_INPUT)
  where NVRTC accepts it; a NULL header pointer aborts in std::string ctor. arrayfire's
  compile_module.cpp passes several string("") dummy header entries -- they must be a
  non-empty placeholder on HIP. (agent_space/af_hiprtc_poc/min4.cpp isolates: "" mode FAILS,
  non-empty mode PASSES.)
- hipRTC REJECTS the NVRTC flag --device-as-default-execution-space ("unknown argument");
  drop it (HIP makes device the default execution space). Likewise map
  --gpu-architecture=compute_XX -> --offload-arch=<gcnArchName stripped of :sramecc+:xnack->,
  and drop NVRTC's --device-debug/--generate-line-info.
- hipRTC does NOT implicitly know /opt/rocm/include: an injected header whose body does
  `#include <hip/...>` fails to find it; pass -I<rocm>/include (ROCM_PATH or /opt/rocm) as a
  runtime-compile option. hipRTC DOES provide __half/__float2half/__half2float and float2/
  double2 as builtins with no header (min2.cpp), so the fp16/vector shims are belt-and-braces.
- The HIP backend keeps `namespace cuda` and builds as library afcuda (reports
  AF_BACKEND_CUDA): do NOT rename the namespace to `hip` (that would churn 354 files and the
  unified/api/test layers). The directory is hip/, the identity stays cuda. AF_BUILD_HIP is
  mutually exclusive with AF_BUILD_CUDA.
