# CV-CUDA notes

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated). Strategy A
(pure CMake, colmap model) HIP port of the CV-operator core + NVCV, gated behind USE_HIP.
Fork: jeffdaily/CV-CUDA @ moat-port. Lead arch gfx90a (MI250X, ROCm 7.2.1).

## NESTED LAYOUT WARNING
The fork clone is at `projects/CV-CUDA/src/`, and the repo itself nests one more level:
real sources live under `projects/CV-CUDA/src/src/cvcuda/...` and `.../src/src/nvcv/...`.
The top CMakeLists is at `projects/CV-CUDA/src/CMakeLists.txt`.

## Build (gfx90a, core + C++ gtests, no Python)
```
cmake -S projects/CV-CUDA/src -B projects/CV-CUDA/src/build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_PYTHON=OFF -DBUILD_TESTS=ON -DBUILD_TESTS_CPP=ON -DBUILD_TESTS_PYTHON=OFF \
  -DBUILD_BENCH=OFF -DBUILD_DOCS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build projects/CV-CUDA/src/build-hip -j 16
```
Followers (gfx1100/gfx1151): same command, only change `-DCMAKE_HIP_ARCHITECTURES=<arch>`
(no source/CMake edit -- the targets read ${CMAKE_HIP_ARCHITECTURES}).
Builds CLEAN today: libnvcv_types.so, libcvcuda.so, and all C++ test exes.
Two C++ standards in one build: operator/legacy .cu compile at -std=gnu++17, the
cudatools_system tests at -std=gnu++20 (they use C++20 NTTPs); any compat-header
construct must be valid at BOTH (see MathOps note below).

## GPU validation (one isolated GCD)
4 GCDs (0-3); pick a free one via `rocm-smi --showpidgpus` / `--showmeminfo vram` (other
agents share the box). Run serially (no -jN). Gate suites:
```
HIP_VISIBLE_DEVICES=<n> build-hip/bin/cvcuda_test_system          # per-operator GPU-vs-CPU
HIP_VISIBLE_DEVICES=<n> build-hip/bin/nvcv_test_cudatools_system  # device cuda_tools
```
Current pass rate (gfx90a): cvcuda_test_system 2565/2619; nvcv_test_cudatools_system 1114/1123.

## Scope decisions
- SCOPED OUT: OpOSD, OpBndBox, OpBoxBlur (+ legacy/osd.cu, box_blur.cu, textbackend/,
  tests/.../OsdUtils.cu, the cvcuda_test_system_smoke exe). They depend on cuOSD, a prebuilt
  CUDA-only static lib (3rdparty/cuOSD/.../libcuosd.a) with NO source -- unportable. Gated out
  under USE_HIP in CvCudaLegacy.h and tests/cvcuda/system/CMakeLists.txt. ~73 operator suites
  remain (the cvcuda_test_system gate).
- nvcv standalone-consumer ExternalProject test skipped on HIP (re-runs cmake without the
  USE_HIP setup, would look for the CUDA toolkit); it is a packaging check, not a GPU test.

## Fault-class fixes applied (all arch-unified; CUDA path byte-for-byte unchanged)
- Compat layer: cmake/hip/ forwarding shim dir (on HIP include path only) + CvCudaHipCompat.h
  force-included on every HIP TU. cstdlib/cstring BEFORE hip_runtime (gpuRIR). Maps the cuda*/
  cub/cublas/cusolver/curand symbols the project uses. Defines NVCV_WARP_FULL_MASK (64-bit).
  MUST NOT define __CUDA_ARCH__ (keeps SaturateCast PTX + NVCV SIMD-intrinsics inert on their
  portable fallbacks). Also provides uint3 operator*/+ for the __hip_builtin_*_t index types and
  a __host__ __device__ declval shim (std::declval is host-only under clang).
- __CUDA_ARCH__ device-math guards extended to `|| __HIP_DEVICE_COMPILE__` (MathWrappersImpl.hpp,
  LinAlg.hpp). DeviceMin/MaxImpl: HIP has no ::umin/::ullmin/::llmin, so the plain ternary (lowers
  to the same min instruction). LinAlg Vector member-initializer dropped on HIP (used as __shared__,
  which clang forbids an initializer on, in BOTH passes).
- Metaprogramming.hpp: char1..4 base_type is `char` on HIP (HIP_vector_type<char,N> members are
  plain char, not signed char) so GetElement's reference return binds. KNOWN COST: this makes
  TypeTraitsMakeTypeVectorTest/3 (signed char) fail is_same_v(BaseType, signed char) -- char and
  signed char are distinct C++ types though same representation; the HIP report is correct for HIP.
- MathOps.hpp: HIP_vector_type ships its own vector operators (CUDA's vector_types.h ships none),
  causing ambiguity for mixed-element vec pairs, vec OP dim3, and cross-type ==/!=. Added HIP-only
  overloads that win partial ordering WITHOUT a requires-clause (must stay C++17-valid for the .cu
  TUs): a both-operands-concrete (vec,vec) for mixed element type (SFINAE !is_same), and
  CONCRETE-dim3 forward/mirror overloads (more specialized than HIP's templated U, which also beats
  HIP's enable_if-constrained integral operators like %). dim3 bodies are spelled out (the shared
  if-constexpr body probes .w, a hard error on the 3-element dim3 even when discarded).
- Warp/wave64: NVCV_WARP_FULL_MASK 64-bit (reduce_kernel_utils FINAL_MASK, OpFindHomography masks).
  OpLabel connected-components + threshold/threshold_var_shape Otsu scans: explicit width-32 on every
  __shfl_*_sync (the block packs 32-lane rows, so a wave64 wavefront is two 32-lane groups), and the
  warp-synchronous reduction tails replaced with fully __syncthreads-synchronized trees (MPPI lesson:
  the low 32 lanes of a 64-lane wavefront are not lockstep across unsynced steps).
- StreamId.cpp: hipStreamGetId + pointer-value fallback. priv/Assert trap -> handled. __ldg ->
  plain load (HIP lacks __ldg for all vector elem types; the cache hint is advisory on CDNA).
  cuBLAS/cuSOLVER -> hipBLAS/hipSOLVER (OpFindHomography), cuRAND -> hipRAND (gaussian_noise),
  CUB -> hipCUB. LTO OFF on HIP. ENABLE_COMPAT_OLD_GLIBC OFF on HIP.
- CudaFwd.h: cudaArray_t/CUstream typedef'd to hip types on HIP.

## THE BIG FIX: zero device allocations on HIP (NVCV DefaultAllocator)
hipMalloc returns RECYCLED device memory with STALE contents; freshly cudaMalloc'd memory on the
NVIDIA setups reads back as zero. ~160 of the operator gtests fill a tensor's valid region, run the
op, then compare the WHOLE strided buffer -- INCLUDING the row-stride padding the operator never
writes -- against a zero-initialized std::vector CPU reference. So they implicitly assume device
padding is zero. On the FIRST op in a process the padding is fresh-zero (passes); every later op
gets recycled dirty padding (fails non-deterministically). Confirmed: any op passes in isolation but
fails after any preceding op; ALL mismatching bytes are at offsets >= validRowBytes within each row
stride. Fix: src/nvcv/src/priv/DefaultAllocator.cpp doAllocCudaMem -> hipMemset(ptr,0,size) under
USE_HIP. Took cvcuda_test_system 2405->2565 and cudatools_system 1015->1114. This is the single
most important non-build fix; keep it.

## Test-source-only fixes (HIP_vector_type ergonomics, not operator bugs)
- 1-element vector from a scalar: `int1 x = {v}` / `ValueAt<int1>(...,{z})` use HIP_vector_type's
  EXPLICIT scalar ctor -> rejected as copy-list-init. Use direct-list-init `int1{v}` or cuda::SetAll.
  Fixed in TestOpMinMaxLoc.cpp, TestOpSIFT.cpp, TestTypeTraits.cpp.
- ttype::Value<vec{...}> NTTP: HIP_vector_type IS a usable C++20 NTTP, but its variadic ctor needs
  EXACTLY N args (CUDA's aggregate zero-fills). Spell all components: float4{a,b,c}->{a,b,c,0.f}
  (TestMathOps.cpp). Only under-specified vector brace-inits needed this (2 sites).
- `typename DependentType::ValueType` and `this->member` for dependent-base members (clang two-phase
  lookup): OpCropFlipNormalizeReformat.cu, InterpolationVarShapeWrap.hpp, DeviceTensorBatchWrap.cu.
- DeviceTensorWrap.cu / DeviceFullTensorWrap.cu: DropCast<N>(threadIdx) -> wrap threadIdx as
  uint3{threadIdx.x,..} (the builtin index type has no NVCV TypeTraits; uint3 is byte-identical on CUDA).

## RESIDUAL GENUINE FAILURES (real GPU bugs, NOT the padding artifact -- they fail in isolation too)
Next porter cycle: these need kernel-level GPU debugging (compute-sanitizer / in-kernel printf of
the actual writes), not static analysis. Verified the obvious suspects are NOT the cause:
hipCUB DeviceReduce::Reduce with a custom int3 max functor is correct (n=1 and n=2); the bare erase
kernel writes its last row standalone; Tensor4DWrap::ptr(b,y,x,c) NHWC addressing is correct standalone;
the erase kernel receives CORRECT args at runtime (verified via injected printf: area0 anchor(0,0)
erasing(10,10,1), area1 anchor(10,10) erasing(20,20,1), blockDim 400). So the bug is in the INTEGRATED
path, likely a subtle interaction. Failing suites (gfx90a, fail standalone-as-group):
- OpSIFT (11): op(...) THROWS an exception (TestOpSIFT EXPECT_NO_THROW fails) + a no_linear_system
  case. SIFT is the most complex operator (cuBLAS/cuSOLVER homography-free path, DoG pyramid,
  descriptor build). Start by catching/printing the thrown nvcv::Exception to find the failing call.
- OpPairwiseMatcher (7, params 12,13,15-19): GPU output is EMPTY {} where the CPU ref has matches.
  Uses hipCUB BlockRadixSort (sorts full key, so the begin_bit!=0 ROCm bug should not apply -- but
  re-verify the block sort/scan produces output on wave64; the empty result smells like a block-wide
  scan/count returning 0).
- OpHistogramEqVarShape (15): varshape correct_output. Shares the histogram-equalization path; check
  the per-image CDF scan (likely another warp/block scan wave64 issue or a varshape stride read).
- OpErase (4: params (1/2, false, false/true), i.e. the non-random uchar cases): the LAST row of an
  erase area and the FIRST row of the next area are not written (area0 row9, area1 row10), while
  rows 0-8 are. Args are correct at runtime; bare kernel works standalone -> integrated-path bug.
- Singletons: OpFindHomography (1), OpGaussian (1), OpMorphology (1), OpMorphologyVarShape (1),
  OpNormalize (1), OpWarpPerspective (1) -- one parameter each; likely the same class as the above.
- nvcv_test_cudatools_system (9): TensorBatchWrapTensorTest (5), InterpolationVarShapeWrapTest (2),
  InterpolationWrapHWTest (1), TypeTraitsMakeTypeVectorTest (1, the known signed-char divergence).

## Repro scratch (agent_space/cvcuda/, gitignored)
Standalone probes used: mathops_probe / full_probe / c17_probe (MathOps overload resolution at C++17
and C++20), cub_reduce_probe / cub_n1 (hipCUB DeviceReduce), erase_repro (bare erase kernel),
twrap_probe (Tensor4DWrap NHWC ptr), streamid_probe (hipStreamGetId uniqueness), sem_check (MathOps
runtime semantics). gtest logs: agent_space/cvcuda/gtest_*_v2.log.

## Inter-project deps
NONE. Submodules (pybind11, googletest, dlpack, nvbench) and 3rdparty (cuOSD, scoped out) are
self-contained. depends_on stays empty.
