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
Current pass rate (gfx90a): cvcuda_test_system 0 failures (2608 pass + disabled/negative, exit 0);
nvcv_test_cudatools_system 1116/1123 (7 residuals, both clusters ROOT-CAUSED to non-port artifacts:
6 InterpolationVarShapeWrap.correct_shift = a TEST-FIXTURE use-after-free on an async copy source
(freed per-iteration dstVec, exposed by HIP's truly-async pageable hipMemcpy2DAsync; production op +
allocator proven correct) + 1 char/signed-char type-identity dictated by the upstream HIP vector header.
Neither is a ROCm/operator defect; see REMAINING below. No source change was needed this cycle.

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

## RESOLVED root causes (2026-05-31 porter cycle: cvcuda 2565->0 failures; cudatools 1110->7)
Five arch-unified source fixes took cvcuda_test_system to ZERO failures (2608 pass + the rest
disabled/negative; `cvcuda_test_system` exit 0) and nvcv_test_cudatools_system to 1116/1123. Earlier
"residual genuine failures" were NOT distinct kernel bugs; they reduced to these root causes. CUDA path
byte-for-byte unchanged (every fix HIP-guarded or a build flag).

1. THE TEXTURE-PITCH DIVERGENCE (the big one; fixed OpErase, OpSIFT, OpGaussian, OpFindHomography,
   ~half of HistogramEq, OpNormalize, TensorBatchWrap, several Interpolation cases). NVCV derives the
   tensor/image ROW-PITCH alignment from cudaDevAttrTexturePitchAlignment. NVIDIA reports 32 (tight: a
   640-byte uchar row stays 640); gfx90a reports 256 (640 -> padded to 768). ~160 gtests fill the valid
   region then compare the WHOLE strided buffer including row-stride padding against a zero CPU ref, so
   they assume the CUDA tight pitch. The OpErase "last row unwritten" symptom was the test indexing
   test[9*640] while the real stride was 768 -- the kernel was always correct (proven: injected printf
   showed every write landing at the right logical coord). No CV-CUDA tensor is HW-texture-bound, so the
   256B pitch is unnecessary. Fix: cmake/hip/CvCudaHipCompat.h wraps cudaDeviceGetAttribute in an inline
   shim that clamps the texture-pitch-alignment query to 32 on HIP. Confirmed gfx90a returns 256 for both
   hipDeviceAttributeTexturePitchAlignment and ...TextureAlignment.

2. HistogramEqVarShape (15): the varshape path NEVER zeroed m_histoArray before the atomicAdd histogram
   accumulation (the tensor HistogramEq path does: cudaMemsetAsync). m_histoArray is a direct cudaMalloc
   (not via the NVCV DefaultAllocator), so the DefaultAllocator hipMemset fix did not cover it; recycled
   hipMalloc gave dirty histograms. Fix: cudaMemsetAsync(m_histoArray,0,...) in HistogramEqVarShape::infer
   (histogram_eq_var_shape.cu), matching the tensor path.

3. OpPairwiseMatcher (7): two bugs. (a) The NB=32/128 PointT cache stored RT(uint32) words and read them
   back as type T via reinterpret_cast on a private RT[] array -- a strict-aliasing violation clang/HIP
   exploited at -O3, eliding the float reads so every L2 distance stayed FLT_MAX (-> empty crossCheck
   output). Fixed with a union (RT words / T elems). (b) cub::BlockReduce/BlockRadixSort TempStorage is
   reused across the two SortKeyValue calls in the crossCheck path; on a 64-thread (=wave64) block the
   collective lowers to a single-wavefront reduce with no syncing epilogue, so TempStorage reuse raced.
   Added __syncthreads() after the reduce and at end of SortKeyValue. (OpPairwiseMatcher.cu)

4. gfx90a __fsqrt_rn IS NOT ALWAYS CORRECTLY ROUNDED (fixed OpNormalize + L2 PairwiseMatcher exactness).
   sqrt(93606.0f): __fsqrt_rn -> 0x4398f9b9 but the correctly-rounded value (host std::sqrt, and
   (float)__dsqrt_rn) is 0x4398f9ba (1 ULP high). CUDA sqrt.rn.f32 is correctly rounded, so the bit-exact
   gtests pass on NVIDIA, fail on gfx90a. Fix: DeviceSqrtImpl routes 32-bit sqrt through the correctly
   rounded f64 __dsqrt_rn on HIP (MathWrappersImpl.hpp); CDNA has fast f64 sqrt.

5. cuda::min/max NaN handling (fixed OpMorphology/OpMorphologyVarShape CLOSE on RGBAf32). The morph tests
   fill float images with RANDOM BYTES (NaN/inf), and host gold + device both call cuda::min/max. Host
   MinImpl/MaxImpl = std::min/std::max (b<a?b:a / a<b?b:a). HIP DeviceMinImpl/MaxImpl were a<b?a:b and
   a>b?a:b -- the OPPOSITE NaN selection. Respelled the HIP device ternaries to exactly match the host
   std::min/std::max forms so device==host bit-for-bit on NaN/signed-zero (MathWrappersImpl.hpp).

6. -ffp-contract=on for HIP (fixed OpWarpPerspective cubic + the cubic-math half of Interpolation tests).
   clang(HIP) defaults to -ffp-contract=fast (forms FMAs ACROSS statements); nvcc only contracts within
   one expression (--fmad=true). The extra contraction drifted HIP float results ~1 ULP from the CUDA
   build and CPU gold (e.g. the bicubic weight chain in InterpolationWrap GetCubicCoeffs), failing
   bit-exact compares. Pinned CMAKE_HIP_FLAGS to -ffp-contract=on (CMakeLists.txt) to match CUDA/host.

## REMAINING cudatools residuals (7; ROOT-CAUSED to NON-PORT artifacts, GPU compute proven correct)
Both residual clusters were ROOT-CAUSED to non-production artifacts in the 2026-05-31 (b) porter cycle.
No production fix is warranted; no source change was needed. cvcuda_test_system stays 0 failures.

- InterpolationVarShapeWrapTest.correct_shift (6-8 per run; non-deterministic SET): a TEST-FIXTURE
  USE-AFTER-FREE on an async copy source, NOT a GPU/operator/allocator bug. ROOT CAUSE (this cycle,
  conclusive): in TestInterpolationVarShapeWrap.cpp the dst-fill loop declares `std::vector<uint8_t>
  dstVec(...,0)` as a PER-ITERATION LOCAL and issues `cudaMemcpy2DAsync(dstBasePtr, ..., dstVec.data(),
  ..., stream)` -- then dstVec is DESTROYED at the end of the iteration while the async H2D copy is still
  pending. On NVIDIA cudaMemcpy2DAsync from PAGEABLE memory is effectively synchronous (driver stages it
  before returning), so the copy finishes before the free -> works. On ROCm hipMemcpy2DAsync from pageable
  memory can be genuinely async, so it reads dstVec's freed/reused memory and writes garbage into the dst.
  The kernel overwrites valid pixels but not the row-stride padding [width,rowStride), so the garbage
  survives there and the full-strided compare against a zeroed CPU ref mismatches; the failing set varies
  run-to-run because freed-buffer contents are nondeterministic.
  PROOF (instrumented probes, all reverted): (a) an ALLOC-PROBE inside DefaultAllocator::doAllocCudaMem
  read back every fresh device buffer right after its hipMemset(0): postMemsetNZ=0 ALWAYS (the production
  zero-init IS complete -- it covers the full padded buffer, e.g. 768B for a w13 h18 Y8 image with rs=32);
  (b) a PRE-kernel probe (after cudaStreamSynchronize) found padNZ>0 BEFORE the kernel launches (e.g.
  case0: row0 cols 13/14/15 = 0xb0/0xd7/0x97), and POST-kernel padding == PRE-kernel padding (kernel never
  writes padding); (c) keeping dstVec alive (push to a kept vector, NO sync change) makes ALL 21 cases pass
  3/3 runs; (d) syncing after each async fill also makes all 21 pass 3/3 runs; (e) baseline (freed dstVec)
  fails a varying 6-8 set. (c) isolates the variable to BUFFER LIFETIME, not stream ordering (prefill,
  kernel, readback are all on the same stream and thus ordered regardless). => production op + allocator
  are correct; the dirt is the test's own freed-buffer async read. A genuine test-side artifact: the test
  relies on CUDA's pageable-async-is-synchronous behavior. Operator-correctness gate (cvcuda_test_system)
  is green. Do NOT edit the test; this is upstream test-fixture UB exposed by HIP semantics.
- TypeTraitsMakeTypeVectorTest/3.correct_type_traits (1): char-vs-signed-char type IDENTITY, dictated by
  the upstream HIP vector-types header. MakeType<signed char,4> -> char4 = HIP_vector_type<char,4> (members
  `char`); the test asserts is_same_v(signed char, BaseType<that>). The port MUST set BaseType<charN>=char
  on HIP (a `signed char&` reference accessor will not bind to a `char` member). Assessed this cycle whether
  CV-CUDA could close it cleanly: HIP_vector_type<signed char,4> IS a distinct, instantiable type (probe:
  is_same vs char4 == 0, members are signed char), so MakeType could in principle map signed char ->
  HIP_vector_type<signed char,N>. REJECTED as a per-platform hack: the canonical 8-bit-signed vector across
  CUDA+HIP is charN; all of GetElement/MathOps/SaturateCast/make_* and the operator kernels key on charN,
  HIP ships vector operators only for HIP_vector_type<char,N>, and a parallel signed-char family would
  diverge the HIP type system from CUDA and break arithmetic. `char`/`signed char` are distinct C++ types
  (identical representation, SCHAR_MIN..SCHAR_MAX). True upstream-header type-identity deferral, not a GPU
  bug. 1122/1123 with B excluded.

## Repro scratch (agent_space/cvcuda/, gitignored)
This cycle: pitchprobe (hip texture pitch/align = 256), sqrt_probe (gfx90a __fsqrt_rn vs (float)__dsqrt_rn
mismatch on 93606), cub_blockreduce_probe (hipCUB BlockReduce<KeyValueT,64> custom-min OK standalone),
img_zero_probe / batch_pad_probe (fresh nvcv::Image + batch alloc both read back all-zero). Baseline/run
logs: agent_space/cvcuda/baseline_*.log, run_*_cvcuda.log, run_*_cudatools.log.
Prior cycle probes: mathops_probe/full_probe/c17_probe, cub_reduce_probe/cub_n1, erase_repro, twrap_probe,
streamid_probe, sem_check.

## Inter-project deps
NONE. Submodules (pybind11, googletest, dlpack, nvbench) and 3rdparty (cuOSD, scoped out) are
self-contained. depends_on stays empty.
