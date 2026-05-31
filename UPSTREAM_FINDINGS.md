# Upstream findings

Bugs and library/toolchain gaps discovered while porting, curated for a later
decision on whether to report them upstream. This is a TRACKING file only:
opening an upstream issue or PR is gated behind explicit approval from jeff (per
CLAUDE.md, any UPSTREAM-visible action requires the `pr-approved-by-user` gate).
Nothing here has been reported. "Report decision" stays PENDING until reviewed.

Two classes:
- A. Genuine bugs in the upstream CUDA projects (latent on NVIDIA, exposed by ROCm).
- B. ROCm / AMD library + toolchain findings (gaps, divergences from CUDA libs).

Porting fault classes that are platform DIFFERENCES (not bugs) live in
PORTING_GUIDE.md; only items with a plausible "report upstream" action are here.

---

## A. Upstream CUDA-project bugs

### A1. arrayfire -- memcopy.cuh index typo (latent upstream)
- Component: `src/backend/cuda/kernel/memcopy.cuh`, `memCopyLoop13`.
- Bug: bounds check reads `(g1 < idims1)` where it should be `(id1 < idims1)`
  (wrong index variable). Real bug in upstream CUDA; latent only because the
  test path never instantiates that template specialization.
- Exposed on ROCm: the HIP build's test path does instantiate it, so it faulted.
- Our fix (HIP fork): corrected to `id1` (the CUDA source still carries the typo).
- Upstream target: arrayfire/arrayfire.
- Report decision: PENDING. Clean, self-contained upstream bug report candidate.

### A2. CV-CUDA -- test-fixture use-after-free on async copy
- Component: `tests/nvcv_types/cudatools_system/TestInterpolationVarShapeWrap.cpp`
  (dst-fill loop ~lines 144-158).
- Bug: a per-iteration local `std::vector<uint8_t> dstVec` is the source of a
  `cudaMemcpy2DAsync(..., stream)`, then destructs at end of iteration while the
  async H2D copy is still pending. Use-after-free / UB.
- Exposed on ROCm: NVIDIA's pageable async copy is effectively synchronous (the
  driver stages it before returning), so it always finishes before the free; HIP's
  `hipMemcpy2DAsync` from pageable memory is genuinely async, so it reads the freed/
  reused host buffer and writes garbage into dst row-stride padding. Failing set
  varies run-to-run (freed-buffer contents are nondeterministic).
- Our fix: NONE (correctly left the upstream test unmodified; the operator + our
  allocator are proven correct). Confined to 6-8 `correct_shift` rows.
- Underlying difference: pageable-async-memcpy semantics (see PORTING_GUIDE).
- Upstream target: CVCUDA/CV-CUDA.
- Report decision: PENDING. Strong report candidate (genuine UB; a 1-line fix
  upstream -- keep dstVec alive or sync -- would also harden their CI on any truly
  async copy implementation).

### A3. cudf -- dangling-reference UB in cast_functor
- Component: `cpp/include/cudf/detail/utilities/cast_functor.cuh`, `cast_fn<T>`
  same-type overload.
- Bug: `operator()(T&&) -> T&&` returned a reference into the transform-iterator's
  expired temporary storage (dangling reference). UB.
- Exposed on ROCm: rocPRIM block-loaded the dangling reference; min/max returned the
  seeded identity (INT32_MIN/MAX) and float/double sum returned nan -- exactly when
  reduction output type equals input element type. nvcc/CUB happen to tolerate it.
- Our fix (HIP fork): return by value (correct on CUDA too).
- Upstream target: rapidsai/cudf.
- Report decision: PENDING. The by-value fix is portable and correct on CUDA, so a
  clean upstream PR candidate (fixes latent UB even for NVIDIA users).

---

## B. ROCm / AMD library + toolchain findings

### B1. __fsqrt_rn not always correctly-rounded on gfx90a
- Symptom: e.g. `sqrt(93606.0f)` -> `__fsqrt_rn` gives `0x4398f9b9`, but the
  correctly-rounded value (host std::sqrt, `(float)__dsqrt_rn`) is `0x4398f9ba`
  (1 ULP). CUDA `sqrt.rn.f32` is IEEE correctly-rounded.
- Impact: breaks bit-exact-vs-CPU-gold tests (hit in CV-CUDA Normalize / PairwiseMatcher L2).
- Workaround (in our fork): route f32 sqrt through f64 `__dsqrt_rn`.
- Upstream target: ROCm (device-libs / HIP math). Report decision: PENDING. Worth
  reporting -- a correctly-rounded `__fsqrt_rn` is a documented CUDA contract.

### B2. rocBLAS / hipblasGemmEx -- no int8->float32 GEMM (int8->int32 IS supported)
- Symptom: arrayfire's schar gemm via `hipblasGemmEx` with HIP_R_8I inputs +
  HIPBLAS_COMPUTE_32F returns HIPBLAS_STATUS_NOT_SUPPORTED.
- Empirical: the hipBLAS API DOES expose `HIPBLAS_R_8I` + `HIPBLAS_COMPUTE_32I`, and
  gfx90a/CDNA2 has int8 MFMA, so int8 x int8 -> int32 GEMM IS supported. The
  unsupported thing is specifically the int8->FLOAT32 compute combo (and likely the
  CUDA path used int32 accumulate too). hipBLASLt also advertises int8.
- Implication: this is closeable in our fork (use COMPUTE_32I + post-convert int32 ->
  result type), NOT a hard library wall. Reclassified from "library gap" to
  "wrong-compute-type in our port + a docs/clarity gap in rocBLAS error messaging."
- Upstream target: rocBLAS (low priority; mostly a clarity issue). Report decision: PENDING.
- Action item: fix arrayfire's int8 gemm to use int32 accumulate (closes the last
  `blas` subcase) -- see arrayfire follow-on.

### B3. rocPRIM BlockRadixSort -- TempStorage union-aliasing causes LDS corruption
- Symptom: putting `BlockRadixSort::TempStorage` in a `union` with another in-use
  `__shared__` buffer (a pattern CUB tolerates) corrupts LDS and faults on rocPRIM
  (hit in arrayfire topk; the primitive is fine standalone).
- Our fix (HIP fork): split into separate `__shared__` allocations.
- Caveat: arguably undefined usage (overlapping live shared storage), but the
  CUB-vs-rocPRIM divergence is a real portability hazard.
- Upstream target: rocPRIM (robustness / document the divergence). Report decision: PENDING.

### B4. clang(HIP) defaults to -ffp-contract=fast (vs nvcc expression-only)
- Not a bug; a documented default difference. clang contracts FMAs across
  statements by default; nvcc only within an expression. Drifts bit-exact math
  (hit in CV-CUDA bicubic ~1 ULP). Workaround: pin `-ffp-contract=on`.
- Report decision: N/A (expectation/documentation note, not a defect).

### B5. hipSPARSE coverage (pending)
- To be populated by the arrayfire sparse-on-hipSPARSE port: any cuSPARSE feature
  with no hipSPARSE equivalent in ROCm 7.2.1. Per jeff, surfacing these gaps is an
  explicit goal of that port.
- Report decision: PENDING (awaiting the sparse port's findings).

---

## Build-config / environment (not bugs; for completeness)

- arrayfire `confidence_connected`: fails with AF_ERR_NOT_CONFIGURED because the
  headless build set AF_WITH_IMAGEIO=OFF (FreeImage absent); the test calls
  af::loadImage for its input and never reaches the GPU algorithm. FreeImage is a
  CPU-side third-party lib (nothing to do with ROCm). Closeable by installing
  libfreeimage-dev + rebuilding with AF_WITH_IMAGEIO=ON. Not an upstream finding.
