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
- Action item: DONE. arrayfire's HIP gemmDispatch now routes the schar (int8)
  path through hipblasGemmEx with HIP_R_8I inputs + HIP_R_32I output +
  HIPBLAS_COMPUTE_32I (int32 accumulate), then casts the int32 result into the
  f32 output array (s8's out_type). MatrixMultiply.schar passes on gfx90a;
  blas 126/127 -> 127/127. The `if constexpr (is_same<Ti,schar>)` guard keeps the
  int32 cast out of the float/complex instantiations. Confirms int8->int32 GEMM
  is supported on gfx90a; only int8->float32 was the (correct) rejection.

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

### B5. hipSPARSE coverage (arrayfire sparse-on-hipSPARSE port) -- NO GAPS FOUND
- Scope: the cuSPARSE surface arrayfire's sparse subsystem uses, ported to
  hipSPARSE 4.2.0 (ROCm 7.2.x) for the gfx90a backend. Per jeff, surfacing any
  cuSPARSE feature with no hipSPARSE equivalent was an explicit deliverable.
- RESULT: hipSPARSE 4.2.0 covers EVERY cuSPARSE entry point arrayfire uses, with
  exact 1:1 naming (cusparse* -> hipsparse*, CUSPARSE_* -> HIPSPARSE_*) and
  matching signatures. No silent stub was needed; the three sparse binaries +
  threading all pass on real gfx90a (sparse 86/86, sparse_convert 41/41,
  sparse_arith 123/123, threading 9/9). The full coverage map (all present):
  - Generic API: cusparseCreateCsr / CreateCsc / CreateCoo / CreateDnVec /
    CreateDnMat / DestroySpMat / DestroyDnVec / DestroyDnMat, SpMV +
    SpMV_bufferSize, SpMM + SpMM_bufferSize, DenseToSparse_bufferSize /
    _analysis / _convert, SparseToDense + _bufferSize, SpMatGetSize,
    CsrSetPointers / CscSetPointers. (in internal/generic/, internal/conversion/.)
  - Legacy / sort / coordinate: Xcsrsort + _bufferSizeExt, Xcoosort_bufferSizeExt
    + XcoosortByRow, Xcsr2coo, Xcoo2csr, CreateIdentityPermutation,
    CreateMatDescr / SetMatType / SetMatIndexBase / DestroyMatDescr.
  - csrgeam2 (typed S/D/C/Z) + _bufferSizeExt + Xcsrgeam2Nnz (sparse+sparse add/sub).
  - Enums: INDEX_32I, INDEX_BASE_ZERO, ORDER_COL, MATRIX_TYPE_GENERAL,
    DIRECTION_ROW/COLUMN, OPERATION_{NON_,}TRANSPOSE / CONJUGATE_TRANSPOSE,
    DENSETOSPARSE_ALG_DEFAULT, SPARSETODENSE_ALG_DEFAULT, SPMV_CSR_ALG1 /
    SPMV_ALG_DEFAULT, SPMM_CSR_ALG1, all status codes.
- Two NON-gap porting notes (differences, not missing features), both handled in
  the fork and documented in arrayfire notes.md / PORTING_GUIDE.md:
  1. void*-aliasing: hipsparseDnVecDescr_t and hipsparseDnMatDescr_t (and
     hipsparseHandle_t / hipsparseMatDescr_t) are all `typedef void*`
     (hipsparseSpMatDescr_t is a distinct struct ptr). The shared type-keyed
     unique_handle<T> machinery redefines ResourceHandler<void*> when one TU
     pulls two of them -- the amgcl gotcha. Fixed with the tag-keyed TaggedHandle.
  2. SpMV/SpMM take the compute type as a hipDataType (getType<T>()), NOT a
     hipblasComputeType_t. Trivial; the dense-gemm getComputeType<T>() in the same
     backend returns hipblasComputeType_t for hipblasGemmEx, a different enum
     family, so sparse must use getType<T>() there.
- Report decision: NOT NEEDED as a gap report (hipSPARSE is complete here). The
  void*-typedef descriptor aliasing is a recurring portability hazard worth a
  hipSPARSE docs note (low priority); same class as the hipBLAS/hipSOLVER void*
  handles. Report decision: PENDING (low priority, docs/robustness only).

### B6. raft lanczos -- cuSOLVER *Host / Fortran-LAPACK symbols not in ROCm aliases
- raft defers its 4 lanczos TUs (RAFT_COMPILE_LANCZOS=OFF). CONSEQUENTIAL: cuvs's
  eigen_solvers.cuh includes raft/sparse/solver/lanczos.cuh (spectral clustering +
  spectral_embedding), so cuvs's spectral path is unbuildable on ROCm without it.
  This is the same consequential-deferral class as raft neighbors (now fixed).
- The blocker is NOT an absent hipSOLVER device kernel (the reviewer corrected the
  earlier csrqrsvBatched/Xsyevd characterization -- those are not called). The
  eigensolve core is HOST LAPACK: Lapack<T>::sterf/steqr/geqrf -> Fortran sgeqrf_/
  dgeqrf_ plus cusolverDn*Host helpers (spectral/detail/lapack.hpp). The ROCm math
  aliases cover the DEVICE cusolverDn* but not the *Host / Fortran symbols -- that
  is the real link blocker. WORKABLE via OpenBLAS/LAPACK Fortran directly or a
  hipSOLVER device syevd; not a hard wall.
- Action: port raft lanczos (write the host-LAPACK glue) BEFORE porting cuvs (a
  prerequisite). cuvs is parked, so deferred for now. Report decision: N/A as an
  upstream bug (it's our glue to write); tracked here as a consequential
  cuvs-prerequisite deferral so it is not lost.

### B7. NVIDIA OptiX has no ROCm equivalent (the OptiX -> HIP-RT cluster)
- Class: a fundamental ROCm ecosystem gap, not a bug. OptiX (the NVIDIA
  ray-tracing API: `optixAccelBuild` GAS/BVH, `optixTrace`, the
  `__raygen__`/`__anyhit__`/`__closesthit__`/`__miss__` program model compiled
  to OptiX-IR/PTX and loaded at runtime via `optixModuleCreate` + SBT) has NO
  drop-in ROCm/HIP equivalent. hipify does not touch it (no symbol mappings);
  there is no `hipOptix`. The AMD analogue is HIP-RT (BVH build/update +
  traversal + custom intersection callbacks) or a hand-written HIP BVH-traversal
  kernel -- a different API and program model, so any OptiX-based renderer needs
  a backend REWRITE, not a mechanical port.
- First seen: EnvGS (`submodules/diff-surfel-tracing`), the environment-Gaussian
  REFLECTION path. It builds a triangle-GAS from 2DGS surfel disks and traces
  reflected rays with custom anyhit alpha-compositing + a differentiable
  backward traversal pass; loaded from PTX. There is NO software/CPU tracing
  fallback, and the reflection feature is wired ONLY through OptiX
  (`HardwareRendering`), so reflections cannot be produced on AMD at all without
  the rewrite. EnvGS's DIFFUSE/rasterized path (the diff-surfel-rasterizations
  2DGS rasterizer) is independent of OptiX and IS ported + GPU-validated on
  gfx90a; only the reflection tracer is deferred (Stage 2). See
  projects/EnvGS/notes.md "Stage 2 deferred".
- Cluster: this is a recurring pattern, not an EnvGS one-off. Other MOAT
  candidates likely share the OptiX-bound-renderer wall and should be triaged
  for the same Stage-1-rasterizer / Stage-2-tracer split or a HIP-RT backend:
  **rmcl** and **splatad** are flagged as probable OptiX users. Any "3DGS/2DGS
  with hardware-traced reflections/secondary-rays" project is suspect.
- Open question (decides port-vs-defer for the whole cluster): HIP-RT
  availability + maturity on ROCm 7.2.x, and whether it can express a
  DIFFERENTIABLE tracer (forward + a backward through traversal). On this host
  `import hiprt` fails and no OptiX SDK is installed. Until HIP-RT is proven to
  cover differentiable surfel tracing, OptiX reflection paths are unvalidatable
  by construction.
- Report decision: N/A as an upstream bug (the gap is the ROCm ecosystem's, not
  the CUDA project's). Tracked here so the OptiX wall is a known, named cluster
  rather than re-discovered per project.

---

## Build-config / environment (not bugs; for completeness)

- arrayfire `confidence_connected`: fails with AF_ERR_NOT_CONFIGURED because the
  headless build set AF_WITH_IMAGEIO=OFF (FreeImage absent); the test calls
  af::loadImage for its input and never reaches the GPU algorithm. FreeImage is a
  CPU-side third-party lib (nothing to do with ROCm). Closeable by installing
  libfreeimage-dev + rebuilding with AF_WITH_IMAGEIO=ON. Not an upstream finding.
