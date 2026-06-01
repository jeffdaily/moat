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

### B6 UPDATE (2026-06-01) -- LINK GAP CLOSED; a SEPARATE upstream runtime divergence remains
- The host-LAPACK/cuSOLVER-*Host link gap is CLOSED. In spectral/detail/lapack.hpp,
  the cuSOLVER *Host helpers (cusolverDnS{gemm,sterf,steqr}Host + D) are
  USE_HIP-guarded out and the private Lapack<T> helpers route directly to Fortran
  LAPACK/BLAS: sgemm_/dgemm_, ssterf_/dsterf_, ssteqr_/dsteqr_ (geqrf/ormqr/geev
  already called the Fortran symbols on both paths). The HIP build now sets
  RAFT_COMPILE_LANCZOS=ON, finds LAPACK (find_package(LAPACK) -> OpenBLAS +
  reference LAPACK; the system resolves a MKL+openblas mix that exports every
  Fortran symbol), and links LAPACK::LAPACK into raft_lib (so raft::compiled
  propagates it) + the lanczos test. Two further HIP build gaps surfaced and were
  fixed under USE_HIP: (1) spectral/detail/matrix_wrappers.hpp included
  thrust/system/cuda/execution_policy.h (pulls the missing
  cub/detail/detect_cuda_runtime.cuh) and named thrust::cuda_cub::
  execute_on_stream_nosync_base -> swapped to thrust/system/hip/execution_policy.h
  + thrust::hip_rocprim::execute_on_stream_nosync_base; (2) three cuSPARSE SpMV
  enum aliases were missing from raft_mathlib_aliases.inc (CUSPARSE_SPMV_ALG_DEFAULT
  / SPMV_CSR_ALG1 / SPMV_CSR_ALG2 -> the HIPSPARSE_* values that already exist).
  With these, libraft (4 rmat + 4 lanczos TUs) and a narrow LANCZOS_TEST both
  compile + link cleanly on gfx90a.
- The device eigensolve (raft::linalg::eig_dc -> hipsolverDn{S,D}syevd via the
  USE_HIP eigDC_legacy path) is CORRECT: a standalone probe confirms info=0,
  ascending eigenvalues, and orthonormal eigenvectors matching A*v=lam*v to ~1e-7
  on gfx90a. The other primitives the iteration uses were each verified correct
  against CPU references on gfx90a (raft norm ALONG_ROWS on a (1,n) row ~1e-8;
  gemv CUBLAS_OP_T; gemm col/col/col).
- SEPARATE, STILL-OPEN issue (NOT the B6 host-LAPACK gap, NOT a wave64 fault in our
  changes): raft's thick-restart Lanczos (sparse/solver/detail/lanczos.cuh) DIVERGES
  at runtime on gfx90a. The first ncv-step factorization and the first restart are
  numerically sane; the SECOND restart's lanczos_aux blows up exponentially
  (alpha/beta ~1e8, then 1e23, 1e40, ... -> NaN), so hipSOLVER syevd eventually
  reports non-convergence (dev_info>0) and the test throws. The divergence is
  IDENTICAL in fp32 AND fp64 (so not a tol=1e-15-in-fp32 artifact) and reproduces
  for all variants (SA/LA/SM/LM), with every device primitive individually proven
  correct above. The file carries leftover hardcoded debug scaffolding (a
  gemv on M={1,2,3,4,5,6} into an unused buffer) indicating experimental/
  unvalidated upstream code; the restart reorthogonalization (the gemv Gram-Schmidt
  + the V_T@eigenvectors_k ritz gemm + the transpose) corrupts the carried basis
  between restarts. Root cause not isolated to a single primitive; it looks like an
  upstream algorithm/code defect (or an fp path that only happens to terminate on
  NVIDIA). Report decision: candidate upstream bug report (rapidsai/raft) once
  isolated, distinct from the (now-closed) ROCm library gap. The lanczos port was
  therefore NOT pushed to the fork (it would move HEAD and force the passing
  platforms to revalidate a non-validatable tip); the working-tree port + the
  build/test recipe live in agent_space/raft_lanczos/ for a follow-up. The raft
  fork tip stays at 70773a9 (neighbors/distance/CK, already validated).

### B6 UPDATE (2026-06-01, second pass) -- ROOT CAUSE ISOLATED: upstream host-side async-copy race (AMD-exposed)

Source-only investigation (no build; GPUs busy). Findings supersede the "root
cause not isolated" note above.

- NOT A PORT BUG. agent_space/raft_lanczos lanczos.cuh is byte-for-byte identical
  to upstream rapidsai/raft branch-25.08 (our base; diff modulo comments empty).
  Our B6 changes never touched this file. The divergence is upstream code.
- The M={1,2,3,4,5,6} scaffolding is harmless DEAD CODE, refuted as the cause:
  `out` is never read; the gemv is in-bounds (OP_N reads 3*2=6 from a 6-elt 2x3
  M_dev) and uses the same HOST pointer mode + per-call cublasSetStream as the
  real gemv, so it corrupts no shared handle/stream state. Upstream PR #2918
  "Lanczos remove dead code" deletes exactly this block (gone from main).
- ROOT CAUSE: a host-vs-stream data race in lanczos_aux. It does
  raft::copy(&b/&alpha_i_host, <device beta/alpha>, 1, stream) (cudaMemcpyAsync
  into HOST scalars, no sync) then feeds &alpha_i_host/&b to cublasaxpy in the
  DEFAULT host pointer mode, which dereferences them on the HOST at enqueue. NVIDIA
  masks this (small pageable D2H is host-synchronous on CUDA); ROCm's
  hipMemcpyAsync to pageable host memory is not, so axpy reads a STALE beta and the
  three-term recurrence loses orthogonality and explodes on the second restart
  (race => fp32==fp64 identical, all SA/LA/SM/LM). Surfaces on the 2nd restart
  because the i=0 carried beta index is still 0 (a stale read reads ~0, looks fine)
  until nonzero betas populate.
- FIX (ours, narrow, UNTESTED): a USE_HIP-guarded resource::sync_stream(handle,
  stream) between those two raft::copy calls and the axpy (staged in
  agent_space/raft_lanczos/.../lanczos.cuh). CUDA path byte-identical. Not pushed
  (would move HEAD to a non-validatable tip and force gfx90a/gfx1100 revalidate).
- SECONDARY upstream fragility (independent of the race): the hardcoded SM/LM
  eigenvalue tolerances are flaky on NVIDIA TOO -- rapidsai/raft #3021 fails
  LanczosTestD_SM on A100/cuda13.2 (actual 0.013678 != expected 0.013692 @ approx
  1e-5); upstream mitigation was PR #3046 "Relax threshold lanczos SM gtests"
  (merged 2026-06-01). See also #2485/#2519 (hardcoded test data). Expect SM/LM to
  stay tolerance-sensitive on AMD even after the race fix; SA/LA should pass.
- Upstream report: the async-copy race is worth filing against rapidsai/raft
  sparse/solver/detail/lanczos.cuh (same lines PR #2918 left intact), but DO NOT
  open upstream issues/PRs/comments without jeff's approval. Details +
  evidence in projects/raft/notes.md "## Lanczos divergence investigation 2026-06-01".

### B6 UPDATE (2026-06-01, third pass) -- SYNC FIX VALIDATED ON gfx90a: race fixed, but a SECOND ROCm bug (rocSOLVER stedc) now gates lanczos

Built + ran the staged USE_HIP sync_stream fix on real gfx90a (GCD1, narrow
RAFT_COMPILE_LANCZOS=ON + RAFT_TEST_LANCZOS-only build; ctest -R LANCZOS twice +
an AMD_LOG_LEVEL=3 dispatch check). Outcome:

- THE SYNC FIX WORKS for its target: the alpha/beta -> NaN divergence is GONE.
  Zero NaN/inf in either run; results are now fully DETERMINISTIC run-to-run (the
  run-to-run nondeterminism the race caused is eliminated). This confirms the B6
  second-pass diagnosis: the host-vs-stream async-copy race was real and the
  one-line resource::sync_stream(handle, stream) guard removes it. CUDA path
  byte-identical (guard is USE_HIP-only).
- BUT lanczos is still 0/11 on gfx90a. A SECOND, INDEPENDENT ROCm bug now blocks
  every variant (SA/LA/SM/LM, F and D, + Rmat): the dense eigensolve inside
  lanczos (raft::linalg::eig_dc -> eigDC_legacy -> hipsolverDn{S,D}syevd, which
  rocSOLVER implements as STEDC divide-and-conquer) returns dev_info != 0 and the
  ASSERT at raft/linalg/detail/eig.cuh:81 throws "eigensolver couldn't converge to
  a solution". AMD_LOG confirmed the rocSOLVER stedc_* / sytd2 kernel family
  dispatching on gfx90a:sramecc+:xnack-. This is NOT a NaN/divergence (our fix
  handled that) and NOT the SM/LM gtest tolerance flakiness (#3021) -- it is a
  THIRD failure class, a clean solver non-convergence that aborts before the
  eigenvalue comparison ever runs. The SM/LM tolerance question is therefore
  MASKED (untestable until stedc converges).
- ROOT CAUSE of the new wall: rocSOLVER's STEDC (divide-and-conquer symmetric-
  tridiagonal eigensolver) is less robust than cuSOLVER's syevd on the
  clustered/near-degenerate tridiagonals lanczos produces. Note the earlier
  syevd_probe that passed to ~1e-7 used a generic dense symmetric matrix, not
  these tridiagonals -- so it did not exercise this path. Candidate follow-ups
  (UNTESTED, jeff's call): route the small ncv x ncv solve through the non-d&c
  steqr/sterf path on HIP, or fall back to host LAPACK syev (Lapack<T> host
  helpers are already linked), or report the rocSOLVER stedc convergence gap.
- Reportability: the original async-copy race (now fix-validated) remains a
  legitimate upstream raft portability bug worth filing against
  sparse/solver/detail/lanczos.cuh. The rocSOLVER stedc non-convergence is an
  AMD-library bug (rocSOLVER), not a raft bug. Do NOT open either upstream without
  jeff's approval.
- NOT pushed: the fix stays staged in agent_space/raft_lanczos; jeffdaily/raft
  HEAD and raft status.json are UNCHANGED (moving HEAD would force the passing
  platforms to revalidate; whether to push/upstream is jeff's call). Full
  build+test commands, per-variant table, and the NaN-vs-tolerance-vs-stedc
  classification are in projects/raft/notes.md "### Fix validation 2026-06-01
  (gfx90a)".

### B6 UPDATE (2026-06-01, fourth pass) -- DEVICE eigensolver (rocSOLVER syevj) converges and is CORRECT; the remaining wall is an upstream thick-restart Lanczos divergence

Built + ran on real gfx90a (GCD1, narrow). Swapped the stedc wall for a device
Jacobi eigensolver and isolated the residual failure with a standalone probe.

- THE EIGENSOLVE WALL IS REMOVED, fully on device. eigDC_legacy's USE_HIP branch
  now calls hipsolverDn{S,D}syevj (rocSOLVER one-sided Jacobi, rocsolver_{s,d}syevj)
  instead of syevd/STEDC, on the same dense ncv x ncv symmetric T. AMD_LOG_LEVEL=3
  confirms rocsolver::syevj_small_kernel dispatching and ZERO stedc_*/sytd2 kernels.
  No host LAPACK fallback -- the eigensolve stays entirely on GPU, per the CUDA
  intent (cusolverDnSsyevd is also a device d&c solver). The dev_info != 0
  non-convergence ASSERT no longer fires for the well-formed T. CUDA #else branch
  byte-identical (still cusolverDnsyevd). All syevj wrappers/aliases pre-existed
  (carried for eigJacobi); no new aliases needed. tol 1e-7, maxSweeps 100.
- WHY syevj converges where stedc didn't: STEDC's divide-and-conquer merge is
  fragile on the clustered/near-degenerate tridiagonal spectra Lanczos produces;
  Jacobi is iterative one-sided rotations (no d&c merge) and robust on clustered
  eigenvalues.
- BUT lanczos is still 0/11. Standalone probe (agent_space/lanczos_probe) +
  temporary in-loop instrumentation proved the eigensolve is now CORRECT and the
  failure is a THIRD-and-final wall in the UPSTREAM thick-restart loop, downstream
  of the eigensolve:
    PRE-LOOP (initial factorization + syevj): ev0 = -2.03697, EXACTLY the SA
      reference -2.0369630; res = 4.24e-05.
    1st restart: ev0 explodes to -1.21e9 (alpha/beta blow up).
    2nd restart: NaN. Final computed = -1.28e19 (vs ref -2.0369630).
  Because res=4.24e-05 > tol=1e-15 the upstream `while (res > tol ...)` restart
  loop runs, and its reorthogonalization corrupts the carried basis on the FIRST
  restart. Deterministic (identical bit patterns run-to-run; NOT the async-copy
  NaN race, which is already fixed -- this explosion is deterministic). On the
  larger Rmat case the diverged 1e19/Inf T then re-trips the eig.cuh ASSERT, but
  that is overflowed INPUT to syevj, not an eigensolver fault.
- CLASSIFICATION: the remaining bug is the upstream thick-restart Lanczos
  divergence already described in the second-pass note (Part B / restart
  reorthogonalization), now the SOLE blocker after both the async-copy race and the
  stedc convergence wall are removed. It is an upstream raft algorithm/code defect
  (candidate rapidsai/raft #3021 family; the restart loop in lanczos_smallest), NOT
  a ROCm library gap and NOT the eigensolver. SM/LM tolerance (#3021) is still
  masked -- the restart never produces a comparable result.
- CORRECTION to the third-pass note: "rocSOLVER stedc non-convergence is an
  AMD-library bug" -- the practical fix is NOT a rocSOLVER report but a one-line
  raft change (use the device Jacobi eigensolver for this solve on HIP). stedc is
  simply the wrong rocSOLVER routine for these spectra; syevj is the right one and
  converges. The true remaining upstream bug is the restart divergence, not stedc.
- The eig.cuh syevj swap is correct, necessary, and fully on-device; it should ride
  along whenever the upstream restart fix lands and lanczos can go green. Staged in
  agent_space/raft_lanczos only; jeffdaily/raft HEAD (70773a9) and raft status.json
  UNCHANGED (pushing would force the passing platforms to revalidate a non-green
  tip -- jeff's call). Per-variant table, probe trace, and build/test commands in
  projects/raft/notes.md "### Device-eigensolver fix 2026-06-01". Do NOT open
  anything upstream without jeff's approval.

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

### B7 UPDATE -- HIPRT feasibility probe (2026-06-01): HIPRT WORKS on gfx90a
- HIPRT 3.1.0 builds against ROCm 7.2.1 (~5s; source GPUOpen-LibrariesAndSDKs/HIPRT, Orochi vendored in-tree) and runs a minimal one-triangle trace CORRECTLY on gfx90a/MI250X (software BVH, HIPRT_RTIP=0, wave64 -- CDNA has no RT cores). The earlier "import hiprt fails / unproven" question is RESOLVED: HIP-RT is feasible here. (HIPRT = the RT SDK; hiprtc = the installed runtime compiler; HIPRT JIT-compiles its device kernels via Orochi->hiprtc.)
- OptiX -> HIPRT mapping: triangle GAS build ~1:1; raygen/closesthit/miss collapse into ONE HIP kernel; anyhit -> a device filter functor (idiomatic); payload is a real void*. Hard parts: the SBT -> functor-table restructure (light for single-record tracers, heavy for multi-shader) and re-validating any DIFFERENTIABLE backward traversal.
- Per-port effort: EnvGS Stage 2 = MEDIUM, the recommended FIRST HIPRT reimplementation (only raygen+anyhit, single SBT record, one triangle GAS; its anyhit is a t-sorted K-closest insert = textbook HIPRT functor; the real work is re-validating the ~1200-LOC differentiable backward + the torch-extension build wiring). rmcl = LARGE (OptiX lives in rmagine's backend ~6700 LOC / 5 sensor models / multiple SBT records -- do rmcl's HIPRT-free rmagine_cuda compute FIRST). splatad = NOT OptiX (spherical-coordinate rasterization, gsplat-class; remove from this cluster -- plain Strategy-B hipify).
- REAL UPSTREAM HIPRT BUG: the device name "AMD Instinct MI250X / MI250" contains a `/`, which std::filesystem::path treats as a directory separator, so HIPRT's JIT-cache file write fails (it never creates the implied subdir). One-line fix: sanitize `/` in the cache name. Without it HIPRT cannot run on ANY `/`-containing AMD device name. Reportable to GPUOpen/HIPRT (Report decision: PENDING). Probe + patch (gitignored): agent_space/hiprt_probe/.
- jeff's decision (2026-06-01): DEFER the OptiX-gated ports for now; HIPRT is proven feasible, so EnvGS Stage 2 is the recommended first reimplementation when revisited.

### B7 REFERENCE -- PBRT-v4 OptiX->HIPRT port (jeff + Daniel Meister, 2023): the exemplar
- jeff (jeffdaily) and Daniel Meister (first author of the HIPRT paper) did a real OptiX->HIPRT port of PBRT-v4 in 2023. It demonstrates the full reimplementation on a substantial renderer: the SBT -> functor-table restructure, the BVH build/traverse mapping, and the build wiring. Unmerged (Matt Pharr is at NVIDIA -- he tried), so it lives as a PR/branch, not in mmp/pbrt-v4 main.
- ACTION: when reimplementing the deferred OptiX-gated ports (EnvGS Stage 2 first, then rmcl), FEED THIS PR to the porter as the reference -- the same lever FAISS-ROCm was for raft's neighbors.
- URL: TBD. gh search / gh pr list on mmp/pbrt-v4 did not surface it (no jeffdaily-authored PR, no HIPRT match). Obtain the exact PR/branch from jeff, or search jeffdaily/pbrt-v4 fork branches + Daniel Meister's account.

---

## Build-config / environment (not bugs; for completeness)

- arrayfire `confidence_connected`: fails with AF_ERR_NOT_CONFIGURED because the
  headless build set AF_WITH_IMAGEIO=OFF (FreeImage absent); the test calls
  af::loadImage for its input and never reaches the GPU algorithm. FreeImage is a
  CPU-side third-party lib (nothing to do with ROCm). Closeable by installing
  libfreeimage-dev + rebuilding with AF_WITH_IMAGEIO=ON. Not an upstream finding.
