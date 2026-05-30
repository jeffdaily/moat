# [Feature request] hipSOLVER: low-level cusolverSp sparse factorization API (csrqr*/csrchol*), csrlsvluHost, and csrsymrcm reordering

## Summary
hipSOLVER's `hipsolverSp` already provides the high-level sparse direct solves `csrlsvqr` and `csrlsvchol` (S/D/C/Z, backed by rocSPARSE + SuiteSparse) -- thanks, that covers the common case. Porting CUDA libraries that use the *lower-level* cuSOLVER sparse API still has no ROCm path, however. This requests the remaining `cusolverSp` surface so a "factor once, solve many" CUDA app can move to hipSOLVER without restructuring:

- **Low-level QR factorization:** `cusolverSpXcsrqrBufferInfo` / `csrqrSetup` / `csrqrFactor` / `csrqrSolve` / `csrqrZeroPivot` (host + device variants)
- **Low-level Cholesky factorization:** `cusolverSpXcsrcholBufferInfo` / `csrcholFactor` / `csrcholSolve` / `csrcholZeroPivot`
- **LU solve:** `cusolverSpXcsrlsvluHost` (the QR and Cholesky `csrlsv*` exist; LU does not)
- **Reordering:** `cusolverSpXcsrsymrcm` (reverse Cuthill-McKee); `csrsymmdq` / `csrsymamd` / `csrmetisnd` would round it out

## Why
A CUDA library that factors a sparse matrix once (`csrqr*`/`csrchol*`) and re-solves repeatedly, or that reorders with `csrsymrcm` before factoring, currently cannot map onto hipSOLVER -- only the all-in-one `csrlsv*` solves are available, which re-factor each call. Filling the low-level API would let these port cleanly.

## Context
Found while porting RXMesh (a CUDA mesh-processing library) to ROCm/HIP as part of MOAT (https://github.com/jeffdaily/moat). Its solver/autodiff path uses the low-level `csrqr`/`csrchol` API plus `csrsymrcm`, which have no hipSOLVER equivalent today.

## Note on a full direct sparse solver (cuDSS)
The same RXMesh path also uses NVIDIA cuDSS. cuDSS is proprietary/closed-source, so it cannot simply be ported; for a cuDSS-class GPU *direct* sparse solver on ROCm the open option is STRUMPACK (BSD, already HIP/ROCm-capable, runs on Frontier). This request is intentionally scoped to the smaller, self-contained `cusolverSp` compatibility gap in hipSOLVER, not a cuDSS-equivalent library.
