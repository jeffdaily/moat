# DRAFT -- upstream rapidsai/raft issue (awaiting jeff's approval to file)

Account: jeffdaily. Target: rapidsai/raft. Status: NOT FILED -- draft only.

---

**Title:** Lanczos: un-synchronized async D2H copy of alpha/beta read by host-pointer-mode axpy in `lanczos_aux`

## Summary
While porting raft to ROCm/HIP we isolated a latent correctness bug in the symmetric-eigensolver Lanczos iteration (`cpp/include/raft/sparse/solver/detail/lanczos.cuh`, `lanczos_aux`). The alpha/beta scalars are copied device->host with an asynchronous copy on the work stream and then immediately dereferenced on the host by a cuBLAS call in the default host pointer mode, with no stream synchronization between the copy and the read. On CUDA this is masked because a small pageable device-to-host `cudaMemcpyAsync` is effectively host-synchronous, so the host happens to read the completed value. Where the async copy is genuinely asynchronous, the host reads a stale scalar, the three-term recurrence loses orthogonality, and the iteration diverges to NaN.

## Location
`cpp/include/raft/sparse/solver/detail/lanczos.cuh`, in `lanczos_aux` (the same region PR #2918 left intact): `raft::copy(&b, <device beta>, 1, stream)` / `raft::copy(&alpha_i_host, <device alpha>, 1, stream)` followed, without a synchronize, by a `cublas<t>axpy` consuming `&b` / `&alpha_i_host` in `CUBLAS_POINTER_MODE_HOST`.

## Symptom
Deterministic divergence (alpha/beta -> NaN) of the Lanczos iteration when the host read races the copy. Identical in float and double (it is a synchronization bug, not precision).

## Fix
Synchronize the stream between the device->host scalar copies and the host-pointer-mode axpy that consumes them (e.g. `resource::sync_stream(handle, stream)`), or pass the scalars in device pointer mode. Correct on all platforms; leaves CUDA behavior unchanged.

## Related: thick-restart reorthogonalization divergence (likely the same root as #3021)
Separately: after fixing the race above, the thick-restart loop (`while (res > tol ...)`) still diverges. The initial factorization produces the exact reference eigenvalue, but the first restart's reorthogonalization corrupts the carried basis (the eigenvalue explodes, then NaN), deterministically and independent of platform. This appears to be the same root cause as #3021 (LanczosTestD_SM failing on A100). We have an isolated trace and can share it.

## How found
Isolated while porting raft to ROCm/HIP (gfx90a / MI250X, ROCm 7.2.1), with AI assistance. The async-copy race relies on undocumented CUDA host-synchronous behavior of small pageable D2H copies; the restart divergence reproduces on the algorithm itself.
