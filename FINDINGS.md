# MOAT Findings

Upstream bugs and notable issues that MOAT porting work uncovered while porting CUDA projects to ROCm/HIP. Each links to a reproducer and, once filed, the upstream issue number.

| # | Title | Surfaced by | Arch / ROCm | Status | Upstream issue |
|---|-------|-------------|-------------|--------|----------------|
| 1 | HIP layered-image texture cache not invalidated at kernel-launch boundary | popsift | gfx90a / CDNA2, ROCm 7.x | Confirmed, repro ready | PENDING |

## 1. HIP layered-image texture cache not invalidated at kernel-launch boundary

A `tex2DLayered` fetch does not observe a `surf2DLayeredwrite` made to the same `cudaArrayLayered` array by a PRIOR kernel launch on the same stream (returns stale/pre-write data); a `surf2DLayeredread` of the same location is fresh. A non-layered 2D array on the same code path is coherent; `hipDeviceSynchronize` and recreating the texture do not help. CUDA and HIP both document cross-launch texture/surface coherency, so in-spec separate-launch code is mishandled. Isolated to the layered-image path; not a write-flush, stale-descriptor, or hardware limit.

- Reproducer: `findings/popsift-texsurf-coherency/repro.cpp`
- Full report: `findings/popsift-texsurf-coherency/BUG_REPORT.md`
- Port workaround: read the consumer via `surf2DLayeredread` (popsift make_dog).
