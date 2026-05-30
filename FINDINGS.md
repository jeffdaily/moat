# MOAT Findings

Upstream bugs and notable issues that MOAT porting work uncovered while porting CUDA projects to ROCm/HIP. Each links to a reproducer and, once filed, the upstream issue number.

| # | Title | Surfaced by | Arch / ROCm | Status | Upstream issue |
|---|-------|-------------|-------------|--------|----------------|
| 1 | HIP layered-image texture cache not invalidated at kernel-launch boundary | popsift | gfx90a / CDNA2, ROCm 7.x | Confirmed, repro ready | PENDING |
| 2 | Integer atomicMin/atomicMax silently no-op on coarse-grained hipMallocManaged memory | cudaKDTree | gfx90a / CDNA2, ROCm 7.2.1 | Candidate (agent micro-test; needs standalone repro + spec check) | PENDING |
| 3 | hipCUB DeviceRadixSort::SortKeys mis-sorts with nonzero begin_bit | cudaKDTree | gfx90a / CDNA2, ROCm 7.2.1 | Candidate (agent direct test: 9974/10000 misordered; needs standalone repro) | PENDING |

## 1. HIP layered-image texture cache not invalidated at kernel-launch boundary

A `tex2DLayered` fetch does not observe a `surf2DLayeredwrite` made to the same `cudaArrayLayered` array by a PRIOR kernel launch on the same stream (returns stale/pre-write data); a `surf2DLayeredread` of the same location is fresh. A non-layered 2D array on the same code path is coherent; `hipDeviceSynchronize` and recreating the texture do not help. CUDA and HIP both document cross-launch texture/surface coherency, so in-spec separate-launch code is mishandled. Isolated to the layered-image path; not a write-flush, stale-descriptor, or hardware limit.

- Reproducer: `findings/popsift-texsurf-coherency/repro.cpp`
- Full report: `findings/popsift-texsurf-coherency/BUG_REPORT.md`
- Port workaround: read the consumer via `surf2DLayeredread` (popsift make_dog).

## 2. Integer atomicMin/atomicMax silently no-op on coarse-grained hipMallocManaged memory

On gfx90a, `atomicMin`/`atomicMax` on 32-bit integers in coarse-grained `hipMallocManaged` memory appear to no-op (the value is not updated), while `atomicAdd` and `atomicCAS` on the same memory work. In cudaKDTree's spatial builder this left the bounding box empty, so all split planes became -FLT_MAX, producing a degenerate tree and an OOB fault. Needs a standalone repro and a spec check before filing: coarse-grained vs fine-grained allocation and atomic memory scope on CDNA may make this expected (a usage requirement) rather than a runtime bug -- confirm against the HIP atomics / memory-coherence docs first, the same way bug 1 was confirmed in-spec.

- Surfaced by: cudaKDTree spatial builder (atomic_grow).
- Port workaround: atomicCAS-loop emulation (atomicMinI32/atomicMaxI32/atomicMinU32), HIP-guarded.
- Confirmation: TODO (standalone repro + doc/spec check).

## 3. hipCUB DeviceRadixSort::SortKeys mis-sorts with a nonzero begin_bit

On gfx90a / ROCm 7.2.1, `hipcub::DeviceRadixSort::SortKeys` with a nonzero `begin_bit` in the [32,64) range mis-sorts (a direct test showed 9974/10000 keys out of order), which corrupted the spatial k-d tree's leaf primitive ranges so queries missed the true nearest neighbor. Sorting the full 64-bit key (begin_bit=0) is correct. Needs a standalone repro isolating the begin_bit parameter (and a check of whether rocPRIM underneath has the same defect).

- Surfaced by: cudaKDTree spatial-kdtree.h sort.
- Port workaround: sort the full 64-bit key on HIP.
- Confirmation: TODO (standalone repro varying begin_bit).
