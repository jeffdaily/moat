# MOAT Findings

Upstream bugs and notable issues that MOAT porting work uncovered while porting CUDA projects to ROCm/HIP. Each links to a reproducer and, once filed, the upstream issue number.

| # | Title | Surfaced by | Arch / ROCm | Status | Upstream issue |
|---|-------|-------------|-------------|--------|----------------|
| 1 | HIP layered cudaArray collapses its layer dimension on cross-launch read | popsift | gfx90a / CDNA2, ROCm 7.2.1 | Confirmed (multi-layer repro); root cause corrected | PENDING |
| 2 | Integer atomicMin/atomicMax silently no-op on coarse-grained hipMallocManaged memory | cudaKDTree | gfx90a / CDNA2, ROCm 7.2.1 | Candidate (agent micro-test; needs standalone repro + spec check) | PENDING |
| 3 | hipCUB DeviceRadixSort::SortKeys mis-sorts with nonzero begin_bit | cudaKDTree | gfx90a / CDNA2, ROCm 7.2.1 | Candidate (agent direct test: 9974/10000 misordered; needs standalone repro) | PENDING |

## 1. HIP layered cudaArray collapses its layer dimension on cross-launch read

A multi-layer `cudaArrayLayered` (+ `cudaArraySurfaceLoadStore`) float array, written one layer per kernel launch via `surf2DLayeredwrite`, returns the LAST-written layer's data for EVERY layer index on any LATER-launch read -- `tex2DLayered`, `surf2DLayeredread`, AND host `hipMemcpy3D` all collapse the layer dimension. `hipDeviceSynchronize` between launches and recreating the texture/surface do not help. CUDA preserves the per-layer contents across launches, so in-spec code is mishandled.

CORRECTION: an earlier "the texture is stale but `surf2DLayeredread` is fresh" finding was an artifact of a repro that touched only ONE layer -- so the last-written layer equalled the layer read. With >=2 layers, surface reads collapse too, so reading via the surface does NOT fix it. A NON-layered 3D array (`hipMalloc3DArray` WITHOUT `cudaArrayLayered`, accessed via `surf3Dwrite`/`surf3Dread`/`tex3D` with the level as a real z coordinate) is fully per-slice coherent across launches (so is a tall 2D array, W x H*L).

- Reproducers (decisive, multi-layer): `findings/popsift-texsurf-coherency/{multilayer_check.cpp,multilayer_check2.cpp,array3d_check.cpp,tall2d_check.cpp}`
- Earlier single-layer repro (misleading, kept for the record): `findings/popsift-texsurf-coherency/repro.cpp`
- Report: `findings/popsift-texsurf-coherency/BUG_REPORT.md` -- PREDATES the corrected root cause; regenerate from the multi-layer repros before filing.
- Port workaround: drop `cudaArrayLayered` on HIP; use a non-layered 3D array (`surf3Dwrite`/`surf3Dread`/`tex3D`, level as z). In popsift this is one compat-header wrapper (`surf2DLayeredwrite` -> `surf3Dwrite`) plus routing reads through the fetch helper to `surf3Dread`.

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
