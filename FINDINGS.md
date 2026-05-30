# MOAT Findings

Upstream bugs and notable issues that MOAT porting work uncovered while porting CUDA projects to ROCm/HIP. Each links to a reproducer and, once filed, the upstream issue number.

| # | Title | Surfaced by | Arch / ROCm | Status | Upstream issue |
|---|-------|-------------|-------------|--------|----------------|
| 1 | HIP layered cudaArray collapses its layer dimension on cross-launch read | popsift | gfx90a / CDNA2, ROCm 7.2.1 | Confirmed (multi-layer repro); root cause corrected | [ROCm/clr#275](https://github.com/ROCm/clr/issues/275) |
| 2 | Integer atomicMin/atomicMax silently no-op on host-coherent (fine-grained) hipMallocManaged memory | cudaKDTree | gfx90a / CDNA2, ROCm 7.2.1 | Expected (not a bug): documented PCIe/fine-grained atomic limitation; standalone repro + ISA + docs confirm | N/A (usage rule) |
| 3 | rocPRIM/hipCUB DeviceRadixSort loses keys when begin_bit>0 and end_bit==key bit-width | cudaKDTree | gfx90a / CDNA2, ROCm 7.2.1 | CONFIRMED (clean repro: keys lost; both hipCUB+rocPRIM, multi-seed/size) | [ROCm/rocPRIM#775](https://github.com/ROCm/rocPRIM/issues/775) |

## 1. HIP layered cudaArray collapses its layer dimension on cross-launch read

A multi-layer `cudaArrayLayered` (+ `cudaArraySurfaceLoadStore`) float array, written one layer per kernel launch via `surf2DLayeredwrite`, returns the LAST-written layer's data for EVERY layer index on any LATER-launch read -- `tex2DLayered`, `surf2DLayeredread`, AND host `hipMemcpy3D` all collapse the layer dimension. `hipDeviceSynchronize` between launches and recreating the texture/surface do not help. CUDA preserves the per-layer contents across launches, so in-spec code is mishandled.

CORRECTION: an earlier "the texture is stale but `surf2DLayeredread` is fresh" finding was an artifact of a repro that touched only ONE layer -- so the last-written layer equalled the layer read. With >=2 layers, surface reads collapse too, so reading via the surface does NOT fix it. A NON-layered 3D array (`hipMalloc3DArray` WITHOUT `cudaArrayLayered`, accessed via `surf3Dwrite`/`surf3Dread`/`tex3D` with the level as a real z coordinate) is fully per-slice coherent across launches (so is a tall 2D array, W x H*L).

- Reproducers (decisive, multi-layer): `findings/popsift-texsurf-coherency/{multilayer_check.cpp,multilayer_check2.cpp,array3d_check.cpp,tall2d_check.cpp}`
- Earlier single-layer repro (misleading, kept for the record): `findings/popsift-texsurf-coherency/repro.cpp`
- FILED: ROCm/clr#275 (https://github.com/ROCm/clr/issues/275). Self-contained report (full repros inline): `findings/popsift-texsurf-coherency/BUG_REPORT.md`.
- Port workaround: drop `cudaArrayLayered` on HIP; use a non-layered 3D array (`surf3Dwrite`/`surf3Dread`/`tex3D`, level as z). In popsift this is one compat-header wrapper (`surf2DLayeredwrite` -> `surf3Dwrite`) plus routing reads through the fetch helper to `surf3Dread`.

## 2. Integer atomicMin/atomicMax silently no-op on host-coherent (fine-grained) hipMallocManaged memory -- EXPECTED, not a bug

VERDICT: **EXPECTED (documented hardware limitation), not a ROCm runtime/compiler defect.** On gfx90a, plain `atomicMin`/`atomicMax(int*)` compile to the hardware `global_atomic_smin`/`smax` instructions, which AMD documents as a **NOP over the PCIe / host-coherent path**. The default `hipMallocManaged` buffer (and `hipHostMalloc`) is **fine-grained host-coherent** memory (NOT coarse-grained -- the original title mislabeled it; confirmed via `hipMemRangeAttributeCoherencyMode`), so the integer min/max RMW is silently dropped. `atomicAdd` and `atomicCAS` work on the same buffer because add/cmpswap *are* serviceable over that path (LLVM AMDGPU expands min/max but explicitly not add/xchg). In cudaKDTree's spatial builder this left the bounding box empty -> all split planes -FLT_MAX -> degenerate tree -> OOB fault.

The real discriminator is host-coherent vs device-local, not coarse vs fine: `hipExtMallocWithFlags(hipDeviceMallocFinegrained)` is fine-grained device VRAM and atomicMin/Max WORK there; only the host-coherent allocations NOP. The fix is the CAS-loop emulation (or `atomicMin_system`, which the compiler expands to a cmpswap loop, or keeping the accumulator in device-local memory). This is a USAGE requirement; do not file a correctness bug.

- Surfaced by: cudaKDTree spatial builder (atomic_grow).
- Port workaround: atomicCAS-loop emulation (atomicMinI32/atomicMaxI32/atomicMinU32), HIP-guarded -- correct and portable; keep it.
- Reproducer + full writeup: `findings/cudakdtree-atomic-minmax/{repro.hip,coherence_probe.hip,EXPLANATION.md}` (variation table, ISA evidence, doc citations).
- Citations: ROCm "Hardware atomics operation support" (PCIe min/max = NOP); amd-lab-notes mi200-memory-space (managed default = fine-grained); LLVM llvm-project#122137 (add/xchg not expanded, min/max are).
- Confirmation: DONE (standalone repro + ISA + doc/spec check). Mechanism label corrected from "coarse-grained" to "fine-grained host-coherent".

## 3. rocPRIM/hipCUB DeviceRadixSort loses keys when begin_bit>0 and end_bit==key bit-width -- CONFIRMED

VERDICT: CONFIRMED real bug (clean standalone repro). On gfx90a / ROCm 7.2.1 (hipCUB 4.2.0, rocPRIM 4.2.0), `rocprim::radix_sort_keys` -- and `hipcub::DeviceRadixSort::SortKeys` which wraps it -- gives a WRONG result when sorting a key sub-range with begin_bit > 0 AND end_bit == 8*sizeof(KeyT) (the full key width): the output is NOT a permutation of the input (keys lost/duplicated) and is misordered. Any end_bit < width is correct even with begin_bit > 0 ([32,48), [24,56), [8,24) all pass), and the full-key sort [0,width) is correct. Confirmed for [32,64),[40,64),[48,64),[56,64) on uint64, both hipCUB and rocPRIM, N=10000 and N=1000000, multiple seeds. The defect is in rocPRIM; hipCUB inherits it.

CORRECTION/history: the original cudaKDTree note framed this as "mis-sorts"; a first confirmation agent's repros were unreliable (their own full-key sanity sort itself failed -- an artifact of comparing the FULL key after a PARTIAL-range sort, which only orders the selected bits), which initially looked like a test artifact. A correct repro (compare only the [begin_bit,end_bit) subkey and verify the output is a permutation of the input) shows it is a genuine key-loss bug for the begin_bit>0 & end_bit==width case.

- Surfaced by: cudaKDTree spatial-kdtree.h (sorts a 64-bit (nodeID<<32)|primID key by the high 32 bits via [32,64) -- exactly the trigger).
- Port workaround: sort the full 64-bit key ([0,64)) -- correct; keep it.
- Reproducers (clean): findings/hipcub-rocprim-beginbit/{sweep.cpp,minrepro.cpp,bbtest.cpp,sweep_out.txt}.
- FILED: ROCm/rocPRIM#775 (https://github.com/ROCm/rocPRIM/issues/775). Self-contained report (full repro inline): findings/hipcub-rocprim-beginbit/BUG_REPORT.md.
- Confirmation: DONE.

## Feature requests filed

- hipSOLVER low-level cusolverSp sparse factorization API (csrqr*/csrchol*, csrlsvluHost, csrsymrcm) -- https://github.com/ROCm/hipSOLVER/issues/443 -- surfaced by RXMesh; body at findings/hipsolver-sparse-fr/REQUEST.md. (High-level csrlsvqr/csrlsvchol already ship in hipSOLVER; cuDSS is closed-source and covered on ROCm by STRUMPACK.)
