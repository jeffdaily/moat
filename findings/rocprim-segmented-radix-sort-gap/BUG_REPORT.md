# rocPRIM segmented radix sort leaves out-of-segment ("gap") elements undefined in the alternate DoubleBuffer (gfx90a)

## Component
rocPRIM segmented radix sort, reached via hipCUB `cub::DeviceSegmentedRadixSort::SortKeys` / `SortPairs` with a `cub::DoubleBuffer`.

## Summary
On gfx90a (CDNA2, ROCm 7.2.1), a segmented radix sort over a `DoubleBuffer` whose segment offsets do **not** cover the whole `[0, num_items)` range writes only the segment-covered elements into the buffer it selects, flips `Current()` to that (alternate) buffer, and leaves the **out-of-segment "gap" elements undefined** there. A caller that then copies the full `[0, num_items)` range back from `Current()` propagates those undefined gaps over its data. CUB on NVIDIA leaves the result such that the full-range copy-back is correct (gap elements retain the input values), so the same code is correct there. This is a behavioral divergence between rocPRIM and CUB for the DoubleBuffer + partial-coverage-segments case; it may be intentional (segments are the defined unit) but it is undocumented and differs from CUB.

## Environment
- AMD Instinct MI250X, gfx90a (CDNA2)
- ROCm 7.2.1, hipCUB -> rocPRIM

## Evidence (in the wild)
catboost's `catboost/cuda/cuda_util/kernel/segmented_sort.cu` sorts with a DoubleBuffer where segments need not tile the whole array; on rocPRIM the copy-back then carried garbage for gap elements. The HIP-only workaround seeds **both** DoubleBuffer halves with the input before the sort, so whichever buffer wins has correct gap elements:
```cpp
#if defined(USE_HIP)
    if (keys   && tmpKeys)   cudaMemcpyAsync(tmpKeys,   keys,   sizeof(K)*size, cudaMemcpyDeviceToDevice, stream);
    if (values && tmpValues) cudaMemcpyAsync(tmpValues, values, sizeof(V)*size, cudaMemcpyDeviceToDevice, stream);
#endif
```

## Minimal reproducer (to build before filing)
1. `num_items` with segments covering only part of the range (leave a gap, e.g. items `[N-k, N)` in no segment).
2. Init both DoubleBuffer halves to distinct sentinels; run `DeviceSegmentedRadixSort::SortKeys`.
3. Read back `d_keys.Current()` over the full `[0, num_items)`; inspect the gap region.
4. Expect (CUB) input values preserved in the gap; observe (rocPRIM) undefined/garbage.

## Expected behavior
Document the contract for out-of-segment elements in the selected DoubleBuffer, and ideally match CUB (gap elements defined). At minimum a clear statement that gaps are undefined would let callers avoid the surprise.

## Workaround
Seed both DoubleBuffer halves with the input (snippet above). HIP-only; CUDA/CUB path unchanged.

## Severity
Medium -- silent wrong data for the partial-coverage + full-range-copy-back pattern; possibly by-design but undocumented and divergent from CUB.

## Verdict (verified 2026-06-10): NOT filing -- most likely within-contract

`segmented_sort_gap.cpp` (in this dir) confirms the behavior: with int (ui32) keys, 2 segments leaving gaps, a DoubleBuffer with the alternate half pre-filled with a sentinel (999):
- end_bit 4,8,20,24 -> odd radix-pass count -> `Current()` flips to the alternate buffer -> gaps read back 999 (the sentinel) = UNDEFINED.
- end_bit 12,16,28,32 -> even passes -> `Current()` == input buffer -> gaps preserved.
Segments themselves always sort correctly.

This matches the catboost symptom, BUT segmented radix sort only defines behavior WITHIN segments; out-of-segment elements with a DoubleBuffer are plausibly unspecified in CUB as well (could not confirm a CUB *guarantee* of preservation without NVIDIA hardware). So this is most likely a portability gotcha (catboost relied on CUB incidentally preserving out-of-segment elements), not a rocPRIM defect. NOT filed. The catboost `segmented_sort.cu` gap-seed workaround (seed both DoubleBuffer halves) is correct defensive coding and stays.
