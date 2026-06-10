# rocPRIM SortPairs faults (hipErrorInvalidValue) when run with temp storage sized by a SortKeys query (gfx90a)

## Component
rocPRIM segmented radix sort temp-storage sizing, via hipCUB `cub::DeviceSegmentedRadixSort::SortKeys` (sizing query) vs `SortPairs` (run).

## Summary
On gfx90a (CDNA2, ROCm 7.2.1), querying `temp_storage_bytes` through the **SortKeys** path (values pointer null) and then performing a **SortPairs** run (values pointer non-null) faults at run time with `hipErrorInvalidValue`, because rocPRIM's SortPairs needs more temp storage than the SortKeys sizing returned. CUB on NVIDIA tolerates this (its keys-only and pairs sizing are effectively interchangeable for the caller here), so the same code runs there.

Note: this is partly a caller contract issue -- strictly, the sizing query should match the keys-vs-pairs path of the actual run. It is filed to (a) confirm rocPRIM's documented contract on this point and (b) flag that the rocPRIM-vs-CUB difference turns a latent caller bug into a hard fault, which is a porting hazard worth a doc note or a more lenient sizing.

## Environment
- AMD Instinct MI250X, gfx90a (CDNA2)
- ROCm 7.2.1, hipCUB -> rocPRIM

## Evidence (in the wild)
catboost's `catboost/cuda/cuda_util/segmented_sort.cpp` sized the temp via a `(V*)nullptr` (SortKeys-shaped) query but ran a pairs sort; on rocPRIM this faulted. The HIP-only fix sizes for the actual pairs path:
```cpp
// size for the SAME keys-vs-pairs path the Run will take
auto valuesForSizing = /* USE_HIP */ Values.Get() /* else (V*)nullptr */;
```

## Expected behavior
Document whether `temp_storage_bytes` is required to be queried through the same keys-vs-pairs entry point as the run (and if so, that a mismatch is a usage error rather than UB). Optionally, size SortKeys/SortPairs identically (as CUB effectively does for this caller) to remove the porting hazard.

## Workaround
Query the temp size through the SortPairs path (pass the real values pointer) when a pairs sort will run. HIP-only; CUDA/CUB path keeps the original `(V*)nullptr` sizing.

## Severity
Low-Medium -- partly a caller CUB-contract mis-use, but rocPRIM converts it into a hard hipErrorInvalidValue where CUB silently tolerates it; confirm the contract / consider lenient sizing.
