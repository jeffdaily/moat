# rocPRIM DeviceScan corrupts the scan non-deterministically when its temp storage is not 256-byte aligned (gfx90a)

## Component
rocPRIM `DeviceScan` (lookback tile-state), reached via hipCUB `cub::DeviceScan::InclusiveScan` / `ExclusiveScan`.

## Summary
On gfx90a (CDNA2, ROCm 7.2.1), `DeviceScan` produces **wrong, non-deterministic results at tile boundaries** when the caller hands it a `d_temp_storage` pointer that is not 256-byte aligned. The decoupled-lookback tile-state that rocPRIM places at the start of the temp buffer appears to assume a minimum alignment; at an unaligned base the tile-state reads/writes tear at tile boundaries and the scan output is corrupted differently from run to run. CUB on NVIDIA has no such requirement: the same caller-provided unaligned pointer scans correctly. The `temp_storage_bytes` query does not over-allocate to absorb a misaligned base, so a caller that sub-allocates the temp from a larger buffer at an arbitrary offset silently gets corruption rather than an error.

## Environment
- AMD Instinct MI250X, gfx90a (CDNA2)
- ROCm 7.2.1, hipCUB -> rocPRIM

## Evidence (in the wild)
catboost's GPU split-points kernel (`catboost/cuda/methods/greedy_subsets_searcher/kernel/split_points.cu`) shares a single scratch buffer between an int scan output region `[0, 4*part.Size)` and the `DeviceScan` working temp placed immediately after it, at byte offset `4*part.Size`. When `part.Size` makes that offset non-256-aligned, the scan corrupts and the resulting split indices are wrong, non-deterministically (catboost's `BinBuilderTest` / split tests fail intermittently). Forcing the temp to start at the next 256-aligned offset makes it correct and deterministic:
```cpp
// workaround in split_points.cu
const ui64 scanTempOffset = (tempOffsetsSize + 255) & ~static_cast<ui64>(255);
auto scanTmp = tempStorage ? (void*)(tempStorage + scanTempOffset) : nullptr;
```

## Minimal reproducer (to build before filing)
A standalone repro is straightforward and should be built/attached before filing:
1. Allocate one device buffer; place the `DeviceScan` `d_temp_storage` at a deliberately non-256-aligned offset inside it (e.g. base + 4 bytes).
2. `cub::DeviceScan::ExclusiveSum` over N ints spanning several tiles (N large enough for multiple lookback tiles, e.g. >= 1<<16).
3. Compare against a host prefix-sum; expect mismatches clustered at tile boundaries, varying across runs.
4. Repeat with a 256-aligned temp base; expect a correct, deterministic result.

## Expected behavior
Either (a) `DeviceScan` works correctly for any caller-provided `d_temp_storage` (handle alignment internally), matching CUB, or (b) the alignment requirement is documented AND `temp_storage_bytes` is increased so the algorithm can self-align inside the returned size. Silent non-deterministic corruption is the worst outcome.

## Workaround
256-byte-align the `d_temp_storage` base offset (see snippet above). HIP-only; the CUDA/CUB path needs no change.

## Severity
High -- silent, non-deterministic data corruption with no error returned.
