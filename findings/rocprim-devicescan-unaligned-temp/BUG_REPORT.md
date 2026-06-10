# rocPRIM DeviceScan returns non-deterministic wrong results when d_temp_storage is not 8-byte aligned (address mod 8 in {4,5,6}) on gfx90a

## Component
rocPRIM `DeviceScan` (decoupled-lookback), reached via hipCUB `cub::DeviceScan::ExclusiveSum` / `InclusiveSum`.

## Summary
On gfx90a (CDNA2, ROCm 7.2.1), `DeviceScan` produces **non-deterministic wrong output** when the caller-provided `d_temp_storage` pointer has certain sub-8-byte misalignments -- specifically when `((uintptr_t)d_temp_storage % 8)` is 4, 5, or 6. Roughly 10-60% of the output elements are wrong, clustered at tile boundaries, and both the count and the locations vary from run to run on identical input. CUB on NVIDIA has no such sensitivity: any `d_temp_storage` of at least `temp_storage_bytes` produces the correct result, which is the documented contract. rocPRIM's internal temp-storage layout (the lookback tile-state placed inside `d_temp_storage`) evidently assumes an alignment of the base pointer that CUB does not require, and `temp_storage_bytes` does not over-allocate to absorb a misaligned base, so a caller sub-allocating its scratch from a larger buffer at an arbitrary offset gets silent corruption rather than an error.

## Environment
- AMD Instinct MI250X, gfx90a (CDNA2)
- ROCm 7.2.1, hipCUB -> rocPRIM

## Reproducer
`devicescan_align.cpp` in this directory. Build/run:
```
hipcc -O2 --offload-arch=gfx90a devicescan_align.cpp -o devicescan_align
HIP_VISIBLE_DEVICES=0 ./devicescan_align
```
It runs `ExclusiveSum` of 2^20 ones (correct result is `out[i] == i`) with `d_temp_storage` placed at byte offset `off` past a (256-aligned) `hipMalloc` base.

Observed (bad = count of wrong elements; firstbad = first wrong index):
```
off  0..3   -> bad=0            (mod 8 = 0,1,2,3 : correct)
off  4      -> bad=252416 / 379136 (two runs; mod 8 = 4 : CORRUPT, non-deterministic)
off  5      -> bad=363776 / 162560 (mod 8 = 5 : CORRUPT)
off  6      -> bad=412160 / 333056 (mod 8 = 6 : CORRUPT)
off  7,8..11,15,16,24,32,64,256 -> bad=0   (mod 8 = 7,0,1,2,3 : correct)
off 12,13,14 -> CORRUPT          (mod 8 = 4,5,6)
off 20,28,36,68,132,260 -> CORRUPT (all mod 8 = 4)
```
Pattern: **corrupt iff `(d_temp_storage % 8)` in {4,5,6}; correct for residues {0,1,2,3,7}.** An 8-byte-aligned `d_temp_storage` (residue 0) is always correct. The corruption is non-deterministic (race-like, at tile boundaries), consistent with torn accesses to the lookback tile-state.

## Evidence (in the wild)
catboost's GPU split-points kernel (`catboost/cuda/methods/greedy_subsets_searcher/kernel/split_points.cu`) sub-allocates the `DeviceScan` temp from a shared buffer at byte offset `4*part.Size`; when that offset lands on a {4,5,6}-mod-8 address the scan corrupts and split indices come out wrong, non-deterministically (its `BinBuilderTest` fails intermittently). Aligning the temp offset up fixed it.

## Expected behavior
`DeviceScan` should accept any `d_temp_storage` of at least `temp_storage_bytes` regardless of its alignment, matching CUB. Failing that, the alignment requirement must be documented and `temp_storage_bytes` increased so the algorithm can self-align within the returned size. Silent, non-deterministic corruption with no error is the worst outcome.

## Workaround
Align `d_temp_storage` to 8 bytes (round the sub-allocation offset up to a multiple of 8). catboost rounds up to 256, which also works (a superset). HIP-only; the CUDA/CUB path needs no change.

## Severity
High -- silent, non-deterministic data corruption with no error returned, triggered by an alignment that CUB explicitly permits.
