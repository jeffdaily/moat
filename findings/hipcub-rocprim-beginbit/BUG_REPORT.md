# rocPRIM / hipCUB DeviceRadixSort loses keys when begin_bit > 0 and end_bit == key bit-width

## Summary
On gfx90a (CDNA2), `rocprim::radix_sort_keys` -- and `hipcub::DeviceRadixSort::SortKeys`, which wraps it -- produces a WRONG result when sorting a sub-range of the key bits with `begin_bit > 0` AND `end_bit == 8*sizeof(KeyT)` (the full key width): the output is not a permutation of the input (keys are lost/duplicated) and is misordered. Any `end_bit < width` is correct (even with `begin_bit > 0`), and the full-key sort (`begin_bit=0, end_bit=width`) is correct. The same code path is correct on NVIDIA CUB.

## Environment
AMD Instinct MI250X, gfx90a (CDNA2); ROCm 7.2.1; hipCUB 4.2.0; rocPRIM 4.2.0.

## Reproducer
`sweep.cpp` (build: `hipcc -O2 sweep.cpp -o sweep`; run: `./sweep`). For each bit range it sorts uint64 keys, then checks (a) the selected `[begin_bit,end_bit)` subkey is non-decreasing and (b) the output is a permutation of the input.

## Observed (uint64, N=1,000,000, seed=12345; "perm" = output is a permutation of input)
```
[ 0,64): subkey-inv=0       perm=yes      [ 0,32): subkey-inv=0  perm=yes
[ 8,24): subkey-inv=0       perm=yes      [16,40): subkey-inv=0  perm=yes
[16,48): subkey-inv=0       perm=yes      [24,56): subkey-inv=0  perm=yes
[32,48): subkey-inv=0       perm=yes      [32,56): subkey-inv=0  perm=yes
[32,64): subkey-inv=469745  perm=NO  (KEYS LOST)
[40,64): subkey-inv=444813  perm=NO  (KEYS LOST)
[48,64): subkey-inv=450697  perm=NO  (KEYS LOST)
[56,64): subkey-inv=358455  perm=NO  (KEYS LOST)
```
Reproduces identically via `rocprim::radix_sort_keys` directly, on a second seed, and at N=10000. `minrepro.cpp` confirms the full-key sort is correct (sanity).

## Trigger
`begin_bit > 0` AND `end_bit == 8*sizeof(KeyT)`. The defect is in rocPRIM (hipCUB inherits it). Likely a digit-extraction / pass-count boundary error when the top bit is included with a nonzero start bit.

## Expected
A partial-range radix sort must order keys by the selected bit range and must always output a permutation of the input (CUDA CUB does).

## Impact
Surfaced porting cudaKDTree to ROCm: its spatial k-d tree sorts a 64-bit `(nodeID<<32)|primID` key by the high 32 bits via `[32,64)`, losing/duplicating leaf primitive indices and corrupting nearest-neighbor queries. Workaround: sort the full key (`[0,64)`).
