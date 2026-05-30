# DeviceRadixSort / radix_sort_keys loses keys when begin_bit > 0 and end_bit == key bit-width (gfx90a)

## Component
rocPRIM device radix sort (and hipCUB `DeviceRadixSort`, which wraps rocPRIM).

## Summary
On gfx90a (CDNA2, ROCm 7.2.1), `rocprim::radix_sort_keys` -- and `hipcub::DeviceRadixSort::SortKeys`, which wraps it -- produces a WRONG result when sorting a sub-range of the key bits with `begin_bit > 0` AND `end_bit == 8*sizeof(KeyT)` (the full key width): the output is **not a permutation of the input** (keys are lost/duplicated) and is misordered. Any `end_bit < width` is correct (even with `begin_bit > 0`), and the full-key sort (`begin_bit=0, end_bit=width`) is correct. The same calls are correct on NVIDIA CUB.

## Environment
- AMD Instinct MI250X, gfx90a (CDNA2)
- ROCm 7.2.1; hipCUB 4.2.0; rocPRIM 4.2.0

## Reproducer
Build: `hipcc -O2 sweep.cpp -o sweep`  Run: `./sweep`
For each bit range it sorts 1,000,000 random `uint64` keys, then checks (a) the selected `[begin_bit,end_bit)` subkey is non-decreasing and (b) the sorted output is a permutation of the input.

```cpp
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <rocprim/rocprim.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#define CK(c) do{hipError_t e=(c); if(e!=hipSuccess){fprintf(stderr,"HIP %d %s\n",__LINE__,hipGetErrorString(e));std::exit(2);}}while(0)
static void run(size_t N,uint64_t seed,int bb,int eb,bool useRocprim){
  std::mt19937_64 rng(seed); std::vector<uint64_t> in(N); for(auto&x:in)x=rng();
  uint64_t *di,*do_; CK(hipMalloc(&di,N*8)); CK(hipMalloc(&do_,N*8));
  CK(hipMemcpy(di,in.data(),N*8,hipMemcpyHostToDevice));
  void* dt=nullptr; size_t tb=0;
  if(useRocprim){ CK(rocprim::radix_sort_keys(dt,tb,di,do_,N,bb,eb,0)); CK(hipMalloc(&dt,tb)); CK(rocprim::radix_sort_keys(dt,tb,di,do_,N,bb,eb,0)); }
  else { CK(hipcub::DeviceRadixSort::SortKeys(dt,tb,di,do_,(int)N,bb,eb)); CK(hipMalloc(&dt,tb)); CK(hipcub::DeviceRadixSort::SortKeys(dt,tb,di,do_,(int)N,bb,eb)); }
  CK(hipDeviceSynchronize());
  std::vector<uint64_t> out(N); CK(hipMemcpy(out.data(),do_,N*8,hipMemcpyDeviceToHost));
  int w=eb-bb; uint64_t m=(w>=64)?~0ull:((1ull<<w)-1);
  size_t inv=0; for(size_t i=1;i<N;++i){uint64_t a=(out[i-1]>>bb)&m,b=(out[i]>>bb)&m; if(a>b)++inv;}
  std::vector<uint64_t> si=in,so=out; std::sort(si.begin(),si.end()); std::sort(so.begin(),so.end());
  printf("  N=%-8zu seed=%-3llu [%2d,%2d) %-7s: subkey-inv=%-7zu perm=%s\n",N,(unsigned long long)seed,bb,eb,useRocprim?"rocPRIM":"hipCUB",inv,(si==so)?"yes":"NO-KEYS-LOST");
  CK(hipFree(di));CK(hipFree(do_));CK(hipFree(dt));
}
int main(){
  int ranges[][2]={{0,64},{0,32},{8,24},{16,40},{16,48},{24,56},{32,48},{32,56},{32,64},{40,64},{48,64},{56,64}};
  printf("uint64 -- subkey-inv MUST be 0 and perm MUST be yes\n");
  for(auto&r:ranges) run(1000000,12345,r[0],r[1],false);
  printf("-- cross-checks (rocPRIM directly + 2nd seed + smaller N) --\n");
  run(1000000,777,32,64,false); run(1000000,777,32,64,true); run(10000,5,40,64,false);
  return 0;
}
```

## Observed
```
  N=1000000  seed=12345 [ 0,64) hipCUB : subkey-inv=0       perm=yes
  N=1000000  seed=12345 [ 0,32) hipCUB : subkey-inv=0       perm=yes
  N=1000000  seed=12345 [ 8,24) hipCUB : subkey-inv=0       perm=yes
  N=1000000  seed=12345 [16,40) hipCUB : subkey-inv=0       perm=yes
  N=1000000  seed=12345 [16,48) hipCUB : subkey-inv=0       perm=yes
  N=1000000  seed=12345 [24,56) hipCUB : subkey-inv=0       perm=yes
  N=1000000  seed=12345 [32,48) hipCUB : subkey-inv=0       perm=yes
  N=1000000  seed=12345 [32,56) hipCUB : subkey-inv=0       perm=yes
  N=1000000  seed=12345 [32,64) hipCUB : subkey-inv=469745  perm=NO-KEYS-LOST
  N=1000000  seed=12345 [40,64) hipCUB : subkey-inv=444813  perm=NO-KEYS-LOST
  N=1000000  seed=12345 [48,64) hipCUB : subkey-inv=450697  perm=NO-KEYS-LOST
  N=1000000  seed=12345 [56,64) hipCUB : subkey-inv=358455  perm=NO-KEYS-LOST
-- cross-checks (rocPRIM directly + 2nd seed + smaller N) --
  N=1000000  seed=777 [32,64) hipCUB : subkey-inv=467869  perm=NO-KEYS-LOST
  N=1000000  seed=777 [32,64) rocPRIM: subkey-inv=438021  perm=NO-KEYS-LOST
  N=10000    seed=5   [40,64) hipCUB : subkey-inv=3902    perm=NO-KEYS-LOST
```
Every range with `end_bit == 64` and `begin_bit > 0` loses keys (output is not a permutation of the input); rocPRIM directly shows the same.

## Expected
A partial-range radix sort must order keys by the selected `[begin_bit,end_bit)` bit range and must always output a permutation of the input. NVIDIA CUB does for these ranges.

## Trigger
`begin_bit > 0` AND `end_bit == 8*sizeof(KeyT)`. The defect is in rocPRIM; hipCUB inherits it. Plausibly a digit-extraction / pass-count boundary error when the top bit is included with a nonzero start bit.

## How this was found
While porting cudaKDTree to ROCm/HIP as part of MOAT (https://github.com/jeffdaily/moat). Its spatial k-d tree sorts a 64-bit `(nodeID<<32)|primID` key by the high 32 bits via `[32,64)`, which lost/duplicated leaf primitive indices and corrupted nearest-neighbor queries. Workaround in the port: sort the full key (`[0,64)`).
