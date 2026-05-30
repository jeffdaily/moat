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
  printf("uint64, hipCUB 4.2.0 / rocPRIM 4.2.0, gfx90a, ROCm 7.2.1 -- subkey-inv MUST be 0 and perm MUST be yes\n");
  for(auto&r:ranges) run(1000000,12345,r[0],r[1],false);
  printf("-- a couple cross-checks (rocPRIM + 2nd seed) --\n");
  run(1000000,777,32,64,false); run(1000000,777,32,64,true); run(10000,5,40,64,false);
  return 0;
}
