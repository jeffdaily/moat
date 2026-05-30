#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#define CK(c) do{hipError_t e=(c); if(e!=hipSuccess){fprintf(stderr,"HIP %d %s\n",__LINE__,hipGetErrorString(e));std::exit(2);}}while(0)
static void run(int bb,int eb){
  const size_t N=100000; std::mt19937_64 rng(12345);
  std::vector<uint64_t> in(N); for(auto&x:in)x=rng();
  uint64_t *di,*do_; CK(hipMalloc(&di,N*8)); CK(hipMalloc(&do_,N*8));
  CK(hipMemcpy(di,in.data(),N*8,hipMemcpyHostToDevice));
  void* dt=nullptr; size_t tb=0;
  CK(hipcub::DeviceRadixSort::SortKeys(dt,tb,di,do_,(int)N,bb,eb));
  CK(hipMalloc(&dt,tb));
  CK(hipcub::DeviceRadixSort::SortKeys(dt,tb,di,do_,(int)N,bb,eb));
  CK(hipDeviceSynchronize());
  std::vector<uint64_t> out(N); CK(hipMemcpy(out.data(),do_,N*8,hipMemcpyDeviceToHost));
  int w=eb-bb; uint64_t m=(w>=64)?~0ull:((1ull<<w)-1);
  size_t inv=0; for(size_t i=1;i<N;++i){ uint64_t a=(out[i-1]>>bb)&m, b=(out[i]>>bb)&m; if(a>b)++inv; }
  std::vector<uint64_t> si=in, so=out; std::sort(si.begin(),si.end()); std::sort(so.begin(),so.end());
  bool perm=(si==so);
  printf("[%2d,%2d): subkey-inversions=%6zu/%zu  perm=%s\n",bb,eb,inv,N-1,perm?"yes":"NO(KEYS LOST)");
  CK(hipFree(di)); CK(hipFree(do_)); CK(hipFree(dt));
}
int main(){ run(0,64); run(32,64); run(8,24); run(40,64); return 0; }
