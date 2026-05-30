#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <rocprim/rocprim.hpp>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#define HIP_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));std::exit(2);} }while(0)
int main(){
  const size_t N=10000; std::mt19937_64 rng(12345);
  std::vector<uint64_t> in(N); for(size_t i=0;i<N;++i){ uint64_t r=rng(); in[i]=(uint64_t)r; }
  for(int which=0; which<2; ++which){
    uint64_t *d_in=nullptr,*d_out=nullptr;
    HIP_CHECK(hipMalloc(&d_in,N*8)); HIP_CHECK(hipMalloc(&d_out,N*8));
    HIP_CHECK(hipMemcpy(d_in,in.data(),N*8,hipMemcpyHostToDevice));
    void* d_temp=nullptr; size_t tb=0;
    if(which==1){
      HIP_CHECK(rocprim::radix_sort_keys(d_temp,tb,d_in,d_out,N,0,64,0));
      HIP_CHECK(hipMalloc(&d_temp,tb));
      HIP_CHECK(rocprim::radix_sort_keys(d_temp,tb,d_in,d_out,N,0,64,0));
    } else {
      HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(d_temp,tb,d_in,d_out,(int)N,0,64,0));
      HIP_CHECK(hipMalloc(&d_temp,tb));
      HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(d_temp,tb,d_in,d_out,(int)N,0,64,0));
    }
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> out(N); HIP_CHECK(hipMemcpy(out.data(),d_out,N*8,hipMemcpyDeviceToHost));
    size_t inv=0; for(size_t i=1;i<N;++i) if(out[i-1]>out[i]) ++inv;
    printf("%s [0,64): full-key inversions=%zu  (out[0]=%016llx out[1]=%016llx out[N-1]=%016llx)\n",
      which?"rocPRIM":"hipCUB", inv,(unsigned long long)out[0],(unsigned long long)out[1],(unsigned long long)out[N-1]);
    HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_out)); HIP_CHECK(hipFree(d_temp));
  }
  return 0;
}
