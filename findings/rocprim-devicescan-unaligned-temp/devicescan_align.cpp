// Minimal reproducer: hipCUB/rocPRIM DeviceScan corrupts when d_temp_storage
// is not 256-byte aligned. CUB's contract allows any d_temp_storage of at least
// temp_storage_bytes; this passes an unaligned base (as a caller sub-allocating
// from a larger buffer would) and checks ExclusiveSum of all-ones (out[i]==i).
//
// Build: hipcc -O2 --offload-arch=gfx90a devicescan_align.cpp -o devicescan_align
// Run:   ./devicescan_align
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return -2;}}while(0)

static int run(size_t off){
  const int N = 1<<20;
  std::vector<int> hin(N,1), hout(N,-1);
  int *din=nullptr,*dout=nullptr; char* base=nullptr;
  CHECK(hipMalloc(&din,  N*sizeof(int)));
  CHECK(hipMalloc(&dout, N*sizeof(int)));
  CHECK(hipMemcpy(din, hin.data(), N*sizeof(int), hipMemcpyHostToDevice));
  size_t tmp=0;
  CHECK(hipcub::DeviceScan::ExclusiveSum((void*)nullptr, tmp, din, dout, N));
  CHECK(hipMalloc(&base, tmp + 256));
  void* dtmp = base + off;
  CHECK(hipMemset(dout, 0, N*sizeof(int)));
  CHECK(hipcub::DeviceScan::ExclusiveSum(dtmp, tmp, din, dout, N));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(hout.data(), dout, N*sizeof(int), hipMemcpyDeviceToHost));
  int bad=0, firstbad=-1;
  for(int i=0;i<N;i++){ if(hout[i]!=i){ bad++; if(firstbad<0) firstbad=i; } }
  printf("off=%3zu base%%256=%3zu dtmp%%256=%3zu tmpBytes=%zu -> bad=%d firstbad=%d\n",
         off,(uintptr_t)base%256,(uintptr_t)dtmp%256,tmp,bad,firstbad);
  hipFree(din);hipFree(dout);hipFree(base);
  return bad;
}
int main(){
  printf("hipCUB DeviceScan::ExclusiveSum of all-ones, N=2^20; correct => out[i]==i\n");
  size_t offs[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,64,68,132,256,260};
  for(size_t o : offs){ int b=run(o); if(b) run(o); }  // re-run failures to show non-determinism
  return 0;
}
