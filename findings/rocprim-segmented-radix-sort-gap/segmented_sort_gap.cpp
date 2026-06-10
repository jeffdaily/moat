// Verify: does rocPRIM DeviceSegmentedRadixSort with a DoubleBuffer leave
// out-of-segment ("gap") elements undefined in Current() after the sort?
// Segments cover [0,4) and [8,12); gaps are [4,8) and [12,16).
// buf0 = input (distinct descending), buf1 = sentinel 999.
// If Current()==buf1 and gaps read 999 -> gaps are undefined (the catboost symptom).
// If gaps read the input values -> preserved (CUB-like).
//
// Build: hipcc -O2 --offload-arch=gfx90a segmented_sort_gap.cpp -o segmented_sort_gap
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdio>
#include <vector>

#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return -2;}}while(0)

int main(){
  const int N=16, S=2, SENT=999;
  std::vector<int> h_begin={0,8}, h_end={4,12}, h_in(N);
  for(int i=0;i<N;i++) h_in[i]=N-i;            // 16,15,...,1
  int *buf0,*buf1,*dbeg,*dend;
  CHECK(hipMalloc(&buf0,N*sizeof(int))); CHECK(hipMalloc(&buf1,N*sizeof(int)));
  CHECK(hipMalloc(&dbeg,S*sizeof(int))); CHECK(hipMalloc(&dend,S*sizeof(int)));
  CHECK(hipMemcpy(buf0,h_in.data(),N*sizeof(int),hipMemcpyHostToDevice));
  std::vector<int> sent(N,SENT);
  CHECK(hipMemcpy(buf1,sent.data(),N*sizeof(int),hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dbeg,h_begin.data(),S*sizeof(int),hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dend,h_end.data(),S*sizeof(int),hipMemcpyHostToDevice));

  void* dtmp=nullptr; size_t cap=0;
  for(int end_bit : {4,8,12,16,20,24,28,32}){
    CHECK(hipMemcpy(buf0,h_in.data(),N*sizeof(int),hipMemcpyHostToDevice));   // reset each run
    CHECK(hipMemcpy(buf1,sent.data(),N*sizeof(int),hipMemcpyHostToDevice));
    hipcub::DoubleBuffer<int> keys(buf0,buf1);
    size_t tmp=0;
    CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(nullptr,tmp,keys,N,S,dbeg,dend,0,end_bit));
    if(tmp>cap){ if(dtmp) hipFree(dtmp); CHECK(hipMalloc(&dtmp,tmp?tmp:1)); cap=tmp; }
    CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys(dtmp,tmp,keys,N,S,dbeg,dend,0,end_bit));
    CHECK(hipDeviceSynchronize());
    std::vector<int> out(N);
    CHECK(hipMemcpy(out.data(),keys.Current(),N*sizeof(int),hipMemcpyDeviceToHost));
    bool gapBad = (out[4]==SENT||out[5]==SENT||out[12]==SENT||out[13]==SENT);
    printf("end_bit=%2d Current=%s  gaps[4..8)=%d %d %d %d  [12..16)=%d %d %d %d  %s\n",
           end_bit, keys.Current()==buf0?"buf0":"buf1",
           out[4],out[5],out[6],out[7],out[12],out[13],out[14],out[15],
           gapBad?"<-- GAP UNDEFINED (sentinel leaked)":"");
  }
  return 0;
}
