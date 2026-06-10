// Verify: does rocPRIM size SortKeys < SortPairs (so a SortKeys-sized temp is
// too small for a SortPairs run), and does running SortPairs with a SortKeys-
// sized temp fault? If SortPairs genuinely needs more temp, sizing via SortKeys
// then running SortPairs is a caller error (under-allocation), not a rocPRIM bug
// -- CUB would also require sizing for the path actually run; it just happens to
// size them equally so the mistake is harmless there.
//
// Build: hipcc -O2 --offload-arch=gfx90a sortpairs_tempsize.cpp -o sortpairs_tempsize
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdio>
#include <vector>

#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return -2;}}while(0)

int main(){
  const int N=1<<16, S=4;
  std::vector<int> beg(S),end(S);
  for(int s=0;s<S;s++){ beg[s]=s*(N/S); end[s]=(s+1)*(N/S); }
  int *k0,*k1,*v0,*v1,*db,*de;
  CHECK(hipMalloc(&k0,N*sizeof(int))); CHECK(hipMalloc(&k1,N*sizeof(int)));
  CHECK(hipMalloc(&v0,N*sizeof(int))); CHECK(hipMalloc(&v1,N*sizeof(int)));
  CHECK(hipMalloc(&db,S*sizeof(int))); CHECK(hipMalloc(&de,S*sizeof(int)));
  CHECK(hipMemcpy(db,beg.data(),S*sizeof(int),hipMemcpyHostToDevice));
  CHECK(hipMemcpy(de,end.data(),S*sizeof(int),hipMemcpyHostToDevice));

  hipcub::DoubleBuffer<int> keys(k0,k1), vals(v0,v1);
  size_t tmpK=0, tmpP=0;
  CHECK(hipcub::DeviceSegmentedRadixSort::SortKeys (nullptr,tmpK,keys,N,S,db,de));
  CHECK(hipcub::DeviceSegmentedRadixSort::SortPairs(nullptr,tmpP,keys,vals,N,S,db,de));
  printf("SortKeys temp_storage_bytes  = %zu\n", tmpK);
  printf("SortPairs temp_storage_bytes = %zu\n", tmpP);
  printf("=> SortPairs needs %s than SortKeys\n", tmpP>tmpK?"MORE":(tmpP<tmpK?"LESS":"the SAME"));

  // Now run SortPairs with only the SortKeys-sized temp (the catboost mistake):
  void* dtmp; CHECK(hipMalloc(&dtmp, tmpK?tmpK:1));
  hipError_t e = hipcub::DeviceSegmentedRadixSort::SortPairs(dtmp,tmpK,keys,vals,N,S,db,de);
  hipError_t es = hipDeviceSynchronize();
  printf("SortPairs run with SortKeys-sized temp: launch=%s sync=%s\n",
         hipGetErrorString(e), hipGetErrorString(es));
  return 0;
}
