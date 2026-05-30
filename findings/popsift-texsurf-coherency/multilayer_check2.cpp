// Variant A: write ALL layers in ONE launch, read in a separate launch.
// Variant B: add hipDeviceSynchronize between each per-layer write launch.
// Variant C: recreate the surface object before the read.
// Determine what (if anything) makes a layered surface read see all layers.
#include <cstring>
#include <hip/hip_runtime.h>
#include <cstdio>
#define CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return -1;}}while(0)
static const int W=32,H=32,L=6;
__global__ void writeAll(hipSurfaceObject_t surf){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H) return;
  for(int l=0;l<L;l++) surf2DLayeredwrite((float)(l*100+7), surf, x*4, y, l);
}
__global__ void writeOneLayer(hipSurfaceObject_t surf, int layer){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H) return;
  surf2DLayeredwrite((float)(layer*100+7), surf, x*4, y, layer);
}
__global__ void readAll(hipSurfaceObject_t surf, float* out, int cx, int cy){
  for(int l=0;l<L;l++){ float v; surf2DLayeredread(&v, surf, cx*4, cy, l); out[l]=v; }
}
static int run(const char* tag, hipArray_t arr, hipSurfaceObject_t surf, float* dout){
  float h[L]; CHECK(hipMemcpy(h,dout,L*sizeof(float),hipMemcpyDeviceToHost));
  int ok=1; printf("%s:\n", tag);
  for(int l=0;l<L;l++){ int e=l*100+7; printf("  L%d got %.1f exp %d %s\n",l,h[l],e,(int)h[l]==e?"OK":"STALE"); if((int)h[l]!=e)ok=0; }
  printf("  => %s\n", ok?"ALL FRESH":"BROKEN"); return ok;
}
int main(){
  hipChannelFormatDesc d=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindFloat);
  hipExtent ext=make_hipExtent(W,H,L);
  dim3 b(8,8), g((W+7)/8,(H+7)/8);
  float* dout; CHECK(hipMalloc(&dout,L*sizeof(float)));

  // Variant A: single-launch write-all
  {
    hipArray_t arr; CHECK(hipMalloc3DArray(&arr,&d,ext,hipArrayLayered|hipArraySurfaceLoadStore));
    hipResourceDesc rd; memset(&rd,0,sizeof(rd)); rd.resType=hipResourceTypeArray; rd.res.array.array=arr;
    hipSurfaceObject_t s; CHECK(hipCreateSurfaceObject(&s,&rd));
    hipLaunchKernelGGL(writeAll,g,b,0,0,s);
    hipLaunchKernelGGL(readAll,dim3(1),dim3(1),0,0,s,dout,16,16);
    CHECK(hipDeviceSynchronize());
    run("[A] single-launch write-all, separate read", arr, s, dout);
    hipDestroySurfaceObject(s); hipFreeArray(arr);
  }
  // Variant B: per-layer writes each in own launch + deviceSync between, then read
  {
    hipArray_t arr; CHECK(hipMalloc3DArray(&arr,&d,ext,hipArrayLayered|hipArraySurfaceLoadStore));
    hipResourceDesc rd; memset(&rd,0,sizeof(rd)); rd.resType=hipResourceTypeArray; rd.res.array.array=arr;
    hipSurfaceObject_t s; CHECK(hipCreateSurfaceObject(&s,&rd));
    for(int l=0;l<L;l++){ hipLaunchKernelGGL(writeOneLayer,g,b,0,0,s,l); CHECK(hipDeviceSynchronize()); }
    hipLaunchKernelGGL(readAll,dim3(1),dim3(1),0,0,s,dout,16,16);
    CHECK(hipDeviceSynchronize());
    run("[B] per-layer writes + sync, separate read", arr, s, dout);
    hipDestroySurfaceObject(s); hipFreeArray(arr);
  }
  // Variant C: per-layer writes, recreate surface object before read
  {
    hipArray_t arr; CHECK(hipMalloc3DArray(&arr,&d,ext,hipArrayLayered|hipArraySurfaceLoadStore));
    hipResourceDesc rd; memset(&rd,0,sizeof(rd)); rd.resType=hipResourceTypeArray; rd.res.array.array=arr;
    hipSurfaceObject_t s; CHECK(hipCreateSurfaceObject(&s,&rd));
    for(int l=0;l<L;l++) hipLaunchKernelGGL(writeOneLayer,g,b,0,0,s,l);
    CHECK(hipDeviceSynchronize());
    hipSurfaceObject_t s2; CHECK(hipCreateSurfaceObject(&s2,&rd)); // fresh surface obj
    hipLaunchKernelGGL(readAll,dim3(1),dim3(1),0,0,s2,dout,16,16);
    CHECK(hipDeviceSynchronize());
    run("[C] per-layer writes, fresh surface obj for read", arr, s2, dout);
    hipDestroySurfaceObject(s); hipDestroySurfaceObject(s2); hipFreeArray(arr);
  }
  return 0;
}
