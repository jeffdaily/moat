// Does a NON-layered 3D array (hipMalloc3DArray without hipArrayLayered) with
// surf3Dwrite / tex3D give per-slice-coherent reads across launches? If so it is
// the minimal-change fix (z stays a real coordinate; only Layered->3D + flag).
#include <cstring>
#include <hip/hip_runtime.h>
#include <cstdio>
#define CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return -1;}}while(0)
static const int W=32,H=32,D=6;
__global__ void writeSlice(hipSurfaceObject_t surf, int z){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H) return;
  surf3Dwrite((float)(z*100+7), surf, x*4, y, z);
}
__global__ void readSlicesSurf(hipSurfaceObject_t surf, float* out, int cx, int cy){
  for(int z=0;z<D;z++){ float v; surf3Dread(&v, surf, cx*4, cy, z); out[z]=v; }
}
__global__ void readSlicesTex(hipTextureObject_t tex, float* out, int cx, int cy){
  for(int z=0;z<D;z++){ out[z]=tex3D<float>(tex, cx+0.5f, cy+0.5f, z+0.5f); }
}
int main(){
  hipChannelFormatDesc d=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindFloat);
  hipExtent ext=make_hipExtent(W,H,D);
  hipArray_t arr; CHECK(hipMalloc3DArray(&arr,&d,ext,hipArraySurfaceLoadStore)); // NO Layered
  hipResourceDesc rd; memset(&rd,0,sizeof(rd)); rd.resType=hipResourceTypeArray; rd.res.array.array=arr;
  hipSurfaceObject_t s; CHECK(hipCreateSurfaceObject(&s,&rd));
  hipTextureDesc td; memset(&td,0,sizeof(td)); td.normalizedCoords=0;
  td.addressMode[0]=hipAddressModeClamp; td.addressMode[1]=hipAddressModeClamp; td.addressMode[2]=hipAddressModeClamp;
  td.readMode=hipReadModeElementType; td.filterMode=hipFilterModePoint;
  hipTextureObject_t tex; CHECK(hipCreateTextureObject(&tex,&rd,&td,nullptr));
  dim3 b(8,8), g((W+7)/8,(H+7)/8);
  for(int z=0;z<D;z++) hipLaunchKernelGGL(writeSlice,g,b,0,0,s,z);
  CHECK(hipDeviceSynchronize());
  float* dout; CHECK(hipMalloc(&dout,D*sizeof(float)));
  hipLaunchKernelGGL(readSlicesSurf,dim3(1),dim3(1),0,0,s,dout,16,16);
  CHECK(hipDeviceSynchronize());
  float h[D]; CHECK(hipMemcpy(h,dout,D*sizeof(float),hipMemcpyDeviceToHost));
  int oks=1; printf("3D-array SURFACE read per slice:\n");
  for(int z=0;z<D;z++){ int e=z*100+7; printf("  z%d got %.1f exp %d %s\n",z,h[z],e,(int)h[z]==e?"OK":"STALE"); if((int)h[z]!=e)oks=0; }
  printf("  => surf3D: %s\n", oks?"ALL FRESH":"BROKEN");
  hipLaunchKernelGGL(readSlicesTex,dim3(1),dim3(1),0,0,tex,dout,16,16);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h,dout,D*sizeof(float),hipMemcpyDeviceToHost));
  int okt=1; printf("3D-array TEXTURE read per slice:\n");
  for(int z=0;z<D;z++){ int e=z*100+7; printf("  z%d got %.1f exp %d %s\n",z,h[z],e,(int)h[z]==e?"OK":"STALE"); if((int)h[z]!=e)okt=0; }
  printf("  => tex3D: %s\n", okt?"ALL FRESH":"BROKEN");
  return 0;
}
