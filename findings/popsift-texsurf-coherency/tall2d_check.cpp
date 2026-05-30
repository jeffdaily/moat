// Validate the "tall 2D array" replacement for a layered array: one non-layered
// 2D array of size W x (H*L); layer k lives at rows [k*H, (k+1)*H). Write each
// layer-band in its own launch, read every band back in a SEPARATE launch (both
// surface and texture). This is the non-layered path proven coherent earlier, so
// it should return per-layer-fresh data -- the fix for popsift's broken layered
// pyramid arrays on gfx90a/ROCm 7.2.1.
#include <cstring>
#include <hip/hip_runtime.h>
#include <cstdio>
#define CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return -1;}}while(0)
static const int W=32,H=32,L=6;
__global__ void writeBand(hipSurfaceObject_t surf, int layer){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H) return;
  surf2Dwrite((float)(layer*100+7), surf, x*4, y + layer*H);
}
__global__ void readBandsSurf(hipSurfaceObject_t surf, float* out, int cx, int cy){
  for(int l=0;l<L;l++){ float v; surf2Dread(&v, surf, cx*4, cy + l*H); out[l]=v; }
}
__global__ void readBandsTex(hipTextureObject_t tex, float* out, int cx, int cy){
  for(int l=0;l<L;l++){ out[l]=tex2D<float>(tex, cx+0.5f, cy + l*H + 0.5f); }
}
int main(){
  hipChannelFormatDesc d=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindFloat);
  hipArray_t arr; CHECK(hipMallocArray(&arr,&d,W,H*L,hipArraySurfaceLoadStore));
  hipResourceDesc rd; memset(&rd,0,sizeof(rd)); rd.resType=hipResourceTypeArray; rd.res.array.array=arr;
  hipSurfaceObject_t s; CHECK(hipCreateSurfaceObject(&s,&rd));
  hipTextureDesc td; memset(&td,0,sizeof(td)); td.normalizedCoords=0;
  td.addressMode[0]=hipAddressModeClamp; td.addressMode[1]=hipAddressModeClamp;
  td.readMode=hipReadModeElementType; td.filterMode=hipFilterModePoint;
  hipTextureObject_t tex; CHECK(hipCreateTextureObject(&tex,&rd,&td,nullptr));
  dim3 b(8,8), g((W+7)/8,(H+7)/8);

  for(int l=0;l<L;l++) hipLaunchKernelGGL(writeBand,g,b,0,0,s,l);
  CHECK(hipDeviceSynchronize());

  float* dout; CHECK(hipMalloc(&dout,L*sizeof(float)));
  hipLaunchKernelGGL(readBandsSurf,dim3(1),dim3(1),0,0,s,dout,16,16);
  CHECK(hipDeviceSynchronize());
  float h[L]; CHECK(hipMemcpy(h,dout,L*sizeof(float),hipMemcpyDeviceToHost));
  int oks=1; printf("tall-2D SURFACE read per band (separate launch):\n");
  for(int l=0;l<L;l++){ int e=l*100+7; printf("  L%d got %.1f exp %d %s\n",l,h[l],e,(int)h[l]==e?"OK":"STALE"); if((int)h[l]!=e)oks=0; }
  printf("  => surface: %s\n", oks?"ALL FRESH":"BROKEN");

  hipLaunchKernelGGL(readBandsTex,dim3(1),dim3(1),0,0,tex,dout,16,16);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h,dout,L*sizeof(float),hipMemcpyDeviceToHost));
  int okt=1; printf("tall-2D TEXTURE read per band (separate launch):\n");
  for(int l=0;l<L;l++){ int e=l*100+7; printf("  L%d got %.1f exp %d %s\n",l,h[l],e,(int)h[l]==e?"OK":"STALE"); if((int)h[l]!=e)okt=0; }
  printf("  => texture: %s\n", okt?"ALL FRESH":"BROKEN");
  return 0;
}
