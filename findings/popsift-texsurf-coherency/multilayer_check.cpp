// Does surf2DLayeredwrite to MULTIPLE distinct layers, read back in a SEPARATE
// kernel via surf2DLayeredread, return the per-layer values on gfx90a? popsift's
// make_dog reads every _data layer (written by aa::vert in a prior launch) and
// observed all layers == layer0's value. Reproduce that pattern in isolation.
#include <cstring>
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return -1;}}while(0)

static const int W=32,H=32,L=6;

// kernel A: write value (layer*100 + 7) into each layer, ONE layer per launch
__global__ void writeOneLayer(hipSurfaceObject_t surf, int layer){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H) return;
  surf2DLayeredwrite( (float)(layer*100+7), surf, x*4, y, layer);
}

// kernel B (separate launch): read all L layers at (cx,cy) via surface
__global__ void readAll(hipSurfaceObject_t surf, float* out, int cx, int cy){
  for(int l=0;l<L;l++){ float v; surf2DLayeredread(&v, surf, cx*4, cy, l); out[l]=v; }
}

int main(){
  hipChannelFormatDesc d=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindFloat);
  hipArray_t arr; hipExtent ext=make_hipExtent(W,H,L);
  CHECK(hipMalloc3DArray(&arr,&d,ext,hipArrayLayered|hipArraySurfaceLoadStore));
  hipResourceDesc rd; memset(&rd,0,sizeof(rd)); rd.resType=hipResourceTypeArray; rd.res.array.array=arr;
  hipSurfaceObject_t surf; CHECK(hipCreateSurfaceObject(&surf,&rd));
  dim3 b(8,8), g((W+7)/8,(H+7)/8);

  // write each layer in its OWN launch (mirrors popsift: each level = separate vert launch)
  for(int l=0;l<L;l++) hipLaunchKernelGGL(writeOneLayer,g,b,0,0,surf,l);

  float* dout; CHECK(hipMalloc(&dout,L*sizeof(float)));
  hipLaunchKernelGGL(readAll,dim3(1),dim3(1),0,0,surf,dout,16,16); // separate launch
  CHECK(hipDeviceSynchronize());
  float h[L]; CHECK(hipMemcpy(h,dout,L*sizeof(float),hipMemcpyDeviceToHost));
  printf("read-all via surface (separate launch):\n");
  int ok=1;
  for(int l=0;l<L;l++){ int exp=l*100+7; printf("  layer %d: got %.1f exp %d %s\n",l,h[l],exp, (int)h[l]==exp?"OK":"STALE"); if((int)h[l]!=exp) ok=0; }
  printf("%s\n", ok?"ALL FRESH":"SOME STALE -> multilayer surface read across launches is broken");
  return 0;
}
