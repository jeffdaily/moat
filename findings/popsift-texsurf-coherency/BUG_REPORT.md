# Layered cudaArray collapses to one layer: surf2DLayeredread / tex2DLayered return the last-written layer for every layer index (gfx90a)

## Component
HIP runtime / clr -- surface/texture over a layered `hipMalloc3DArray`.

## Summary
On gfx90a (CDNA2, ROCm 7.2.1), a **layered** cudaArray (`hipMalloc3DArray` with `hipArrayLayered | hipArraySurfaceLoadStore`) that is written one layer at a time via `surf2DLayeredwrite` in separate kernel launches returns the **last-written layer's data for every layer index** when read back in a later launch -- via `surf2DLayeredread`, `tex2DLayered`, or even host `hipMemcpy3D`. The per-layer contents are lost; the array behaves as if it holds a single layer. A non-layered 3D array of the same dimensions (no `hipArrayLayered`, accessed with `surf3Dwrite`/`surf3Dread`/`tex3D` and the layer as the z coordinate) is correct. CUDA preserves the per-layer contents.

## Environment
- AMD Instinct MI250X, gfx90a (CDNA2)
- ROCm 7.2.1

## Reproducer (`multilayer_check.cpp`, build: `hipcc -O2 multilayer_check.cpp -o mlc`)
```cpp
#include <cstring>
#include <hip/hip_runtime.h>
#include <cstdio>
#define CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return -1;}}while(0)
static const int W=32,H=32,L=6;
__global__ void writeOneLayer(hipSurfaceObject_t surf,int layer){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H) return;
  surf2DLayeredwrite((float)(layer*100+7), surf, x*4, y, layer);
}
__global__ void readAll(hipSurfaceObject_t surf,float* out,int cx,int cy){
  for(int l=0;l<L;l++){ float v; surf2DLayeredread(&v,surf,cx*4,cy,l); out[l]=v; }
}
int main(){
  hipChannelFormatDesc d=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindFloat);
  hipArray_t arr; hipExtent ext=make_hipExtent(W,H,L);
  CHECK(hipMalloc3DArray(&arr,&d,ext,hipArrayLayered|hipArraySurfaceLoadStore));
  hipResourceDesc rd; memset(&rd,0,sizeof(rd)); rd.resType=hipResourceTypeArray; rd.res.array.array=arr;
  hipSurfaceObject_t surf; CHECK(hipCreateSurfaceObject(&surf,&rd));
  dim3 b(8,8), g((W+7)/8,(H+7)/8);
  for(int l=0;l<L;l++) hipLaunchKernelGGL(writeOneLayer,g,b,0,0,surf,l);  // one layer per launch
  float* dout; CHECK(hipMalloc(&dout,L*sizeof(float)));
  hipLaunchKernelGGL(readAll,dim3(1),dim3(1),0,0,surf,dout,16,16);        // separate launch
  CHECK(hipDeviceSynchronize());
  float h[L]; CHECK(hipMemcpy(h,dout,L*sizeof(float),hipMemcpyDeviceToHost));
  for(int l=0;l<L;l++){ int exp=l*100+7; printf("  layer %d: got %.1f exp %d %s\n",l,h[l],exp,(int)h[l]==exp?"OK":"STALE"); }
  return 0;
}
```

## Observed (each layer L is written `L*100+7`, then all layers read at one (x,y) in a separate launch)
```
  layer 0: got 507.0 exp 7   STALE
  layer 1: got 507.0 exp 107 STALE
  layer 2: got 507.0 exp 207 STALE
  layer 3: got 507.0 exp 307 STALE
  layer 4: got 507.0 exp 407 STALE
  layer 5: got 507.0 exp 507 OK     <- 507 is the value written to the LAST layer
```
Every layer returns 507.0 = the last-written layer's value. `tex2DLayered` and a host `hipMemcpy3D` of the array read back the same collapsed result.

## Expected
Each layer returns its own value (7, 107, 207, 307, 407, 507), as on CUDA. A **non-layered 3D array** with the same data (drop `hipArrayLayered`; write/read via `surf3Dwrite`/`surf3Dread`/`tex3D` with the layer as z) is correct on ROCm -- both surface and texture reads return the right per-slice value -- so the data path itself is fine; only the *layered* addressing collapses:
```
  z0..z5 via surf3D: 7,107,207,307,407,507  => ALL FRESH
  z0..z5 via tex3D : 7,107,207,307,407,507  => ALL FRESH
```

## Notes
- All layers are written before any read, with `hipDeviceSynchronize` between the writes and the read; recreating the surface/texture object after the writes does not help. The single discriminator is the `hipArrayLayered` flag.
- Impact: surfaced porting popsift (GPU SIFT). Its Gaussian-pyramid and DoG octaves are layered arrays written per-level via surface and read per-level via texture; the layer collapse made the DoG all-zero, yielding 0 detected features. Workaround: make the octave arrays non-layered 3D (`surf3D`/`tex3D`, level as z).
- Repros attached: `multilayer_check.cpp` (the bug), `array3d_check.cpp` (the non-layered 3D control, passes).
