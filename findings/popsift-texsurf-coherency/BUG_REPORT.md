# Layered cudaArray collapses to one layer: surf2DLayeredread / tex2DLayered return the last-written layer for every layer index (gfx90a)

## Component
HIP runtime / clr -- surface/texture over a layered `hipMalloc3DArray`.

## Summary
On gfx90a (CDNA2, ROCm 7.2.1), a **layered** cudaArray (`hipMalloc3DArray` with `hipArrayLayered | hipArraySurfaceLoadStore`) written one layer at a time via `surf2DLayeredwrite` in separate kernel launches returns the **last-written layer's data for every layer index** when read back in a later launch -- via `surf2DLayeredread`, `tex2DLayered`, or host `hipMemcpy3D`. Per-layer contents are lost; the array behaves as if it holds one layer. A non-layered 3D array of the same dimensions (no `hipArrayLayered`; `surf3D`/`tex3D` with the layer as the z coordinate) is correct. CUDA preserves the per-layer contents.

## Environment
- AMD Instinct MI250X, gfx90a (CDNA2)
- ROCm 7.2.1

## Reproducer (the bug)
Build: `hipcc -O2 multilayer_check.cpp -o mlc`  Run: `./mlc`

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
Each layer returns its own value (7, 107, 207, 307, 407, 507), as on CUDA.

## Control: the same data in a NON-layered 3D array is correct on ROCm
Dropping `hipArrayLayered` and using `surf3Dwrite`/`surf3Dread`/`tex3D` with the layer as the z coordinate returns the right per-slice value through both surface and texture -- so the data path itself is fine; only the *layered* addressing collapses. Build: `hipcc -O2 array3d_check.cpp -o a3c`  Run: `./a3c`

```cpp
#include <cstring>
#include <hip/hip_runtime.h>
#include <cstdio>
#define CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ \
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
  printf("3D-array SURFACE read per slice:\n");
  for(int z=0;z<D;z++){ int e=z*100+7; printf("  z%d got %.1f exp %d %s\n",z,h[z],e,(int)h[z]==e?"OK":"STALE"); }
  hipLaunchKernelGGL(readSlicesTex,dim3(1),dim3(1),0,0,tex,dout,16,16);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h,dout,D*sizeof(float),hipMemcpyDeviceToHost));
  printf("3D-array TEXTURE read per slice:\n");
  for(int z=0;z<D;z++){ int e=z*100+7; printf("  z%d got %.1f exp %d %s\n",z,h[z],e,(int)h[z]==e?"OK":"STALE"); }
  return 0;
}
```
Output: every slice is correct via both paths --
```
3D-array SURFACE read per slice:  z0..z5 -> 7,107,207,307,407,507  (all OK)
3D-array TEXTURE read per slice:  z0..z5 -> 7,107,207,307,407,507  (all OK)
```

## Notes
- All layers are written before any read, with `hipDeviceSynchronize` between the writes and the read; recreating the surface/texture object after the writes does not help. The single discriminator is the `hipArrayLayered` flag.

## How this was found
While porting popsift (a CUDA GPU-SIFT library) to ROCm/HIP as part of MOAT (https://github.com/jeffdaily/moat). Its Gaussian-pyramid and DoG octaves are layered arrays written per-level via surface and read per-level via texture; the layer collapse made the DoG all-zero, yielding 0 detected features. Workaround in the port: make the octave arrays non-layered 3D (`surf3D`/`tex3D`, level as z).
