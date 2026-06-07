// Smoketest that the PUBLIC umbrella headers this change ungates now compile on
// HIP/AMD (the change removed the __HIP_PLATFORM_AMD__ #error from these two
// headers). Pre-change this TU fails to compile with that #error; post-change it
// compiles and the device kernel instantiates both semaphore flavors.
//   hipcc -std=c++17 --offload-arch=<arch> -I <libhipcxx>/include sem_umbrella_smoke.cpp -o sem_umbrella_smoke
//
// NOTE: HIP does not allow in-place construction of __shared__ variables with
// non-trivial constructors. Use aligned char storage + placement new instead.
#include <cstdio>
#include <hip/hip_runtime.h>
#include <cuda/semaphore>
#include <cuda/std/semaphore>

using BSem = cuda::binary_semaphore<cuda::thread_scope_device>;
using CSem = cuda::counting_semaphore<cuda::thread_scope_device, 4>;

__global__ void touch() {
  __shared__ __attribute__((aligned(alignof(BSem)))) char bsem_buf[sizeof(BSem)];
  __shared__ __attribute__((aligned(alignof(CSem)))) char csem_buf[sizeof(CSem)];
  if (threadIdx.x == 0) {
    BSem* bsem = new (bsem_buf) BSem(1);
    CSem* csem = new (csem_buf) CSem(2);
    bsem->acquire(); bsem->release();
    csem->acquire(); csem->release();
  }
}

int main() {
  touch<<<dim3(1), dim3(1)>>>();
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("FAIL: %s\n", hipGetErrorString(e)); return 1; }
  printf("PASS: <cuda/semaphore> + <cuda/std/semaphore> compile and run on HIP/AMD\n");
  return 0;
}
