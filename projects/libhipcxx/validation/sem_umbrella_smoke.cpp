// Smoketest that the PUBLIC umbrella headers this change ungates now compile on
// HIP/AMD (the change removed the __HIP_PLATFORM_AMD__ #error from these two
// headers). Pre-change this TU fails to compile with that #error; post-change it
// compiles and the device kernel instantiates both semaphore flavors.
//   hipcc -std=c++17 --offload-arch=<arch> -I <libhipcxx>/include sem_umbrella_smoke.cpp -o sem_umbrella_smoke
#include <cstdio>
#include <hip/hip_runtime.h>
#include <cuda/semaphore>
#include <cuda/std/semaphore>

__global__ void touch() {
  __shared__ cuda::binary_semaphore<cuda::thread_scope_device> bsem;
  __shared__ cuda::counting_semaphore<cuda::thread_scope_device, 4> csem;
  if (threadIdx.x == 0) {
    new (&bsem) cuda::binary_semaphore<cuda::thread_scope_device>(1);
    new (&csem) cuda::counting_semaphore<cuda::thread_scope_device, 4>(2);
    bsem.acquire(); bsem.release();
    csem.acquire(); csem.release();
  }
}

int main() {
  touch<<<dim3(1), dim3(1)>>>();
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("FAIL: %s\n", hipGetErrorString(e)); return 1; }
  printf("PASS: <cuda/semaphore> + <cuda/std/semaphore> compile and run on HIP/AMD\n");
  return 0;
}
