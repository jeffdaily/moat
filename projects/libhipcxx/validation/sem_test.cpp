// De-risk: does the present-but-#error-gated cuda::std semaphore machinery in
// ROCm/libhipcxx actually work on gfx90a? Bypass the gated public umbrella by
// including the internal extended header directly, and exercise oneflow's exact
// type: cuda::binary_semaphore<cuda::thread_scope_device> across two blocks
// (producer/consumer) to test correctness AND device forward progress.
#include <cstdio>
#include <hip/hip_runtime.h>
#include <cuda/std/atomic>
#include <cuda/__semaphore/counting_semaphore.h>

__device__ cuda::binary_semaphore<cuda::thread_scope_device> g_sem(0);
__device__ int g_out[2] = {0, 0};

__global__ void producer_consumer() {
  if (blockIdx.x == 0) {
    g_sem.acquire();        // must block until the producer releases
    g_out[0] = 42;
  } else {
    for (volatile long i = 0; i < 4000000; ++i) {}  // let the consumer reach acquire first
    g_out[1] = 7;
    g_sem.release();
  }
}

int main() {
  producer_consumer<<<dim3(2), dim3(1)>>>();
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("LAUNCH/SYNC FAIL: %s\n", hipGetErrorString(e)); return 2; }
  int out[2] = {-1, -1};
  hipMemcpyFromSymbol(out, HIP_SYMBOL(g_out), sizeof(out));
  printf("out[0]=%d (expect 42), out[1]=%d (expect 7)\n", out[0], out[1]);
  bool ok = (out[0] == 42 && out[1] == 7);
  printf("%s: device cuda::binary_semaphore<thread_scope_device> acquire/release on gfx90a\n",
         ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
