// Intra-wavefront forward-progress probe for cuda::binary_semaphore at
// thread_scope_block. Two threads of the SAME block take a producer/consumer
// relationship: thread 0 acquires (must block) and thread 1 releases. On wave32
// (RDNA gfx1100/gfx1201) both threads are lanes of one wavefront but execute
// independently, so this should complete. On wave64 CDNA (gfx90a) the two lanes
// share one wavefront program counter and the acquiring lane spins without ever
// yielding to the releasing lane -> deadlock (the forward-progress hazard).
//
// KEY follower question: does this hazard reproduce on wave32, or does RDNA
// complete it? If RDNA passes, the upstream caveat narrows to CDNA/wave64.
//
// Run with a watchdog: this probe is EXPECTED to hang on wave64. Bound it, e.g.
//   timeout 30 ./sem_block_probe; echo "exit=$? (124 = hang/deadlock)"
#include <cstdio>
#include <hip/hip_runtime.h>
#include <cuda/std/atomic>
#include <cuda/__semaphore/counting_semaphore.h>

__device__ int g_out[2] = {0, 0};

__global__ void block_producer_consumer() {
  __shared__ cuda::binary_semaphore<cuda::thread_scope_block> sem;
  if (threadIdx.x == 0) new (&sem) cuda::binary_semaphore<cuda::thread_scope_block>(0);
  __syncthreads();
  if (threadIdx.x == 0) {
    sem.acquire();          // block until lane 1 releases
    g_out[0] = 42;
  } else if (threadIdx.x == 1) {
    for (volatile long i = 0; i < 4000000; ++i) {}  // let lane 0 reach acquire first
    g_out[1] = 7;
    sem.release();
  }
}

int main() {
  block_producer_consumer<<<dim3(1), dim3(2)>>>();
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("LAUNCH/SYNC FAIL: %s\n", hipGetErrorString(e)); return 2; }
  int out[2] = {-1, -1};
  hipMemcpyFromSymbol(out, HIP_SYMBOL(g_out), sizeof(out));
  printf("out[0]=%d (expect 42), out[1]=%d (expect 7)\n", out[0], out[1]);
  bool ok = (out[0] == 42 && out[1] == 7);
  printf("%s: intra-wavefront cuda::binary_semaphore<thread_scope_block> acquire/release\n",
         ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
