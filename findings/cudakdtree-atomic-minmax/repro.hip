// Standalone repro: integer atomicMin/atomicMax silently no-op on
// coarse-grained hipMallocManaged memory on gfx90a (CDNA2), while
// atomicAdd / atomicCAS on the SAME memory work.
//
// Surfaced by cudaKDTree's spatial builder (atomic_grow). This file has NO
// dependency on cudaKDTree; it is pure HIP.
//
// Build:
//   hipcc --offload-arch=gfx90a -O2 repro.hip -o repro
// Run:
//   HIP_VISIBLE_DEVICES=0 ./repro
//
// The variation table printed is:
//   (op: atomicMin / atomicMax / atomicAdd / CAS-min / CAS-max)
//     x (alloc: coarse-managed / device / fine-grained-managed / hostmalloc /
//        managed+SetCoarseGrain)
//   -> CORRECT or NO-OP.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__,       \
             __LINE__);                                                        \
    }                                                                          \
  } while (0)

// Sentinels: int min starts at INT-extreme so any real value updates it.
static const int32_t SENT_MIN = 2139095039;   // +FLT_MAX bit pattern (large +)
static const int32_t SENT_MAX = -2139095039;  // large negative

// ---- kernels: the plain (default-scope, AGENT) HIP atomics --------------
__global__ void k_min(int32_t* v, const int32_t* in, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  atomicMin(&v[0], in[i]);
}
__global__ void k_max(int32_t* v, const int32_t* in, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  atomicMax(&v[0], in[i]);
}
__global__ void k_add(int32_t* v, const int32_t* in, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  atomicAdd(&v[0], in[i]);
}

// ---- the cudaKDTree workaround: CAS-loop emulation ----------------------
__device__ int32_t casMin(int32_t* addr, int32_t val) {
  int32_t old = *addr, assumed;
  do {
    assumed = old;
    if (assumed <= val) break;
    old = atomicCAS((int*)addr, assumed, val);
  } while (assumed != old);
  return old;
}
__device__ int32_t casMax(int32_t* addr, int32_t val) {
  int32_t old = *addr, assumed;
  do {
    assumed = old;
    if (assumed >= val) break;
    old = atomicCAS((int*)addr, assumed, val);
  } while (assumed != old);
  return old;
}
__global__ void k_casmin(int32_t* v, const int32_t* in, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  casMin(&v[0], in[i]);
}
__global__ void k_casmax(int32_t* v, const int32_t* in, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  casMax(&v[0], in[i]);
}

// ---- the same min, but with an explicit SYSTEM-scope atomic -------------
// atomicMin_system carries [[clang::atomic(fine_grained_memory,remote_memory)]]
__global__ void k_min_system(int32_t* v, const int32_t* in, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  atomicMin_system(&v[0], in[i]);
}
__global__ void k_max_system(int32_t* v, const int32_t* in, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  atomicMax_system(&v[0], in[i]);
}

enum Op { MIN, MAX, ADD, CASMIN, CASMAX, MIN_SYS, MAX_SYS };
const char* opName(Op o) {
  switch (o) {
    case MIN: return "atomicMin";
    case MAX: return "atomicMax";
    case ADD: return "atomicAdd";
    case CASMIN: return "CAS-min  ";
    case CASMAX: return "CAS-max  ";
    case MIN_SYS: return "atomicMin_system";
    case MAX_SYS: return "atomicMax_system";
  }
  return "?";
}

void launch(Op op, int32_t* v, const int32_t* in, int n) {
  dim3 grid((n + 127) / 128), block(128);
  switch (op) {
    case MIN: k_min<<<grid, block>>>(v, in, n); break;
    case MAX: k_max<<<grid, block>>>(v, in, n); break;
    case ADD: k_add<<<grid, block>>>(v, in, n); break;
    case CASMIN: k_casmin<<<grid, block>>>(v, in, n); break;
    case CASMAX: k_casmax<<<grid, block>>>(v, in, n); break;
    case MIN_SYS: k_min_system<<<grid, block>>>(v, in, n); break;
    case MAX_SYS: k_max_system<<<grid, block>>>(v, in, n); break;
  }
}

int32_t initFor(Op op) {
  if (op == MAX || op == CASMAX || op == MAX_SYS) return SENT_MAX;
  if (op == ADD) return 0;
  return SENT_MIN;  // min ops
}

int32_t expectFor(Op op, int cmin, int cmax, long csum) {
  if (op == MAX || op == CASMAX || op == MAX_SYS) return cmax;
  if (op == ADD) return (int32_t)csum;
  return cmin;
}

// allocation strategies
enum Alloc { COARSE_MANAGED, DEVICE, FINE_MANAGED, HOSTMALLOC, MANAGED_SET_COARSE };
const char* allocName(Alloc a) {
  switch (a) {
    // NOTE: hipMallocManaged default is FINE-grained host-coherent on gfx90a
    // (confirmed by coherence_probe.hip), despite the historical "coarse" label.
    case COARSE_MANAGED: return "managed-default (hipMallocManaged, fine/host-coherent)";
    case DEVICE: return "device (hipMalloc, coarse VRAM)";
    case FINE_MANAGED: return "fine-device-VRAM (hipExtMallocWithFlags Finegrained)";
    case HOSTMALLOC: return "hostmalloc (hipHostMalloc, fine/host-coherent)";
    case MANAGED_SET_COARSE: return "managed+hipMemAdviseSetCoarseGrain";
  }
  return "?";
}

// Allocate v (1 int) and in (n ints) per strategy. in is always filled from h.
// Returns whether device-side; caller frees with the matching free.
struct Buffers {
  int32_t* v = nullptr;
  int32_t* in = nullptr;
  bool device_only = false;  // needs memcpy to read back
};

Buffers makeBuffers(Alloc a, const std::vector<int32_t>& h) {
  int n = (int)h.size();
  Buffers b;
  switch (a) {
    case COARSE_MANAGED: {
      CHECK(hipMallocManaged(&b.v, sizeof(int32_t)));
      CHECK(hipMallocManaged(&b.in, n * sizeof(int32_t)));
      memcpy(b.in, h.data(), n * sizeof(int32_t));
      break;
    }
    case DEVICE: {
      CHECK(hipMalloc(&b.v, sizeof(int32_t)));
      CHECK(hipMalloc(&b.in, n * sizeof(int32_t)));
      CHECK(hipMemcpy(b.in, h.data(), n * sizeof(int32_t), hipMemcpyHostToDevice));
      b.device_only = true;
      break;
    }
    case FINE_MANAGED: {
      // True fine-grained device-visible: hipExtMallocWithFlags +
      // hipDeviceMallocFinegrained.
      CHECK(hipExtMallocWithFlags((void**)&b.v, sizeof(int32_t),
                                  hipDeviceMallocFinegrained));
      CHECK(hipExtMallocWithFlags((void**)&b.in, n * sizeof(int32_t),
                                  hipDeviceMallocFinegrained));
      memcpy(b.in, h.data(), n * sizeof(int32_t));
      break;
    }
    case HOSTMALLOC: {
      CHECK(hipHostMalloc(&b.v, sizeof(int32_t)));
      CHECK(hipHostMalloc(&b.in, n * sizeof(int32_t)));
      memcpy(b.in, h.data(), n * sizeof(int32_t));
      break;
    }
    case MANAGED_SET_COARSE: {
      CHECK(hipMallocManaged(&b.v, sizeof(int32_t)));
      CHECK(hipMallocManaged(&b.in, n * sizeof(int32_t)));
      CHECK(hipMemAdvise(b.v, sizeof(int32_t), hipMemAdviseSetCoarseGrain, 0));
      CHECK(hipMemAdvise(b.in, n * sizeof(int32_t), hipMemAdviseSetCoarseGrain, 0));
      memcpy(b.in, h.data(), n * sizeof(int32_t));
      break;
    }
  }
  return b;
}

void freeBuffers(Alloc a, Buffers& b) {
  if (a == HOSTMALLOC) {
    hipHostFree(b.v);
    hipHostFree(b.in);
  } else {
    hipFree(b.v);
    hipFree(b.in);
  }
}

int32_t setInit(Buffers& b, int32_t init) {
  if (b.device_only) {
    CHECK(hipMemcpy(b.v, &init, sizeof(int32_t), hipMemcpyHostToDevice));
  } else {
    *b.v = init;
  }
  return init;
}

int32_t readBack(Buffers& b) {
  if (b.device_only) {
    int32_t hv;
    CHECK(hipMemcpy(&hv, b.v, sizeof(int32_t), hipMemcpyDeviceToHost));
    return hv;
  }
  return *b.v;
}

int main() {
  const int N = 4096;
  std::vector<int32_t> h(N);
  int cmin = 0x7fffffff, cmax = (int)0x80000000;
  long csum = 0;
  for (int i = 0; i < N; i++) {
    h[i] = (int)((i * 2654435761u) % 20000) - 10000;  // range [-10000,9999]
    cmin = std::min(cmin, h[i]);
    cmax = std::max(cmax, h[i]);
    csum += h[i];
  }
  printf("N=%d  CPU ref: min=%d max=%d sum=%ld\n\n", N, cmin, cmax, csum);

  int dev = 0;
  CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

  Alloc allocs[] = {COARSE_MANAGED, DEVICE, FINE_MANAGED, HOSTMALLOC,
                    MANAGED_SET_COARSE};
  Op ops[] = {MIN, MAX, ADD, CASMIN, CASMAX, MIN_SYS, MAX_SYS};

  // Header
  printf("%-50s", "allocation \\ op");
  for (Op o : ops) printf(" | %-16s", opName(o));
  printf("\n");

  for (Alloc a : allocs) {
    printf("%-50s", allocName(a));
    for (Op o : ops) {
      Buffers b = makeBuffers(a, h);
      int32_t init = initFor(o);
      setInit(b, init);
      CHECK(hipDeviceSynchronize());
      launch(o, b.v, b.in, N);
      CHECK(hipDeviceSynchronize());
      int32_t got = readBack(b);
      int32_t exp = expectFor(o, cmin, cmax, csum);
      const char* verdict;
      if (got == exp)
        verdict = "CORRECT";
      else if (got == init)
        verdict = "NO-OP";
      else
        verdict = "WRONG";
      // print compact: verdict only, value in detail pass below
      printf(" | %-16s", verdict);
      freeBuffers(a, b);
    }
    printf("\n");
  }

  // Detailed pass: exact values for the cells of interest.
  printf("\n--- exact values (got vs expected) ---\n");
  for (Alloc a : allocs) {
    for (Op o : ops) {
      Buffers b = makeBuffers(a, h);
      int32_t init = initFor(o);
      setInit(b, init);
      CHECK(hipDeviceSynchronize());
      launch(o, b.v, b.in, N);
      CHECK(hipDeviceSynchronize());
      int32_t got = readBack(b);
      int32_t exp = expectFor(o, cmin, cmax, csum);
      printf("  %-42s %-16s init=%-11d got=%-11d exp=%-11d %s\n",
             allocName(a), opName(o), init, got, exp,
             (got == exp) ? "OK" : "**MISMATCH**");
      freeBuffers(a, b);
    }
  }
  return 0;
}
