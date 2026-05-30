// Probe the coherence mode of each allocation so the variation table can be
// labeled by the REAL coarse/fine axis (not by the API name). On gfx90a with
// XNACK, hipMallocManaged returns FINE-grained (host-coherent) memory, which is
// why the integer min/max hardware atomic NOPs on it.
//
// Build: hipcc --offload-arch=gfx90a -O2 coherence_probe.hip -o coherence_probe
// Run:   HIP_VISIBLE_DEVICES=0 ./coherence_probe

#include <hip/hip_runtime.h>
#include <cstdio>

static const char* coh(unsigned v) {
  // hipMemRangeCoherencyMode: 0 = FineGrain, 1 = CoarseGrain, 2 = Indeterminate
  switch (v) {
    case 0: return "FINE-grained";
    case 1: return "COARSE-grained";
    default: return "indeterminate";
  }
}

static void probe(const char* tag, void* p, size_t bytes) {
  unsigned mode = 99;
  hipError_t e = hipMemRangeGetAttribute(&mode, sizeof(mode),
                                         hipMemRangeAttributeCoherencyMode, p,
                                         bytes);
  if (e != hipSuccess) {
    printf("  %-42s coherency query unsupported (%s)\n", tag,
           hipGetErrorString(e));
    return;
  }
  printf("  %-42s -> %s (mode=%u)\n", tag, coh(mode), mode);
}

int main() {
  const size_t B = 256;
  void* p;

  hipMallocManaged(&p, B);
  probe("hipMallocManaged (default)", p, B);
  hipFree(p);

  hipMallocManaged(&p, B);
  hipMemAdvise(p, B, hipMemAdviseSetCoarseGrain, 0);
  probe("hipMallocManaged + SetCoarseGrain", p, B);
  hipFree(p);

  hipExtMallocWithFlags(&p, B, hipDeviceMallocFinegrained);
  probe("hipExtMallocWithFlags(Finegrained)", p, B);
  hipFree(p);

  hipHostMalloc(&p, B);
  probe("hipHostMalloc (default)", p, B);
  hipHostFree(p);

  hipMalloc(&p, B);
  probe("hipMalloc (device)", p, B);
  hipFree(p);

  return 0;
}
