# Integer atomicMin/atomicMax NOP on host-coherent (fine-grained) memory on gfx90a -- EXPECTED, not a runtime bug

## Verdict

**EXPECTED (documented hardware limitation), NOT a ROCm runtime/compiler defect.**

On gfx90a/CDNA2, the plain HIP `atomicMin`/`atomicMax` on 32-bit integers compile
to the hardware `global_atomic_smin`/`global_atomic_smax` instructions. Those
hardware integer min/max atomics are **not supported on memory that has to be
reached over the host-coherent (PCIe / Infinity Fabric host) path** -- i.e.
fine-grained, host-coherent allocations. AMD documents that any atomic the PCIe
bus does not support is turned into a non-atomic load-op-store, and for integer
min/max the PCIe controller's answer is a **NOP** (no read-modify-write happens),
so the value is silently left unchanged. `atomicAdd` and `atomicCAS` do work on
the same memory because add/cmpswap *are* supported on that path. This is the
asymmetry the cudaKDTree spatial builder hit.

The correct way to make integer min/max work on such memory is to either (a) use
a `cmpxchg` (atomicCAS) loop -- which is exactly the cudaKDTree port workaround --
or (b) use the **system-scope** intrinsic `atomicMin_system`/`atomicMax_system`,
which the compiler emits as a `cmpxchg` loop precisely so it is safe on
fine-grained/remote memory, or (c) keep the data in device-local (coarse-grained
or device fine-grained VRAM) memory. So this finding is a **USAGE requirement**,
not a runtime bug to file against ROCm.

One caveat worth flagging upstream as a documentation/usability papercut (not a
correctness bug): the *default* `atomicMin`/`atomicMax` silently NOP instead of
either working or refusing to compile, and the failing allocation is the
`hipMallocManaged` **default**, which is the most natural buffer a CUDA port
reaches for. That silent-NOP-by-default is the trap. See "Recommendation" below.

## Environment

- GPU: AMD Instinct MI250X / MI250 (`gfx90a:sramecc+:xnack-`, CDNA2)
- ROCm: 7.2.1, HIP 7.2.53211-e1a6bc5663
- Compiler: AMD clang 22.0.0git (roc-7.2.1)
- `HIP_VISIBLE_DEVICES=0`

## The variation table (exact, from `repro.hip`)

CPU ref over N=4096 ints in [-10000,9989]: min=-10000, max=9989, sum=-159616.

| allocation                                   | probed coherence | atomicMin | atomicMax | atomicAdd | CAS-min | CAS-max | atomicMin_system | atomicMax_system |
|----------------------------------------------|------------------|-----------|-----------|-----------|---------|---------|------------------|------------------|
| `hipMallocManaged` (default)                 | FINE-grained     | **NO-OP** | **NO-OP** | CORRECT   | CORRECT | CORRECT | CORRECT          | CORRECT          |
| `hipMalloc` (device)                         | COARSE-grained   | CORRECT   | CORRECT   | CORRECT   | CORRECT | CORRECT | CORRECT          | CORRECT          |
| `hipExtMallocWithFlags(Finegrained)`         | FINE-grained     | CORRECT   | CORRECT   | CORRECT   | CORRECT | CORRECT | CORRECT          | CORRECT          |
| `hipHostMalloc` (pinned)                     | FINE-grained     | **NO-OP** | **NO-OP** | CORRECT   | CORRECT | CORRECT | CORRECT          | CORRECT          |
| `hipMallocManaged` + `SetCoarseGrain`        | COARSE-grained   | CORRECT   | CORRECT   | CORRECT   | CORRECT | CORRECT | CORRECT          | CORRECT          |

NO-OP = the kernel ran (`hipDeviceSynchronize` returned success) but the target
was left at its init sentinel (atomicMin left +FLT_MAX-bits = 2139095039;
atomicMax left -2139095039). Exact got/exp values are in `repro.out.txt`.

### The real discriminator is NOT "coarse vs fine"

Note `hipExtMallocWithFlags(hipDeviceMallocFinegrained)` is FINE-grained yet
atomicMin/Max are CORRECT, while `hipMallocManaged` default is also FINE-grained
but NO-OPs. The true axis is **host-coherent / PCIe-reachable** memory:

- `hipMallocManaged` (default, with XNACK) and `hipHostMalloc` are
  **host-coherent** fine-grained memory. RMWs can be routed over the
  PCIe/host-coherent path, where hardware integer min/max is a documented NOP.
- `hipExtMallocWithFlags(hipDeviceMallocFinegrained)` is fine-grained **device-local
  VRAM** (the "EXTENDED FINE GRAINED" pool in `rocminfo`); it never leaves the
  GPU, so the device L2 path services the min/max correctly.
- `hipMalloc` and managed+`SetCoarseGrain` are coarse-grained device VRAM
  (cacheable in L2), also correct.

So the cudaKDTree finding's original title ("coarse-grained hipMallocManaged")
mislabels the memory: the default managed buffer is fine-grained host-coherent,
not coarse-grained. The symptom and the fix are unchanged; only the mechanism
label needed correcting. (The same imprecise phrasing is in an in-source comment
at `projects/cudaKDTree/src/cukd/spatial-kdtree.h:199-202`; the CAS-loop
workaround there is correct and robust regardless.)

## Root cause (codegen evidence)

The plain HIP atomics default to `__HIP_MEMORY_SCOPE_AGENT` *without* the
`[[clang::atomic(fine_grained_memory, remote_memory)]]` attribute, so the AMDGPU
backend emits the bare hardware instruction. The `_system` variants carry that
attribute, so the backend expands them to a `cmpxchg` loop that is safe on
fine-grained/remote memory. ISA from `repro.hip` (`isa_atomic_instructions.txt`):

```
global_atomic_smin   ...   .amdhsa_kernel _Z5k_minPiPKii          <- plain atomicMin: bare HW smin -> NOPs on host-coherent mem
global_atomic_smax   ...   .amdhsa_kernel _Z5k_maxPiPKii          <- plain atomicMax: bare HW smax -> NOPs
global_atomic_add    ...   .amdhsa_kernel _Z5k_addPiPKii          <- atomicAdd: HW add works over PCIe
global_atomic_cmpswap glc  .amdhsa_kernel _Z8k_casminPiPKii       <- CAS-min workaround
global_atomic_cmpswap glc  .amdhsa_kernel _Z8k_casmaxPiPKii       <- CAS-max workaround
global_atomic_cmpswap glc  .amdhsa_kernel _Z12k_min_systemPiPKii  <- atomicMin_system EXPANDED to cmpswap loop
global_atomic_cmpswap glc  .amdhsa_kernel _Z12k_max_systemPiPKii  <- atomicMax_system EXPANDED to cmpswap loop
```

HIP header defining the scopes (no fine-grained attribute on the plain forms):
- `/opt/rocm/include/hip/amd_detail/amd_hip_atomic.h:432-434` -- `atomicMin(int*)`
  -> `__hip_atomic_fetch_min(..., __HIP_MEMORY_SCOPE_AGENT)` (bare).
- `:436-440` -- `atomicMin_system(int*)` wraps it in
  `__HIP_ATOMIC_BACKWARD_COMPAT_MEMORY`, which (`:34`) is
  `[[clang::atomic(fine_grained_memory, remote_memory)]]` -> forces cmpxchg
  expansion.
- `:126-130` -- `atomicCAS(int*)` and `:217-218` -- `atomicAdd(int*)` also default
  to AGENT scope, but cmpswap/add are natively serviceable over the
  host-coherent path, so they are correct without expansion.

## Documentation / upstream citations

1. ROCm "Hardware atomics operation support":
   https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html
   - PCIe atomics tables (fine-grained, device scope) list "32 bit atomicMin /
     atomicMax -> NOP" and "64 bit atomicMin / atomicMax -> NOP" for
     MI100/MI200/gfx9.
   - "Any atomic not supported by the PCIe bus will be a NOP and give incorrect
     result." "If the PCIe bus does not support the requested atomic, the GPU's
     PCIe controller changes it into a load-op-store sequence... the CPU sees it
     as a non-atomic load-op-store sequence."

2. AMD lab notes, "MI200 memory space overview":
   https://github.com/amd/amd-lab-notes/blob/release/mi200-memory-space/Overview.md
   - "Coarse-grained memory is marked as cacheable, and atomic operations will be
     processed in the L2 cache." "Fine-grained memory is marked uncacheable...
     Atomics that hit uncached memory are forwarded to the Infinity Fabric."
   - "managed memory shares a pointer between host and device and (by default)
     supports fine-grained coherence" (i.e. `hipMallocManaged` default is
     fine-grained, NOT coarse-grained).
   - Guidance: "Try to design your algorithms to avoid host-device memory
     coherence (e.g. system scope atomics)."

3. LLVM AMDGPU codegen, which defines exactly which atomics are NOT safe over the
   host-coherent/remote path and must be cmpxchg-expanded:
   https://github.com/llvm/llvm-project/pull/122137
   ("AMDGPU: Expand remaining system atomic operations")
   - Expands "sub, or, xor, and, min, max, umin, umax, ..." for system-scope
     flat/global atomics, but explicitly: "Don't expand xchg and add, those
     theoretically should work over PCIe." This is precisely the observed
     asymmetry: add works, min/max NOP.
   - Related metadata work: https://github.com/llvm/llvm-project/pull/122138

## Search for prior reports

No ROCm/HIP, ROCm/clr, or ROCm/rocm-systems issue was found reporting integer
`atomicMin`/`atomicMax` NOP specifically on managed/fine-grained memory; the
nearest hits are about *float/double* atomicMin/Max being implemented as CAS
loops (e.g. ROCm/hipamd#65, LLVM D108208), which is a different topic. The
behavior here is consistent with the documented PCIe-atomics limitation, so it is
a known/expected hardware constraint rather than an unreported runtime bug.

## Recommendation

**Do not file a correctness bug** -- this is documented, expected hardware
behavior. The cudaKDTree port fix (CAS-loop `atomicMinI32`/`atomicMaxI32`/
`atomicMinU32`, HIP-guarded) is the correct and portable resolution and should
stay.

The only thing potentially worth raising upstream is a **documentation/usability
papercut**, NOT a bug: the plain `atomicMin`/`atomicMax` silently NOP on the
default `hipMallocManaged` buffer rather than working or warning. Equivalent
in-spec alternatives that DO work on host-coherent memory, in order of
preference for a port:

1. CAS-loop emulation (what cudaKDTree does) -- portable to CUDA and HIP.
2. `atomicMin_system`/`atomicMax_system` -- compiler expands to a safe cmpxchg
   loop; HIP-only spelling.
3. Keep the accumulator in device-local memory (`hipMalloc`, or
   `hipMemAdviseSetCoarseGrain` on the managed range) and copy out -- only viable
   if the host does not need coherent concurrent access.

## Reproducers

- `repro.hip` -- the full variation table (op x allocation), self-checking, pure
  HIP (no cudaKDTree dependency). Output recorded in `repro.out.txt`.
- `coherence_probe.hip` -- probes `hipMemRangeAttributeCoherencyMode` for each
  allocation, proving `hipMallocManaged` default is fine-grained. Output in
  `coherence_probe.out.txt`.
- `isa_atomic_instructions.txt` -- the load-bearing ISA showing plain min/max ->
  `global_atomic_smin/smax` vs `_system`/CAS -> `global_atomic_cmpswap`.

Build/run:
```
hipcc --offload-arch=gfx90a -O2 repro.hip -o repro && HIP_VISIBLE_DEVICES=0 ./repro
hipcc --offload-arch=gfx90a -O2 coherence_probe.hip -o coherence_probe && HIP_VISIBLE_DEVICES=0 ./coherence_probe
```
