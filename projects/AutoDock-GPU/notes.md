# AutoDock-GPU notes

Already runs on AMD via OpenCL (DEVICE=OCLGPU); this port adds a native HIP/ROCm
backend (DEVICE=HIP) as an improvement, not to fill a gap. The HIP backend ports
the maintained CUDA kernels (cuda/) rather than the parallel OpenCL copy
(device/), so AMD users get the same kernels NVIDIA users get.

## Backend selection (top-level Makefile)

- DEVICE=CUDA / DEVICE=GPU (when nvcc + a CUDA test compile pass) -> Makefile.Cuda
  (nvcc compiles cuda/kernels.cu; g++ links the host .cpp + -lcurand -lcudart).
- DEVICE=OCLGPU / DEVICE=OPENCL / DEVICE=CPU -> Makefile.OpenCL (device/ kernels).
- DEVICE=HIP (NEW) -> Makefile.Hip: hipcc compiles cuda/kernels.cu with
  --offload-arch, g++ links the host .cpp + -lamdhip64. Bypasses the nvcc probe.
cuda/kernels.cu is one TU that #includes the other cuda/*.cu.

## Port = Strategy A (compat header + .cu compiled by hipcc)

- cuda/cuda_to_hip.h: the only file that knows HIP names. Under USE_HIP it
  #includes <cstring>/<cstdlib> BEFORE <hip/hip_runtime.h> (so host memcpy binds
  to libc, not HIP device overloads) and aliases the cuda* runtime symbols the
  project uses to hip*; on NVIDIA it is a plain <cuda_runtime.h> include, so the
  CUDA build is byte-for-byte unchanged. Included from cuda/kernels.cu,
  cuda/GpuData.h, and host/inc/performdocking.h.
- The HIP build defines USE_CUDA too (performdocking.h) so the ~30 shared
  `#ifdef USE_CUDA` GPU-host blocks compile unchanged; the NVIDIA-only includes
  (cuda.h/curand.h/cuda_runtime_api.h) are gated to NOT fire under USE_HIP.
  curand is never actually called (PRNG seeds are made on the host CPU; the
  device PRNG is a hand-rolled LCG in auxiliary_genetic.cu), so no hipRAND link
  is needed.
- Makefile.Hip kernel TU needs -fPIE: g++ links the host as PIE by default, so
  the hipcc-built kernels.o must be position-independent or ld rejects the
  R_X86_64_32 relocation.

## Fault classes hit

### Warp size / wave64 (the real one)
gfx90a is wave64. AutoDock-GPU is partly warp-size parameterized: kernels read
cData.warpmask / cData.warpbits, but the HOST hardcoded them to 32-lane values
and a few device spots assumed 32 lanes. Ported NATIVELY to wave64 (warpmask=63,
warpbits=6), not as two 32-lane halves, because the reductions are genuinely
warp-wide and combine per-warp partials via a block-wide atomicAdd, so a true
64-lane warp is the natural mapping and matches the upstream parameterization.
Fixes:
- host/src/performdocking.cpp ~785: derive warpmask/warpbits from the device's
  props.warpSize at runtime (cudaGetDeviceProperties) instead of the literals
  31/5. Yields 31/5 on CUDA (warpSize 32) and 63/6 on gfx90a (64); zero behavior
  change for NVIDIA.
- cuda/kernels.cu: device-side WARP_SIZE constant (64 for __GFX9__, else 32, else
  CUDA 32). WARPMINIMUM2 gains a stride-32 exchange under wave64; REDUCEFLOATSUM /
  REDUCEINTEGERSUM gain a stride-32 shuffle-add under wave64.
- cuda/kernel4.cu:93 `blockDim.x / 32` -> `blockDim.x / WARP_SIZE` (warp count for
  the final cross-warp reduction).
- HIP mask width: ROCm 7.2 static_asserts that __shfl_sync/__any_sync/__ballot_sync
  take a 64-BIT mask (sizeof==8) regardless of active wave width. WARP_FULL_MASK is
  0xffffffffffffffffULL under USE_HIP, 0xffffffffU on CUDA. (Keyed on USE_HIP, NOT
  on WARP_SIZE>32 -- a 32-bit literal fails to compile even before any wave64
  concern.)

### Inline PTX
cuda/kernels.cu llitoulli/ullitolli use asm("mov.b64"). NVIDIA-only; replaced with
a memcpy bit-cast under USE_HIP (they are pure 64-bit reinterprets; in fact unused
in the codebase, but kept and guarded to preserve the CUDA path).

### Not hit
Atomics: only atomicAdd on int/float (no int atomicMin/Max -> no CAS-emulation
needed). No textures/surfaces -> none of those fault classes apply. One
cudaMallocManaged buffer (pMem_gpu_evals_of_runs) but it is only touched by
atomicAdd and a single-thread +=, so the managed-memory int-atomicMin/Max drop
class does not bite.

## Build (gfx90a)

    cd projects/AutoDock-GPU/src
    make DEVICE=HIP NUMWI=64 HIP_ARCH=gfx90a       # one wavefront/block
    make DEVICE=HIP NUMWI=128 HIP_ARCH=gfx90a      # two wavefronts/block

Arch is configurable (HIP_ARCH ?= gfx90a, default only when unset); followers
build with HIP_ARCH=gfx1100 / gfx1151 and no source/Makefile edit. ROCm 7.2.1,
HIP 7.2, on a 4x MI250X (gfx90a) host. Build artifacts (bin/, kernels.o) are
gitignored upstream.

## Validation -- REAL GPU, gfx90a (HIP_VISIBLE_DEVICES=3, MI250X)

Input: shipped input/1stp (streptavidin-biotin, the canonical AutoDock case),
--nrun 10, fixed --seed. Best binding energy + best-pose reference RMSD (RMSD to
the crystal ligand) from the .dlg RMSD table:

| build / seed     | best binding energy | best-pose reference RMSD | cluster      |
|------------------|---------------------|--------------------------|--------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.48 A                   | 1, 10/10 runs|
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.37 A                   | top cluster  |
| 128wi / seed 42  | -8.33 kcal/mol      | 0.39 A                   | top cluster  |

All runs recover the native biotin pose (sub-0.5 A reference RMSD), best energy
clustered tightly around -8.3 kcal/mol, all 10 runs converging into a single
cluster -- the known-good 1stp result. The GA min-reduction (kernel4 elitism)
and the energy/gradient sum-reductions (calcenergy / calcMergeEneGra) are thus
numerically correct on wave64. The 128wi case (2 wavefronts/block) additionally
exercises the cross-warp final reduction in kernel4 and is also correct.

Cross-check against the OpenCL backend (DEVICE=OCLGPU) was attempted as a
same-machine reference but the upstream OpenCL kernels fail to compile under the
ROCm 7.2 OpenCL runtime (a pre-existing argument-type error in gradient_inter_z,
unrelated to this port) -- which only reinforces the value of the native HIP
backend. The HIP result stands on its own against the established 1stp reference.

## Scope / deliverables

Source edits live as working-tree changes under projects/AutoDock-GPU/src (parent
delivers; not forked/pushed here). 6 files modified + 2 new (cuda/cuda_to_hip.h,
Makefile.Hip); CUDA and OpenCL/CPU paths untouched and behaviorally unchanged. No
GitHub Actions added, README not regenerated.
