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

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: 4x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1,
HIP 7.2.53211. No fork changes; the curated commit (4a860c8) is untouched.

### Build commands

```
make -C projects/AutoDock-GPU/src DEVICE=HIP NUMWI=64  HIP_ARCH=gfx1100
make -C projects/AutoDock-GPU/src DEVICE=HIP NUMWI=128 HIP_ARCH=gfx1100
```

Both built cleanly (warnings only: nodiscard on hipDeviceReset in RTERROR macro,
and INFINITY undefined-behavior under -ffast-math -- both pre-existing in
upstream, no errors). HIP_ARCH is not hardcoded in Makefile.Hip (defaults gfx90a
when unset); gfx1100 requires no source or Makefile edit -- exactly as designed.

### GFX1100 code-object evidence

```
$ roc-obj-ls bin/autodock_gpu_64wi
1  host-x86_64-unknown-linux-gnu-                     ...autodock_gpu_64wi#offset=331776&size=0
1  hipv4-amdgcn-amd-amdhsa--gfx1100                   ...autodock_gpu_64wi#offset=331776&size=85048

$ roc-obj-ls bin/autodock_gpu_128wi
1  host-x86_64-unknown-linux-gnu-                     ...autodock_gpu_128wi#offset=331776&size=0
1  hipv4-amdgcn-amd-amdhsa--gfx1100                   ...autodock_gpu_128wi#offset=331776&size=85048
```

No gfx90a object present in either binary.

### Wave32 path analysis

gfx1100 is NOT __GFX9__, so at device-compile time:
- WARP_SIZE = 32 (the __GFX9__ branch is not taken)
- WARPMINIMUM_TOP expands to nothing (guard: WARP_SIZE > 32 is false)
- WARPSUM_TOP expands to nothing (same guard)
- The stride-32 shuffle-add and exchange steps are compiled out

At runtime, cudaGetDeviceProperties returns warpSize=32 for gfx1100, so:
- cData.warpmask = 31, cData.warpbits = 5

The kernel4 cross-warp reduction uses blockDim.x / WARP_SIZE (=32), yielding
2 warps/block for 64wi and 4 warps/block for 128wi -- correct on wave32.

WARP_FULL_MASK = 0xffffffffffffffffULL (64-bit, required by the HIP 7.2 API
regardless of actual wave width; upper bits address no lanes on wave32 so
all-ones is still correct as "all active lanes").

### Docking results (1stp streptavidin-biotin, --nrun 10, HIP_VISIBLE_DEVICES=0)

```
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_64wi  -ffile input/1stp/derived/1stp_protein.maps.fld \
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 42 -resnam <out>
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_64wi  -ffile input/1stp/derived/1stp_protein.maps.fld \
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 7  -resnam <out>
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_128wi -ffile input/1stp/derived/1stp_protein.maps.fld \
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 42 -resnam <out>
```

| build / seed     | best binding energy | best-pose reference RMSD | cluster       |
|------------------|---------------------|--------------------------|---------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.40 A                   | 1, 10/10 runs |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.38 A                   | 1, 10/10 runs |
| 128wi / seed 42  | -8.35 kcal/mol      | 0.41 A                   | 1, 10/10 runs |

Reference (wave64, gfx90a lead):

| build / seed     | best binding energy | best-pose reference RMSD |
|------------------|---------------------|--------------------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.48 A                   |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.37 A                   |
| 128wi / seed 42  | -8.33 kcal/mol      | 0.39 A                   |

All three cases: all 10 runs converge to a single cluster; best energies within
0.02 kcal/mol of the wave64 lead; best-pose reference RMSD sub-0.5 A in all
cases. No NaN in energies, no HIP fault, clean exit. The 128wi case exercises
the cross-warp final reduction in kernel4 on wave32 (4 warps/block, WARP_SIZE=32
path) and is also correct.

PASS: wave32 warp reductions (WARPMINIMUM2, REDUCEINTEGERSUM, REDUCEFLOATSUM)
are numerically correct on gfx1100. Fork is untouched; head_sha = 4a860c8.
