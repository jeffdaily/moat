# GPUMD notes

## Existing AMD support (pre-port state)

GPUMD already ships an upstream HIP path, so this was a validate-and-improve
job, not a fresh CUDA->HIP conversion:

- `src/makefile.hip` -- `hipcc --offload-arch=gfx90a -DUSE_HIP`, links
  `-lhipblas -lhipsolver -lhipfft`.
- `src/utilities/gpu_macro.cuh` -- Strategy-A compat header. Defines a `gpu*`
  symbol family aliased to `hip*` under `USE_HIP`, else to
  `cuda*`/`cublas*`/`cusolver*`/`cufft*`. Covers memory, stream, error, device
  props, cuRAND, cuBLAS, cuSOLVER Dn eigensolvers, cuFFT.
- 16 `USE_HIP` guard sites; the only behavioral ones select
  `thrust::hip::par` vs `thrust::cuda::par` (rocThrust drop-in).

## CUDA surface

- Libraries: hipBLAS, hipSOLVER (Dn dense eigensolvers), hipFFT, hipRAND,
  rocThrust -- all already wired through gpu_macro.cuh.
- Warp intrinsics: NONE. Reductions are shared-memory block tree-reductions
  with `__syncthreads()` after every step (force.cu:292, neighbor.cu:674,
  nep_charge.cu:582), including sub-warp offsets -- no implicit
  warp-synchronous assumption, so wave64-safe by construction. The warp-size
  fault class does not apply to this project.
- Textures/surfaces: NONE. The texture fault classes are moot.
- Atomics: atomicAdd, atomicCAS only. No atomicMin/Max, so the gfx90a
  coarse-grained int atomicMin/Max-drop bug does not apply.

## Changes made (HIP path only; CUDA makefile untouched)

`src/makefile.hip`, two edits, both behind the existing HIP build:

1. `-std=c++14` -> `-std=c++17`. ROOT CAUSE of the only build failure:
   rocThrust/rocPRIM in ROCm 7.2 hard-`#error` "rocPRIM requires at least
   C++17" (rocprim/config.hpp); the headers use `std::variant`, `std::is_same_v`
   etc. CUDA's Thrust compiles fine under C++14, which is why upstream's
   makefile.hip inherited `-std=c++14` and never caught this -- it bitrots
   against any modern rocThrust. Only the 3 TUs that include thrust
   (neighbor.cu, nep_multigpu.cu, dp.cu) failed; the bump fixes all.
2. Hardcoded `--offload-arch=gfx90a` -> `HIP_ARCH ?= gfx90a` (overridable).
   Followers build `make -f makefile.hip HIP_ARCH=gfx1100` with no file edit,
   so the curated commit head_sha does not churn and already-passed platforms
   are not forced to revalidate. (PORTING_GUIDE configurable-arch rule.)

No source (.cu/.cuh) changes were needed: no wave64 fix (no warp intrinsics),
no texture fix (no textures), no atomic emulation. The existing upstream compat
header and USE_HIP guards were already correct for ROCm 7.2.

## Build (gfx90a)

    cd projects/GPUMD/src/src
    export HIP_VISIBLE_DEVICES=2     # a free gfx90a on the shared host
    make -f makefile.hip clean
    make -f makefile.hip -j16        # default HIP_ARCH=gfx90a
    # produces ./gpumd (8.7 MB) and ./nep (974 KB); only benign nodiscard
    # warnings on ignored hipError_t returns, zero errors.

ROCm 7.2.1, hipcc clang 7.2.53211. Host: 4x gfx90a (MI250X/MI250, CDNA2,
wave64); GPU 3 was busy, used GPU 2.

## Validation (real gfx90a, three independent MD workloads)

Ran shipped regression cases under tests/gpumd/. Each runs a real MD trajectory.

1. carbon -- NEP4 machine-learning potential, NVE, 100 steps. Exercises the NEP
   force kernel + shared-mem reductions.
   - Energy conservation (the NVE physics gate): KE+PE relative drift 2.6e-7,
     std/|mean| 3.8e-7. Temperature 300.4 K (target 300). Correct NVE.
   - Cross-check vs committed CUDA reference (thermo1.out) at first dump
     (step 10): PE matches to 5.8e-6 relative, T/KE to 4.0e-4. Forces match
     CUDA to near fp precision.
2. graphene_dos -- Tersoff potential + phonon DOS via hipFFT, NPT then NVE,
   2x10000 steps. Exercises hipFFT + velocity-autocorrelation.
   - NVE-phase energy conservation: rel drift 2.2e-6, std/|mean| 5.4e-6.
     Temperature mean 297.7 K.
   - DOS output: 200 freq points 2-400 THz, all finite, no negative values
     (physically valid). Spectral shape matches CUDA ref to 5.7e-2.
3. graphene_kappa_emd -- EMD (Green-Kubo) thermal conductivity, heat-current
   autocorrelation.
   - Runs clean, T 297.3 K, HAC finite/well-formed (50x11). HAC tail differs
     more from the CUDA ref (sensitive long-time transport correlation under
     chaos); function is physically valid.

Final error-checked run: empty stderr, no HIP fault (GPUMD's GPU_CHECK_KERNEL
calls gpuGetLastError after kernels), no nan, energy conserved.

## Determinism (root-caused)

Run-to-run on the SAME gfx90a is NOT bit-identical: two fresh carbon runs
diverge to ~16 (in KE units) by step 40, but each run independently conserves
total energy (|dEtot| between runs stays 3e-3..1.6e-1 while KE/PE individually
decorrelate) -- the signature of deterministic physics + chaotic
(Lyapunov) amplification of ULP-level force noise.

Root cause is upstream and vendor-agnostic, NOT a HIP defect:
`force/neighbor.cu:80-81`,
`const int ind = atomicAdd(&cell_count[cell_id], 1); cell_contents[base+ind] = n1;`
-- the atomicAdd return value sets each atom's slot in its cell bin, and
atomicAdd ordering across concurrent threads is non-deterministic on any GPU.
So cell_contents (hence neighbor-list order, hence force-summation order) varies
each run on CUDA too. The velocity init is CPU `rand()` (glibc default seed),
identical run-to-run, so it is not the source.

Consequence for validation: bit-reproducible thermo (what the upstream
`tests/run_tests.sh` `diff -q` expects) is unattainable for chaotic MD across
runs or vendors. The correct acceptance gate is energy conservation +
agreement with the reference within physical tolerance (PE 5.8e-6, DOS 5.7e-2),
which the port passes.

## Followers

gfx1100/gfx1151 (RDNA, wave32): reuse this branch, build with
`make -f makefile.hip HIP_ARCH=<arch>`, no source change. Since the code has no
warp-size assumptions, the lead fix should carry; they validate on their own
hardware per MOAT policy.
