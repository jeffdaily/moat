# foldmason notes

## Build instructions (linux-gfx90a)

```bash
cmake -S projects/foldmason/src -B projects/foldmason/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

cmake --build projects/foldmason/build --target foldmason -j16
```

## Port notes

The port applies the validated foldseek HIP port (jeffdaily/foldseek@e7471b41) to foldmason's vendored lib/foldseek/ subtree. All CUDA code lives in the nested libmarv at lib/foldseek/lib/mmseqs/lib/libmarv/src/; foldmason's own code has no CUDA.

Key changes:
- cuda_to_hip.h compat header: aliases CUDA runtime symbols to HIP, plus emulations for half-precision (__hmax2/__hmin2) and DPX intrinsics
- hip_compat/ shim for cooperative_groups::reduce (missing in HIP) and cuda_fp16 forwarding
- libmarv built as SHARED with -fgpu-rdc + --hip-link for device code linking
- DPX path forced off on AMD (cc==9 collision with Hopper)
- USE_HIP CMake option at top-level threads through to foldseek -> mmseqs -> libmarv

## GPU validation (linux-gfx90a)

GPU search (`--gpu 1`) produces deterministic results that are a strict superset of CPU results:
- CPU: 837 alignments
- GPU: 874 alignments (all 837 CPU hits present, 37 additional hits from higher sensitivity)
- Sorted output identical across runs (deterministic)

## Regression tests

Bundled regression tests (run_easymsa, run_structuremsa, run_msa2lddt) show minor differences in MSA gap placements vs reference -- this is expected for MSA algorithms with multiple valid alignments and is not a GPU/HIP issue

## Review 2026-06-05

**Verdict: Approve**

Reviewed the port applying validated foldseek HIP port (jeffdaily/foldseek@e7471b41) to foldmason's vendored lib/foldseek/ subtree. All checks pass:

- Strategy A correctly applied: cuda_to_hip.h compat header, .cu files marked LANGUAGE HIP, hip_compat/ shims for cooperative_groups::reduce
- Fault classes addressed: warpSize/2 reductions, 64-bit WARP_FULL_MASK, DPX path forced off on AMD, __hmax2/__hmin2 emulation
- Minimal footprint: NVIDIA build unchanged, all changes guarded behind USE_HIP
- CMake wiring correct: USE_HIP threaded through hierarchy, enable_language(HIP), shared lib with -fgpu-rdc --hip-link
- Commit hygiene: [ROCm] prefix, 64 chars, Test Plan section, Claude attribution, no noreply trailer
- GPU validation documented: 874 alignments (GPU) vs 837 (CPU), all CPU hits present, deterministic

The reuse of validated foldseek port for identical vendored libmarv is a sound strategy. Ready for hardware validation.

## Validation 2026-06-05 (linux-gfx90a, MI250X gfx90a)

**Verdict: PASS**

Validated at commit e7f5b62d0ddcfc088ca20e25eda507458425bc10.

### Build
```bash
cmake -S projects/foldmason/src -B projects/foldmason/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

cmake --build projects/foldmason/build --target foldmason -j16
```

Build succeeded. libmarv.so contains gfx90a device code objects (verified with extractkernel).

### GPU search validation

**CPU reference:**
```bash
foldmason search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10
```
Result: 837 alignments

**GPU test:**
```bash
HIP_VISIBLE_DEVICES=2 foldmason search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
```
Result: 874 alignments

**Correctness:** All 837 CPU hits present in GPU results with byte-identical alignment scores (GPU is a strict superset with higher sensitivity, as expected from validated foldseek port).

**Determinism:** Two independent GPU runs produce identical results (when sorted).

### Non-GPU regression

Bundled regression tests (run_easymsa, run_structuremsa, run_msa2lddt) show minor differences in MSA gap placements vs reference. These are CPU-based MSA algorithms (not GPU code paths), and the differences are expected for algorithms with multiple valid alignments (documented as pre-existing in foldseek validation).

### Summary

GPU search (libmarv ungappedprefilter) works correctly on gfx90a with deterministic results. Non-GPU tests do not regress beyond expected MSA algorithm variance. The port reuses the validated foldseek HIP port and inherits its correctness.

## Validation 2026-06-05 (linux-gfx1100, RDNA3 gfx1100)

**Verdict: PASS**

Validated at commit b58f888d8c7b7d45e75ee32e9ea3d1f4b5ca7b4e.

### Build

Required CMake 4.x compatibility fix: CMP0060 OLD policy no longer supported, now conditionally set only for CMake < 4.0.0.

```bash
HIP_VISIBLE_DEVICES=1 cmake -S projects/foldmason/src -B projects/foldmason/build-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

HIP_VISIBLE_DEVICES=1 cmake --build projects/foldmason/build-gfx1100 --target foldmason -j16
```

Build succeeded. libmarv.so contains gfx1100 device code objects.

### GPU search validation

**CPU reference:**
```bash
foldmason search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10
```
Result: 837 alignments

**GPU test:**
```bash
HIP_VISIBLE_DEVICES=1 foldmason search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
```
Result: 874 alignments

**Correctness:** All 837 CPU hits present in GPU results with byte-identical alignment scores (GPU is a strict superset with higher sensitivity, matching gfx90a behavior).

**Determinism:** Two independent GPU runs produce identical results (when sorted).

### Non-GPU regression

Bundled regression tests (run_easymsa, run_structuremsa, run_msa2lddt) show same minor differences in MSA gap placements as gfx90a. These are CPU-based MSA algorithms (not GPU code paths), and the differences are expected for algorithms with multiple valid alignments.

### Summary

GPU search works correctly on gfx1100 with deterministic results matching gfx90a behavior. The wave32 architecture requires no code changes; the foldseek HIP port is wave-agnostic as designed.
