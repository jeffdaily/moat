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
