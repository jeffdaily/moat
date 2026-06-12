# Pointcept notes

## Port summary (lead: linux-gfx90a)
Strategy B (torch build-time hipify) port of the four in-tree PyTorch CUDA-extension
GPU libraries under `libs/`: pointops, pointops2, pointgroup_ops, pointrope.
pointseg is a CppExtension (CPU only, no `.cu`) and needs no port. spconv is an
external dependency (separate MOAT project, unported) and does NOT block these libs --
the four libs build, install, and pass their op tests without it. Sparse-conv MODEL
configs (SpUNet/OACNN/PointGroup end-to-end) wait on the spconv ROCm port; that is out
of scope for this port's validation.

## The only source edit
`libs/pointrope/setup.py`: added a `torch.version.hip` branch. hipcc rejects the
nvcc-only flags (`--ptxas-options=-v`, `--use_fast_math`) and the
`cuda.get_gencode_flags()` `-gencode arch=compute_*` list. On ROCm we pass
`["-O3", "-ffast-math"]` and drop the gencode list (target arch comes from
PYTORCH_ROCM_ARCH / --offload-arch). The CUDA path is unchanged.
No `.cu`/`.cpp`/`.cuh`/`.h` kernel source needed editing: no warp intrinsics, no
textures, no cuBLAS/cuFFT/Thrust/CUB, every symbol is hipify-1:1. The other three
setups pass only `-O2`/`-g`/`-O3`, all hipcc-safe, unchanged.

## Build (gfx90a)
ROCm PyTorch env (torch 2.13.0a0, torch.version.hip 7.2.53211, ROCm 7.2.1), GPU
gfx90a (MI250X). Pin `HIP_VISIBLE_DEVICES=0` on this 4-GCD host.

```
export HIP_VISIBLE_DEVICES=0
cd libs/<lib>            # pointops, pointops2, pointgroup_ops, pointrope (in this order)
PYTORCH_ROCM_ARCH=gfx90a python setup.py install
```

All four produce real gfx90a device code objects (verified `roc-obj-ls <ext>.so | grep gfx90a`:
pointops 9 bundles, pointops2 many, pointgroup_ops 1, pointrope 1).

### Build gotcha: hipify artifacts must not be committed
torch's build-time hipify writes `*_hip.cpp`, `*_hip_kernel.h`, `*_hip_kernel.hip`,
`kernels.hip` next to each CUDA source. These are GENERATED; do NOT git-add them
(Strategy B keeps sources in CUDA spelling). `pointops2/setup.py` collects sources via
`os.walk(src)` filtering `.cpp`/`.cu`, so a rebuild over a dirty tree could double-collect
generated `_hip.cpp`. Clean the generated files between builds:
`find libs -type f \( -name '*_hip.cpp' -o -name '*_hip_kernel*.h' -o -name '*_hip_kernel*.hip' -o -name 'kernels.hip' \) -delete`
and remove `libs/*/build libs/*/*.egg-info`. Only `libs/pointrope/setup.py` is a tracked change.

## Test dependency: torch_scatter (env, not a deliverable)
The shipped pointops2 op tests `import torch_scatter`. No prebuilt pyg-rocm wheel matches
torch 2.13a/py3.12, and the PyPI sdist (torch_scatter 2.1.2) fails to build on ROCm 7.2.1:
1. `at::Half __shfl_down_sync` wrapper takes a 32-bit `unsigned` mask; ROCm 7.2.1 now
   static_asserts the mask is 64-bit.
2. its `USE_ROCM` `SHFL_*_SYNC` macros call `__shfl_up(var,delta)`, which is ambiguous
   for `at::Half` (no Half overload) on ROCm 7.2.1.
Both are upstream torch_scatter / ROCm-version issues, independent of the Pointcept port.
Fix in a LOCAL sdist for the test reference only (agent_space, throwaway):
`csrc/cuda/utils.cuh`: widen the wrapper mask to `uint64_t`, and route the `USE_ROCM`
`SHFL_UP_SYNC`/`SHFL_DOWN_SYNC` macros through `__shfl_up_sync`/`__shfl_down_sync`
(the Half-aware wrappers) with `(uint64_t)(mask)`; widen `FULL_MASK` to
`0xffffffffffffffffULL` in `segment_csr_cuda.cu` and `segment_coo_cuda.cu`. Then
`PYTORCH_ROCM_ARCH=gfx90a FORCE_ONLY_CUDA=1 pip install . --no-build-isolation --no-deps`.

## Tests run on gfx90a (all PASS)
Shipped pointops2 op tests (run from `libs/pointops2/functions/`, where `import pointops`
resolves to the local wrapper). Several have interactive `input()` pauses; feed stdin
(`yes "" | python test_...py`). Pass criterion is reference-agreement, printed AFTER the pauses:
- `test_attention_op_step1.py`: v1 vs v2 attention forward, max sq err 9.1e-13; `(diff**2<1e-8).all()`=True.
- `test_attention_op_step1_v2.py`: same, True.
- `test_attention_op_step2.py`: op forward+backward run on GPU OK. NOTE: the test's final
  comparison line is a PRE-EXISTING test bug (references variable `x` whose definition is
  commented out at lines 32-34) -- fails identically on CUDA, not a port issue.
- `test_relative_pos_encoding_op_step1{,_v2,_v3}.py`: rpe forward; v2/v3 vs v1 max sq err 2.3e-10.
- `test_relative_pos_encoding_op_step2.py`: forward+backward run OK.
- `test_relative_pos_encoding_op_step2_v2.py` (most thorough -- forward AND backward grads):
  forward v1-vs-v2 max sq err 7.1e-10; attn_grad 2.6e-21, v_grad 2.6e-21, table_grad 1.3e-16.

Custom op driver `agent_space/pointcept_op_driver.py` (pointops + pointrope have no shipped
tests). Reusable by followers:
- pointops `knn_query`: GPU vs brute-force CPU `torch.cdist` k-NN, set-match fraction 1.0.
- pointrope: GPU vs the extension's own CPU path (same kernel), max abs err 9.1e-6 (fast-math);
  forward-then-inverse round-trip recovers input, max abs err 7.2e-7. positions are int64;
  head dim must be divisible by 6 (kernel uses D=dim/6).

pointgroup_ops driver (inline; ballquery_batch_p + bfs_cluster on GPU):
- ballquery_batch_p: all returned neighbor pairs within radius (0 violations on sampled set).
- bfs_cluster: produces a sensible clustering (1 cluster, 1981/2000 pts for r=0.1 uniform).

## Numeric notes
No fp-contract / fast-math drift observed. pointops2 uses `-O2` (no fast-math); pointrope
uses `-ffast-math` and still agrees with the CPU reference to 9e-6. Risk #5 (fp-contract
drift) did not materialize; no need to pin `-ffp-contract=on`.

## Warp size (followers)
No warp intrinsics anywhere (no `__shfl*`/`__ballot`/`warpSize`/cooperative groups). The FPS
shared-memory reduction in `sampling_cuda_kernel.cu` has an explicit `__syncthreads()` after
every step (including the `tid<32` tail; no implicit warp-synchronous tail), so it is correct
on wave64 (gfx90a) AND wave32 (gfx1100/gfx1201). `<32>` template instantiations are block-tile
sizes, not warp-width. Low warp-size risk; followers should still run the cross-arch consistency
gate.

## PR-prep TODO (lead)
Add a ROCm/AMD build note in the README Installation section (CUDA `setup.py install` block,
~line 157-219) in house style. Scope the PR claim to the `libs/` ops; end-to-end sparse-conv
model validation waits on spconv.

## Review 2026-06-12 (reviewer, linux-gfx90a, read-only)
Reviewed fork moat-port @ 68551e3 vs base d727225 with /pr-review. Verdict: review-passed,
no problems found. Diff is a single file (libs/pointrope/setup.py, +18/-11). Verified
independently: CUDA path byte-for-byte preserved (original nvcc flags + gencode now flow
through the else branch into nvcc_args; only torch.version.hip-is-not-None changes behavior);
no hipify artifacts tracked (git ls-files clean of *_hip.cpp/*_hip_kernel*/*.hip/kernels.hip)
and working tree clean; no warp intrinsics / warpSize / hardcoded-32 / textures / CUDA math
libs / __CUDA_ARCH__ in any of the four libs; both FPS reductions (pointops, pointops2
sampling_cuda_kernel.cu) have an explicit __syncthreads() after every step including the
tid<32 tail (wave-agnostic, no volatile, no implicit warp-sync tail); the other three setups
pass only -O2/-g (hipcc-safe), unchanged. Commit hygiene clean: [ROCm] title <=72 chars,
Claude attribution, Test Plan, no noreply trailer, jeffdaily author, no MOAT jargon, no
AMD-internal account refs. The real-GPU validation run is the validator stage's job (not a
review blocker).
