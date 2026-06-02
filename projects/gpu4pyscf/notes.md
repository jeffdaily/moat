# gpu4pyscf notes

ROCm/HIP port. Lead platform linux-gfx90a (MI250X, ROCm 7.2.1). Fork:
https://github.com/jeffdaily/gpu4pyscf , branch `moat-port`. Strategy A
(CUDA spelling kept, mapped to HIP via one compat header).

## Milestone status

- **Milestone 1 (HF / integrals) -- DONE on gfx90a.** gint, gvhf, gvhf-rys,
  gvhf-md, cupy_helper, plus the import-/runtime-coupled gecp, gdft and pbc.
  Validated: see Test Plan in the commit. J/K matches CPU PySCF to <1e-11,
  RHF energy to <1e-11 Ha, deterministic to print precision. The gvhf-md
  wave64 J-engine fix is multi-arch (one fat binary holds gfx90a + gfx1100).
- **Deferred (not built under USE_HIP):**
  - CUTLASS grouped GEMM (cupy_helper/grouped_gemm.cu, grouped_dot.cu) ->
    reimplement as a hipBLAS per-group loop or Composable Kernel grouped GEMM.
    Gates the DFT numint path.
  - External libxc-GPU functional library (wxj6000/libxc, CUDA-only) -> a
    separate libxc-HIP port; scaffold as a sibling MOAT project gpu4pyscf
    `depends_on`. Gates all DFT functionals.
  - DFT (gdft kernels build, but the functional eval path needs libxc-HIP;
    gen_grids double3 init was fixed for HIP), analytic gradients/Hessian
    (the gvhf ejk_ip1 derivative engines), solvent, sem, multigrid, periodic
    DFT/df (pbc.scf/pbc.dft, libmgrid). The single failing scf test
    `test_jk_energy_per_atom` is the analytic gradient engine (deferred);
    `test_scf.py::test_to_cpu/test_to_gpu` fail on the DFT RKS path (deferred).

## Build recipe (gfx90a, or multi-arch)

Env (one-time): in the py_3.12 conda env, cupy_rocm_7_0 is broken by a numpy
ABI mismatch (built for numpy 2.x, env had 1.26.4). Align numpy to 2.x:
`pip install 'numpy>=2.0,<2.3'` (satisfies cupy>=2.0, scipy 1.14 <2.3, numba
<2.5, torch). pyscf 2.13 is installed (the CPU reference / validation gold).
Verify: `python -c "import cupy; cupy.linalg.eigvalsh(cupy.eye(4))"` runs on
gfx90a.

```
export HIP_VISIBLE_DEVICES=0
cmake -S gpu4pyscf/lib -B build -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

The .so are emitted into gpu4pyscf/lib (LIBRARY_OUTPUT_DIRECTORY). The build
also drops unbundled `*.so.0.hipv4-*` / `*.so.0.host-*` RDC offload objects in
that dir -- do not commit them (`.so` is gitignored; the split objects are not,
so `rm -f gpu4pyscf/lib/*.so.0.hipv4-* gpu4pyscf/lib/*.so.0.host-*` before
committing). Run from the repo root with `PYTHONPATH=$PWD`.

Followers: replace the arch list with `gfx1100` (or `gfx1151`). The wave64
J-engine fix is wave-agnostic, so a follower should validate first and only
delta-port on failure.

## Key port facts / gotchas

- gint nr_fill_ao_ints l>=4 (NROOTS 7/8) instantiates a >360 KB per-thread
  gout[] array that overflows the gfx9 scratch frame; those legacy template
  instantiations are #ifndef USE_HIP'd out (the HF path uses gvhf-rys, not this
  legacy engine). nr_fill_ao_ints also compiled at -O1 to fit the frame.
- get_smid: HIP `__smid()` returns a sparse (SE,CU) id that can exceed
  multiProcessorCount; the scratch pool is sized to the full id range under
  USE_HIP (see vhf.cuh).
- pbc int3c2e uses cub::BlockScan/BlockReduce; hip_compat/cub/cub.cuh aliases
  cub -> hipcub. The wave64 `__ballot` path in int3c2e_create_tasks_o1.cuh is
  NOT compiled (no .cu includes it) and is left for the periodic int3c2e
  milestone.
- -ffp-contract=on pins fp contraction so the J/K fingerprints match CPU to
  the test's 7 d.p.; -fgpu-rdc + -fvisibility=default for the cross-TU device
  globals (Rys-root / index tables); -Wno-register for the C++17 register kw.
- hipSOLVER enums differ from cuSOLVER: HIPSOLVER_EIG_TYPE_1=211,
  EIG_MODE_VECTOR=202, HIPBLAS_FILL_MODE_LOWER=122 (cuSOLVER uses 1/1/0).
  Passing the cuSOLVER values gives HIPSOLVER_STATUS_INVALID_ENUM (10). The
  CuPy `get_cusolver_handle()` is a rocSOLVER handle, NOT a hipSOLVER one, so
  eigh creates its own hipSOLVER handle and binds the current stream.

## Install as a dependency

This is a leaf application (a CuPy plugin loaded via PYTHONPATH), not a base
library other MOAT targets build against. No install-as-dependency section
needed. A future libxc-HIP project would be the dependency gpu4pyscf consumes.
