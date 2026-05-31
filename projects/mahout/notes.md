# mahout notes

Port target is the QDP (Quantum Data Plane) Rust workspace under `qdp/`, NOT the
JVM Mahout. Native engine: `qdp-kernels` (6 hand-written `.cu`, ~20 encoder/L2-norm
kernels) compiled by the Rust `cc` crate, driven from `qdp-core` through the
`cudarc` crate + a CUDA-runtime FFI shim. The port adds an AMD/HIP build behind a
Cargo `hip` feature (and `QDP_USE_HIP=1`); the NVIDIA build (default `cuda`
feature) is byte-for-byte unchanged. The separate Triton `backend="amd"` Python
path is orthogonal and untouched.

## Environment (lead: linux-gfx90a)
- ROCm 7.2.1, hipcc 7.2 (AMD clang 22), 4x MI250X GCDs (gfx90a, wave64).
- Rust stable via rustup (workspace needs edition 2024 / rust 1.85+).
- Python: conda env `py_3.12` already has ROCm torch 2.13 (hip 7.2, 4 devices),
  pytest 9. The uv `dev` group pins a generic torch and, if installed, would
  clobber the working ROCm torch -- so we install only the `_qdp` extension into
  the existing env (see below), never `maturin develop` (which runs the dev-group
  pip install).

## What the port does

### A. Kernels (`qdp/qdp-kernels`) -- same `.cu` compiled by hipcc
- `build.rs`: added a HIP branch (`build_hip`) gated by `hip_requested()` (the
  Cargo `hip` feature OR `QDP_USE_HIP=1`). It compiles the same six `.cu` with
  hipcc, `--offload-arch` from `QDP_HIP_ARCH_LIST` (default `gfx90a` only when
  unset -- never a hardcoded literal that overrides the env, so followers
  gfx1100/gfx1151 build the same source with only `QDP_HIP_ARCH_LIST=<arch>`),
  and links `amdhip64`. The CUDA branch (nvcc, gencode, cudart) is unchanged.
- hipcc ships NO `<cuda_runtime.h>`, `<cuComplex.h>`, or `<vector_types.h>`. Rather
  than edit every include line, `qdp-kernels/hip_compat/` holds forwarding shim
  headers of those exact names (MPPI lesson); `build_hip` adds that dir to the
  include path FIRST. On the CUDA build the dir is absent, so the real toolkit
  headers win -> CUDA path untouched.
  - `hip_compat/cuda_runtime.h` -> `<hip/hip_runtime.h>` + the small set of cuda*
    runtime aliases the kernels use (cudaError_t, cudaSuccess,
    cudaErrorInvalidValue, cudaStream_t, cudaGetLastError, cudaGetDevice,
    cudaDeviceGetAttribute, cudaDevAttrMaxGridDimX, cudaMemsetAsync, cudaMalloc).
  - `hip_compat/cuComplex.h` -> `<hip/hip_complex.h>` + aliases. cuDoubleComplex/
    cuComplex -> hipDoubleComplex/hipFloatComplex; make_cu* -> make_hip*. The
    `cuC*` ops (cuCreal/cuCimag/cuCadd/cuCsub/cuCmul/cuConj) are called only on
    cuDoubleComplex in iqp.cu, so they alias to HIP's DOUBLE helpers (hipCreal,
    ... -- NOT the float `f` set) via tiny inline wrappers.
  - `hip_compat/vector_types.h` -> `<hip/hip_runtime.h>` (HIP provides double2/
    float2 there).
- `src/kernel_compat.h` (NEW, included by amplitude.cu): on HIP defines
  `QDP_FULL_WARP_MASK = 0xffffffffffffffffULL`, on CUDA `0xffffffffu`.
- `src/amplitude.cu` source fixes (the only kernel needing them; all others are
  warp-agnostic and compile unchanged):
  - `__shfl_down_sync(0xffffffff, ...)` x2 -> `__shfl_down_sync(QDP_FULL_WARP_MASK, ...)`.
    ROCm 7.x static_asserts a 64-bit mask (`sizeof(MaskT)==8`); the 32-bit literal
    fails to COMPILE (confirmed). CUDA keeps the 32-bit value.
  - `int warp_id = threadIdx.x >> 5;` x2 -> `threadIdx.x / warpSize`. A genuine
    wave64 CORRECTNESS bug: `>> 5` assumes 32-lane warps, so on gfx90a the
    per-warp L2-norm partial lands in the wrong shared slot and the final
    reduction reads the wrong slot -> wrong norm. `/ warpSize` is arch-unified
    (== `>>5` on CUDA/RDNA wave32, `>>6` on CDNA wave64). The `lane =
    threadIdx.x & (warpSize-1)` and `__shared__ shared[32]` were already correct
    (16 warps max at 1024 threads on wave64).

### B. THE LINCHPIN -- displacing cudarc on the HIP build (cc-crate, NOT cudarc-over-HIP)
cudarc 0.13 is CUDA-only (no ROCm backend, no `dynamic-linking`-to-HIP that works
with its `CudaSlice`/`DeviceRepr` semantics). Approach B1 (thin HIP runtime shim)
was implemented in full; B2 (cudarc-over-HIP) was not attempted (cudarc's sys
layer hard-binds the CUDA driver). The whole cudarc surface QDP uses is small and
uniform, so a same-named shim collapses every call site with zero body changes:

- `qdp-kernels/src/device.rs` (NEW): vendor-selected device module.
  - On `cuda`: `pub use cudarc::driver::{CudaDevice, CudaSlice, CudaStream,
    DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, ValidAsZeroBits}`.
  - On `hip`: a self-contained shim with the SAME type names + method signatures,
    backed by `extern "C"` libamdhip64 calls. `CudaDevice` (ordinal + bind),
    `CudaSlice<T>` (owns hipMalloc'd ptr as u64, Drop -> hipFree), `CudaViewMut`
    (slice_mut sub-view), `CudaStream { pub stream: *mut c_void }`. Methods:
    `new`, `alloc`(unsafe), `alloc_zeros`, `htod_sync_copy`, `htod_copy`,
    `htod_sync_copy_into`(generic over DevicePtrMut so the slice_mut view works),
    `dtoh_sync_copy`, `synchronize`, `ordinal`, `fork_default_stream`, `wait_for`.
    Marker traits `DeviceRepr`/`ValidAsZeroBits` and accessor traits
    `DevicePtr`/`DevicePtrMut`(return `&u64` so `*x.device_ptr() as *mut T` is
    unchanged)/`DeviceSlice`. `DriverError(i32)` is a Debug wrapper (call sites
    only `{:?}` it). The marker traits live in qdp-kernels (lowest crate) because
    it impls them on its complex structs.
- `qdp-core/src/gpu_rt.rs` (NEW): `pub use qdp_kernels::device::{...}` -- the
  single import point. Every `use cudarc::driver::{...}` in qdp-core src + tests
  became `use crate::gpu_rt::{...}` (or `qdp_core::gpu_rt::` in tests);
  `safe::CudaStream` flattened to `CudaStream`. ~16 src files + 3 test files +
  2 qdp-kernels test files (-> `qdp_kernels::device`). Bodies are byte-identical.
- `qdp-core/src/gpu/cuda_ffi.rs`: kept the public `cuda*` fn names + constants the
  pinned-pool/OOM-guard/pipeline call; split into a `cuda_rt` mod (extern
  libcudart, default) and a `hip_rt` mod (thin wrappers over libamdhip64:
  hipHostMalloc/hipHostFree/hipMemGetInfo/hipMemcpyWithStream/hipMemcpy/hipEvent*/
  hipStream*/hipMemsetAsync/hipPointerGetAttributes), selected by feature. Added
  `cudaMemcpy` (sync) + `CUDA_MEMCPY_DEVICE_TO_HOST`.
- `qdp-core/src/gpu/metrics.rs`: the two `download_complex_*` test helpers used
  cudarc's raw `sys::lib().cuMemcpyDtoH_v2`; now use cuda_ffi's `cudaMemcpy`
  (D2H) -- cross-vendor and out of cudarc's sys layer.
- Cargo features: `qdp-kernels` {default=["cuda"], cuda=["dep:cudarc"], hip=[]};
  `qdp-core` {default=["cuda"], cuda=["dep:cudarc","qdp-kernels/cuda"],
  hip=["qdp-kernels/hip"]} with cudarc + qdp-kernels deps `optional`/
  `default-features=false`; `qdp-python` {default=["cuda"], cuda=["qdp-core/cuda"],
  hip=["qdp-core/hip"]}.

## Build commands (gfx90a)
```
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal && . "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm

# kernels + core (HIP). --no-default-features turns off the cuda feature (cudarc).
cd projects/mahout/src/qdp
cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16

# Python extension: build a wheel (does NOT touch torch), then install ONLY the
# extension into the conda env that already has ROCm torch. Do NOT `maturin
# develop` -- it runs `pip install --group dev` which would replace ROCm torch.
conda activate py_3.12
maturin build --features hip --manifest-path qdp/qdp-python/Cargo.toml --out <wheeldir>
pip install --no-deps --force-reinstall <wheeldir>/qumat_qdp-*.whl
```
Follower arches: same commit, `QDP_HIP_ARCH_LIST=gfx1100` / `gfx1151`, no source edit.

## Test commands (gfx90a) -- pick a FREE GCD
This box has 4 GCDs; check `rocm-smi --showuse --showmemuse` and use a free one via
`HIP_VISIBLE_DEVICES=<n>` (the Rust code hardcodes device ordinal 0; HIP_VISIBLE_DEVICES
remaps it). Run serially (single GPU): `--test-threads=1`.
```
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=2
cd projects/mahout/src/qdp
cargo test -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
# Python parity (system interpreter with ROCm torch + installed _qdp wheel):
cd projects/mahout/src && python -m pytest testing qdp/qdp-python/tests -q
```

## Gotchas / lessons (see PORTING_GUIDE changelog for generalized entries)
- `maturin develop` runs `pip install --group dev` (pytest + torch>=2.2,<=2.9 +
  triton) into the active env, which DOWNGRADES/replaces a preinstalled ROCm
  torch with a non-ROCm wheel -> GPU parity tests then see no device. Build a
  wheel with `maturin build` and `pip install --no-deps` it instead.
- hipcc finds none of `<cuda_runtime.h>` / `<cuComplex.h>` / `<vector_types.h>`;
  use forwarding shim headers on a HIP-only include dir.
- hip_complex.h exposes only `hipC*` (no `cuC*`); cuCreal/cuCadd/... are the
  DOUBLE versions in CUDA, so alias them to hipCreal/hipCadd (double), not the
  `f` float set, unless the call site is float-complex.
- `cc::Build` with `compiler("hipcc")` needs `.flag("-x").flag("hip")` so the
  `.cu` are compiled as HIP (the cc crate would otherwise pass C/C++ mode).

## Validation result (lead linux-gfx90a, MI250X, ROCm 7.2.1) -- PASS
All GPU + non-GPU tests RUN (no longer SKIP) and PASS:
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core GPU: gpu_angle 12, gpu_api_workflow 8, gpu_basis 7, gpu_dlpack 9,
  gpu_fidelity 17, gpu_iqp 22, gpu_memory_safety 4, gpu_norm_f32 2,
  gpu_ptr_encoding 64, gpu_validation 8; lib unit tests 77.
- Non-GPU regression: arrow_ipc 5, null_handling 6, numpy 4, parquet 8,
  preprocessing 14, tensorflow_io 9, torch_io 3, types 6. 0 failures total.
- Python parity (testing/qdp + testing/qdp_python, ROCm torch 2.13): 275 passed,
  9 skipped, 0 failed. Skips are legit + pre-existing: 2 multi-GPU,
  1 tensorflow-absent, 1 loader path-validation timing, 5 in test_torch_ref.py
  (the Triton/torch reference path's CUDA-centric `sm_<cap>` arch check vs ROCm
  torch's gfx arch list -- not the native engine; torch compute works on the GCD).
- Backend routing (qdp/qdp-python/tests/test_backend_routing.py): 7 passed.
- Two DLPack device-type tests (Rust gpu_api_workflow.rs::test_dlpack_device_id,
  Python testing/qdp/test_bindings.py::test_dlpack_device) hardcoded kDLCUDA(2);
  made arch-aware to expect kDLROCM(10) on the HIP build (the correct value, and
  what ROCm torch's from_dlpack requires). NOT a port bug.

## Known caveat: release LTO + HIP cdylib (gpuRIR lesson)
The workspace `[profile.release]` has `lto = "fat"`. A RELEASE build of the
qdp-python cdylib under the HIP toolchain produces a 0-byte / bitcode-only `.so`
with no PyInit (import fails "file too short"); the Rust .a/.rlib + all test
binaries are unaffected. For the Python extension, build with `--profile dev`
(no LTO) -- validated working (192MB .so, PyInit__qdp present). The Rust+HIP
link itself is fine (correctly pulls libamdhip64.so.7); only fat-LTO on the
final cdylib breaks. maturin's auditwheel repair also chokes on ROCm libs
("patchelf: missing ELF header"); use `--compatibility linux` to skip it.
Validation installed the dev cdylib directly into the conda env's _qdp pkg.

## Inter-project deps
None. Do not set `depends_on`. (No "Install as a dependency" section: nothing
in MOAT consumes QDP.)
