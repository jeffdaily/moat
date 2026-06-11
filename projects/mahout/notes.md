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

## Review 2026-05-31 (reviewer, linux-gfx90a) -- CHANGES REQUESTED

Verdict: the port is structurally sound and the wave64 fault-class analysis is
correct and verified, but two cheap, genuine defects should land before GPU
validation. Reviewed `git diff ac30a8c...HEAD` (36 files, +953/-115) on
moat-port @ 79a257cd. Findings are problems only.

1. (must-fix, behavior) qdp-core/src/gpu/cuda_ffi.rs:181-189 -- the HIP shim maps
   `cudaMemcpyAsync` to `hipMemcpyWithStream`, which synchronizes the stream and
   blocks the host before returning. The sole call site (pipeline.rs:112, via
   async_copy_to_device) is the dual-stream overlap path whose explicit intent is
   "true async copy ... non-blocking" (pipeline.rs:98, 442-443). A host-blocking
   copy silently serializes the pipeline, defeating the dual-stream overlap on
   AMD -- the very native-engine feature that justifies this port over the
   existing Triton backend (plan.md). The exact 1:1 of cudaMemcpyAsync is
   `hipMemcpyAsync` (present in ROCm 7.x hip_runtime_api.h alongside
   hipMemcpyWithStream). Fix: bind/wrap `hipMemcpyAsync` instead. Correctness is
   not affected (the copy-done event + stream-wait still order things), so the
   validated test results stand; this is a behavior/perf regression of the port's
   headline feature. Re-validate on GPU after the swap since it changes the H2D
   path the pipeline tests exercise.

2. (must-fix, latent safety) qdp-kernels/src/device.rs:318-336 --
   `htod_sync_copy_into` copies `size_of_val(src)` bytes into `dst` without
   checking `dst`'s length, whereas cudarc's `htod_sync_copy_into` asserts
   `src.len() == dst.len()`. Safe today only because the single external caller
   (encoding/basis.rs:146-149) builds `indices_cpu` to exactly `samples_in_chunk`
   (indices_cpu.clear() at basis.rs:108 then one push per chunk element) and
   slices the dst view to the same length. But the dropped invariant turns a
   future length mismatch from a cudarc panic into a silent device-buffer
   overflow (OOB write). Fix: assert dst.len() (DeviceSlice::len) == src.len() in
   the shim to match cudarc semantics.

3. (must-fix, trivial/style) qdp-core/src/gpu/cuda_ffi.rs:55 and :120 -- the new
   section-divider comments use Unicode box-drawing characters (the long bar
   glyphs around "CUDA backend"/"HIP backend"). CLAUDE.md requires ASCII-only in
   new comments. Replace with ASCII (e.g. `// ---- CUDA backend ... ----`).

4. (minor, doc) qdp-kernels/src/kernel_compat.h:19 -- the header comment says
   "Included by every kernel translation unit", but only amplitude.cu includes it
   (confirmed by grep; it is the only kernel with warp intrinsics). Reword to
   "Included by the kernel TUs that use warp intrinsics (amplitude.cu)".

Verified sound (no action): the amplitude.cu wave64 fixes are correct and match
the AutoDock-GPU lessons -- 64-bit QDP_FULL_WARP_MASK keyed on
__HIP_PLATFORM_AMD__ (not wave width) and warp_id = threadIdx.x / warpSize
(== >>5 on wave32, >>6 on wave64), with __shared__[32] still a valid upper bound
(<=16 warps at 1024 threads on wave64); the CUDA path is byte-identical
(0xffffffffu, /32). The Cargo feature gating makes cudarc optional and keeps the
default `cuda` build binding cudarc; every host change is a pure
`cudarc::driver::` -> `crate::gpu_rt::` / `qdp_kernels::device::` import swap with
byte-identical bodies. DLPack tags kDLROCM=10 via NATIVE_GPU_DEVICE_TYPE
(feature-gated; kDLCUDA on CUDA) and the two device-type tests are correctly made
arch-aware (not weakened). The cuda_ffi.rs cuda_rt mod is byte-identical to the
original extern block under `all(cuda, not(hip))`. The metrics.rs download
helpers switch from cudarc's driver-API cuMemcpyDtoH_v2 to the runtime-API
cudaMemcpy (D2H) -- a semantically-equivalent swap in a test/validation helper,
needed to leave cudarc's sys layer; it does touch the CUDA path, so the validator
should confirm the NVIDIA build still passes there. Rule-of-five on the shim
handles is satisfied: CudaSlice (Drop guards ptr!=0) and CudaStream (Drop guards
!is_null), both move-only, no Clone/Copy, no default-constructed handle is
destroyed. Commit hygiene is clean: `[ROCm]` title (<=72), Test Plan present,
Claude disclosed, no noreply trailer, no ghstack, author is the jeffdaily user
identity (no AMD-internal account), single curated commit on moat-port.

## Porter fix 2026-05-31 (changes-requested -> review-passed) -- fork @ 2b0544a

Addressed all 4 review findings; nothing else touched (the wave64 fixes, cudarc
displacement, DLPack, and the CUDA default path are left as-is). Default CUDA
build still binds cudarc (`cargo check -p qdp-core -p qdp-kernels` with default
features type-checks; the QDP_NO_CUDA=1 only skips nvcc kernel compilation).

1. (behavior) cuda_ffi.rs hip_rt: cudaMemcpyAsync now maps to hipMemcpyAsync (the
   exact 1:1 enqueue-and-return, ROCm 7.x hip_runtime_api.h:5037), not
   hipMemcpyWithStream (which synchronizes the stream and blocks the host). The
   old mapping silently serialized the dual-stream H2D overlap pipeline
   (pipeline.rs async_copy_to_device, line 461). Correctness was never affected
   (the copy-done event + wait_for_copy still order copy->compute), so the
   validated results stand; this restores the non-blocking behavior the native
   engine's headline feature depends on.
   - GPU RE-VALIDATION of the async path (HIP_VISIBLE_DEVICES=3, gfx90a):
     * A direct libamdhip64 latency probe on a 256MB pinned H2D copy: hipMemcpyAsync
       returns to the host in ~11 us (transfer 18.64 ms proceeds on the stream);
       hipMemcpyWithStream blocks ~18.63 ms (the full transfer) before returning
       -- 1694x longer. Confirms hipMemcpyAsync is genuinely non-blocking.
     * The dual-stream async-pipeline tests pass with QDP_ENABLE_OVERLAP_TRACKING=1:
       test_amplitude_encoding_async_pipeline, test_angle_encoding_async_pipeline
       (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path
       (gpu_angle_encoding). The H2D-vs-compute overlap timeline records cleanly
       through hipMemcpyAsync (no "invalid resource handle"/event-lifecycle
       errors); a 32MB / 4-chunk multi-chunk pipeline runs and the OverlapTracker
       reports per-chunk overlap, with correct encoder output.
2. (latent safety) device.rs htod_sync_copy_into: added
   `assert_eq!(dst.len(), src.len())` and a `+ DeviceSlice<T>` bound on the
   destination type param, matching cudarc's contract. Both call sites (the
   internal htod_sync_copy at device.rs:300 and encoding/basis.rs:149 via a
   slice_mut view) already satisfy it; this turns a future length mismatch from a
   silent device-buffer OOB write into a clean panic. gpu_memory_safety (4) still
   passes.
3. (style) cuda_ffi.rs:55,120: section dividers changed from Unicode box-drawing
   glyphs to ASCII `// ---- ... ----`.
4. (doc) kernel_compat.h:19: comment corrected from "Included by every kernel
   translation unit" to "Included by the kernel TUs that use warp intrinsics
   (amplitude.cu)".

Regression re-validation (HIP_VISIBLE_DEVICES=3, gfx90a, ROCm 7.2.1) -- all GREEN,
identical to the prior validation:
- qdp-kernels: amplitude 21, angle 10.
- qdp-core lib 77; GPU suites gpu_angle 12, gpu_api_workflow 8, gpu_basis 7,
  gpu_dlpack 9, gpu_fidelity 17, gpu_iqp 22, gpu_memory_safety 4, gpu_norm_f32 2,
  gpu_ptr_encoding 64, gpu_validation 8; non-GPU arrow 5, null 6, numpy 4,
  parquet 8, preprocessing 14, tensorflow 9, torch 3, types 6. 0 failures.
- Python parity (dev-profile HIP wheel, pip --no-deps into the ROCm-torch env;
  testing/qdp + testing/qdp_python + qdp/qdp-python/tests): 301 passed, 12
  skipped, 0 failed. Skips are pre-existing/legit (2 multi-GPU, 1 tensorflow,
  1 loader path-timing, 5 torch_ref sm_-arch check, 2 AmdQdpEngine-not-built,
  1 NVIDIA-ref-absent). NOTE: the full `testing/` tree also has
  testing/qumat/test_amazon_braket_backend.py, which fails COLLECTION (no
  `braket` module) -- that is the qumat quantum-backend layer, orthogonal to the
  QDP native engine and unrelated to this port; scope parity to testing/qdp*.

Fork HEAD: 2b0544a40bcaf60d35539ba8be62cf791e6c0846 (amended single curated
commit, force-with-lease pushed to jeffdaily/mahout @ moat-port).

## Validation 2026-05-31 (validator, linux-gfx90a, MI250X, ROCm 7.2.1) -- PASS

Platform: linux-gfx90a, GCD: HIP_VISIBLE_DEVICES=2 (MI250X gfx90a), ROCm 7.2.1.
Fork: jeffdaily/mahout @ moat-port HEAD 2b0544a40bcaf60d35539ba8be62cf791e6c0846.
Build: `cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16` -- exit 0 (cached, 0.16s).

Rust tests (HIP_VISIBLE_DEVICES=2, --test-threads=1, 10.3s):
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU suites: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total.
Async-pipeline tests confirmed passing: test_amplitude_encoding_async_pipeline,
test_angle_encoding_async_pipeline (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding) -- all via hipMemcpyAsync (non-blocking H2D).

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests, 11.9s):
- 301 passed, 12 skipped, 0 failed.
- Skips: 2 multi-GPU, 1 tensorflow-absent, 1 loader path-timing, 5 torch_ref sm_-arch check
  (Triton/torch CUDA reference path; not native engine), 2 AmdQdpEngine-not-built,
  1 NVIDIA-ref-absent -- all pre-existing/legit.

Transition: review-passed -> completed (validated_sha = 2b0544a).
Followers unblocked: linux-gfx1100, windows-gfx1151 -> port-ready.

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1) -- PASS

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), HIP_VISIBLE_DEVICES=0, ROCm 7.2.1.
Fork: jeffdaily/mahout @ moat-port HEAD 2b0544a40bcaf60d35539ba8be62cf791e6c0846 -- no fork interaction, no source change.

Build commands:
```
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal && . "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm
cd projects/mahout/src/qdp
cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
maturin build --features hip --profile dev --manifest-path qdp/qdp-python/Cargo.toml --out <wheeldir> --compatibility linux
pip install --no-deps --force-reinstall <wheeldir>/qumat_qdp-0.2.0-cp312-cp312-linux_x86_64.whl
```
Both steps exit 0. Wheel imports cleanly (import qumat_qdp ok, QdpEngine/NativeQuantumTensor present).

gfx1100 code-object evidence (llvm-objdump --offloading on libkernels.a):
All 6 kernel TUs target `hipv4-amdgcn-amd-amdhsa--gfx1100`, no gfx90a:
  amplitude.cu, basis.cu, angle.cu, validation.cu, iqp.cu, phase.cu -> gfx1100.

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1):
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU suites: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total. Matches gfx90a baseline exactly.
- Async-pipeline tests pass: test_amplitude_encoding_async_pipeline,
  test_angle_encoding_async_pipeline (gpu_api_workflow),
  test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding).

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 301 passed, 12 skipped, 0 failed. Matches gfx90a baseline exactly.
- Skips: 2 multi-GPU, 1 tensorflow-absent, 1 loader path-timing,
  5 torch_ref sm_-arch (sm_110 on gfx1100 vs sm_-cap list; Triton/torch ref path, not native engine),
  2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.

Wave32 / L2-norm warp-reduction verdict -- CORRECT on gfx1100:
- gpu_norm_f32 (2/2) and amplitude_encode L2-norm tests (10/10 l2_norm* variants in
  test_l2_norm_single_kernel{,_f32}, test_l2_norm_batch_kernel_{f32,odd,stream,zero_*})
  all pass on wave32.
- The arch-unified fix (warp_id = threadIdx.x / warpSize == >>5 on wave32, ==>>6 on wave64)
  places the per-warp partial in the correct shared[warp_id] slot on both widths.
  __shared__ shared[32] holds up to 32 warps; 1024 threads / warpSize=32 = 32 warps
  exactly on gfx1100, with no slot overflow. The QDP_FULL_WARP_MASK=0xffffffffffffffff
  (64-bit) has upper 32 bits zero on wave32, behaving identically to 0xffffffff.
- Determinism: L2-norm tests re-run independently -> 10/10 identical pass.

Transition: port-ready -> completed (validated_sha = 2b0544a40bcaf60d35539ba8be62cf791e6c0846).

## Validation 2026-06-03 (windows-gfx1151) -- delta-port + PASS

Platform: AMD Radeon(TM) 8060S Graphics (gfx1151, RDNA3.5 wave32), Windows 11,
TheRock ROCm 7.14 (pip wheels, rocm-sdk-devel 7.14.0a20260531),
torch 2.12.0+rocm7.14.0a20260531, Rust 1.96.0 (msvc target), maturin 1.13.3.

Fork base: 2b0544a40bcaf60d35539ba8be62cf791e6c0846 (review-passed unified commit).
Delta commit: addf01141f64bf09476ce32274ee61481b57e325 (pushed to moat-port).

### Windows delta required (port-ready -> validated)

The original port gated the entire GPU stack on `target_os = "linux"` as a proxy
for "GPU available." This prevented compilation and test execution of any GPU code
on Windows. A delta commit adds Windows GPU support behind the `hip` feature.

Delta changes (39 files):
1. qdp-core/build.rs, qdp-kernels/build.rs, NEW qdp-python/build.rs: emit
   `cfg(qdp_gpu_platform)` on Linux (always) or Windows+hip. Replaces the
   `target_os = "linux"` proxy with an intent-accurate flag.
2. All source + test files (38 files): mechanical rename
   `target_os = "linux"` -> `qdp_gpu_platform` and
   `not(target_os = "linux")` -> `not(qdp_gpu_platform)`.
3. qdp-kernels/hip_compat/cuda_runtime.h: add `#ifndef M_SQRT1_2` define.
   MSVC <math.h> does not define POSIX math constants; phase.cu uses M_SQRT1_2.
4. qdp-core/src/platform/mod.rs: windows stub now gated
   `all(target_os = "windows", not(qdp_gpu_platform))` to avoid duplicate
   `encode_from_parquet` symbol when qdp_gpu_platform fires on Windows+hip.

Linux/CUDA builds are byte-identical (qdp_gpu_platform == target_os="linux"
on those paths; hip_compat shim is only on the HIP include path).

### Build commands (gfx1151)
```
# Env setup (from agent_space/mahout_build.sh)
VENV=/d/Develop/moat/agent_space/venv-gsplat
ROOT=$($VENV/Scripts/python.exe -m rocm_sdk path --root)
export HIP_DEVICE_LIB_PATH=$ROOT/lib/llvm/amdgcn/bitcode
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1151
export QDP_HIPCC=$ROOT/bin/hipcc.exe
export ROCM_PATH=$ROOT
export PATH=$MSVC_DIR:$CARGO_HOME/bin:$ROOT/bin:$TARGET/debug/deps:$PATH

# Kernels + core (HIP, dev profile)
cargo build --manifest-path qdp/Cargo.toml -p qdp-core -p qdp-kernels \
  --no-default-features --features hip -j6  # -j6: APU power cap

# Python extension wheel (dev profile -- release LTO breaks cdylib on HIP)
maturin build --manifest-path qdp/qdp-python/Cargo.toml \
  --features hip --profile dev --out <wheeldir> \
  --interpreter $VENV/Scripts/python.exe
pip install --no-deps --force-reinstall --ignore-requires-python <wheel>

# Deploy TheRock runtime DLLs next to test binaries (loader prefers exe dir over System32)
cp $ROOT/bin/amdhip64_7.dll $ROOT/bin/amd_comgr.dll $ROOT/bin/rocm_kpack.dll \
   target/debug/deps/
cp $ROOT/bin/amdhip64_7.dll $ROOT/bin/amd_comgr.dll $ROOT/bin/rocm_kpack.dll \
   $VENV/Lib/site-packages/_qdp/

# Windows needs D:/tmp for tests that create temp files with hardcoded /tmp paths
mkdir -p D:/tmp
```

### Test results (gfx1151, dev profile, --test-threads=1)

qdp-kernels (cargo test -p qdp-kernels --no-default-features --features hip):
- amplitude_encode 21/21
- angle_encode 10/10

qdp-core (cargo test -p qdp-core --no-default-features --features hip):
- lib unit tests: 77/77
- gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6
- 0 failures total. Matches gfx90a/gfx1100 baseline exactly.

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 282 passed, 31 skipped, 0 failed.
- Skips vs Linux baseline (12 skips) -- 19 additional:
  - 20 Triton AMD backend tests skip (triton not installed in TheRock pip venv);
    on Linux the conda env had triton installed. Not a GPU regression.
  - 1 test_file_loader_unsupported_extension_raises (Windows path behavior diff)
  - Remaining skips identical to Linux: 2 multi-GPU, 1 tensorflow-absent,
    1 loader path-timing, 5 torch_ref sm_-arch check (CUDA-only reference path),
    2 AmdQdpEngine-not-built. 0 failures.

### Windows-specific notes
- D:/tmp must exist: tests use hardcoded `/tmp/` paths; Windows resolves this to
  D:\tmp (the D: drive, where the cargo workspace lives). Create with mkdir.
- TheRock amdhip64_7.dll must be deployed next to test exes: System32's Adrenalin
  amdhip64_7.dll is broken (undefined blit symbols); TheRock's is self-consistent.
- qdp-python/tests PATH: $ROOT/bin must be on PATH before pytest so _qdp.pyd
  can find amdhip64_7.dll when conftest.py imports it (before torch is loaded).
- -fgpu-rdc is NOT used (no device-link step); hipcc compiles each .cu independently.
  The Windows -fgpu-rdc bundler bug (clang-offload-bundler mmap) is not triggered.

Wave32 verdict: gfx1151 RDNA3.5 wave32 -- all L2-norm/amplitude tests pass,
identical to gfx1100. The arch-unified warp_id = threadIdx.x / warpSize fix
is correct on wave32 (== >>5). QDP_FULL_WARP_MASK 0xffffffffffffffff upper
32 bits zero on wave32, identical to CUDA's 0xffffffff in practice.

Transition: port-ready -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).
linux-gfx90a and linux-gfx1100 -> revalidate (delta touches Rust source;
binary equivalence check expected to confirm no change on Linux paths).

## Validation 2026-06-04 (gfx90a, revalidate, binary-equiv carry-forward)

Platform: linux-gfx90a, GPU: MI250X gfx90a (wave64), ROCm 7.2.1.
Transition: revalidate -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).
Method: binary-equivalence carry-forward (no GPU re-run required).

Delta (2b0544a..addf01141f, 39 files):
- 2 modified build.rs (qdp-core, qdp-kernels): emit `cfg(qdp_gpu_platform)` on Linux (always) or Windows+hip. On Linux `qdp_gpu_platform` is always set -- identical to old `target_os = "linux"` condition.
- 1 new qdp-python/build.rs: same cfg emit for Windows+hip support.
- 35 Rust source + test files: mechanical rename `#[cfg(target_os = "linux")]` -> `#[cfg(qdp_gpu_platform)]` and `#[cfg(not(target_os = "linux"))]` -> `#[cfg(not(qdp_gpu_platform))]`.
- 1 qdp-kernels/hip_compat/cuda_runtime.h: added `#ifndef M_SQRT1_2` guard (MSVC POSIX math constant); this header is only on the HIP include path and only affects Windows MSVC compilation.
- 1 qdp-core/src/platform/mod.rs: windows stub gated `all(target_os = "windows", not(qdp_gpu_platform))` to avoid duplicate symbol on Windows+hip.
None of the 6 .cu kernel source files changed. On Linux, every cfg evaluation is identical: `qdp_gpu_platform` is always true (same as `target_os = "linux"` was), and the M_SQRT1_2 shim is only reached by Windows MSVC, not by Linux/hipcc.

Binary-equivalence check:
- git worktree at validated_sha (2b0544a40b) built into `/tmp/mahout-old-gfx90a-target`; HEAD (addf01141f) built into the default target dir. Both: `QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm --no-default-features --features hip -j 16` -> exit 0.
- Compared libkernels.a (archive of 6 .cu HIP kernel objects) with `python3 utils/codeobj_diff.py`.
- Result: `verdict=identical` -- exported symbols + device ISA identical (0 exports). All 6 gfx90a kernel TUs compile to byte-identical device code objects.
- 256-byte .a size difference is AR metadata (build.rs cfg strings), not GPU code.

Build commands:
```
source "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm
# HEAD build
bash utils/timeit.sh mahout compile -- cargo build \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# validated_sha build (worktree)
cd projects/mahout/src && git worktree add /tmp/mahout-old-gfx90a 2b0544a40bcaf60d35539ba8be62cf791e6c0846
CARGO_TARGET_DIR=/tmp/mahout-old-gfx90a-target cargo build \
  --manifest-path /tmp/mahout-old-gfx90a/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# compare
python3 utils/codeobj_diff.py \
  /tmp/mahout-old-gfx90a-target/debug/build/qdp-kernels-e8e72e39df1ee785/out/libkernels.a \
  projects/mahout/src/qdp/target/debug/build/qdp-kernels-e8e72e39df1ee785/out/libkernels.a
# verdict=identical
```

## Validation 2026-06-04 (gfx1100, revalidate, binary-equiv carry-forward)

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), ROCm 7.2.1.
Transition: revalidate -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).
Method: binary-equivalence carry-forward (no GPU re-run required).

Delta (2b0544a..addf01141f, 39 Rust files + build.rs):
Mechanical rename of `#[cfg(target_os = "linux")]` -> `#[cfg(qdp_gpu_platform)]` across all
GPU-gated code paths. build.rs now emits `cargo::rustc-cfg=qdp_gpu_platform` when
`is_linux || (is_windows && hip_feature)`. On Linux, `is_linux` is always true, so
`qdp_gpu_platform` is always set -- the compiled Linux output is byte-identical to the
old `target_os = "linux"` build. No `.cu` kernel source files changed.

Binary-equivalence check:
- Built at validated_sha (2b0544a40b) with `CARGO_TARGET_DIR` pointing to a separate dir
  (git worktree) and at HEAD (addf01141f) in the default target dir, both with
  `QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm --no-default-features --features hip`.
  Both builds: `cargo build -p qdp-core -p qdp-kernels --manifest-path ... -j 16` -> exit 0.
- Compared libkernels.a (the archive of the 6 .cu HIP kernel objects -- the only artifact
  containing device code) using `python3 utils/codeobj_diff.py <old>/libkernels.a <head>/libkernels.a`.
- Result: `verdict=identical` -- exported symbols + device ISA identical (0 exports).
  All 6 gfx1100 kernel TUs (amplitude.cu, basis.cu, angle.cu, validation.cu, iqp.cu, phase.cu)
  compile to byte-identical device code objects.
- The 94-byte size difference in the .a files is in host-side AR metadata (build.rs cfg
  strings), not in the GPU code objects.

Build commands:
```
source "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm
# HEAD build
cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# validated_sha build (worktree)
git worktree add /tmp/mahout-old 2b0544a40bcaf60d35539ba8be62cf791e6c0846
CARGO_TARGET_DIR=/tmp/mahout-old-target cargo build \
  --manifest-path /tmp/mahout-old/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# compare
python3 utils/codeobj_diff.py \
  /tmp/mahout-old-target/debug/build/qdp-kernels-e8e72e39df1ee785/out/libkernels.a \
  projects/mahout/src/qdp/target/debug/build/qdp-kernels-e8e72e39df1ee785/out/libkernels.a
# verdict=identical
```

## Validation 2026-06-06 (windows-gfx1201, AMD Radeon RX 9070 XT, TheRock ROCm 7.14) -- PASS

Platform: windows-gfx1201, GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4 wave32, warpSize=32),
HIP_VISIBLE_DEVICES=0 (gfx1201 only GPU present; gfx1101 absent from bus),
Windows 11 (26200), TheRock ROCm 7.14.0a20260604,
torch 2.9.1+rocm7.14.0a20260604, Rust 1.96.0 (msvc target), maturin 1.13.3.
Fork: jeffdaily/mahout @ moat-port HEAD addf01141f64bf09476ce32274ee61481b57e325.

Build commands:
```
ROOT=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
MSVC_DIR=/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64
export CARGO_HOME=/b/develop/moat/agent_space/cargo RUSTUP_HOME=/b/develop/moat/agent_space/rustup
export PATH="$CARGO_HOME/bin:$MSVC_DIR:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
# LIB must use Windows-style paths (semicolon-separated) for MSVC link.exe test binary linking
export LIB="$(cygpath -w /c/.../MSVC/14.44.35207/lib/x64);$(cygpath -w .../ucrt/x64);$(cygpath -w .../um/x64)"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1201 QDP_HIPCC="$ROOT/bin/hipcc.exe"
export ROCM_PATH="$ROOT" HIP_VISIBLE_DEVICES=0

# Kernels + core (dev profile, incremental -- only kernel objects rebuilt for gfx1201)
cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip
# -> exit 0, 3.23s. gfx1201 kernel code objects confirmed via strings: gfx1201, gfx1250.

# TheRock runtime DLLs -- already deployed from gfx1101 run (same TheRock build):
# target/debug/deps/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll}

# Python extension wheel for gfx1201
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
maturin build --features hip --profile dev \
  --manifest-path projects/mahout/src/qdp/qdp-python/Cargo.toml \
  --out /b/develop/moat/agent_space/mahout_wheels_gfx1201 \
  --interpreter $VENV/Scripts/python.exe
pip install --no-deps --force-reinstall mahout_wheels_gfx1201/qumat_qdp-0.2.0-cp312-cp312-win_amd64.whl
# Deploy DLLs to _qdp package dir
cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   $VENV/Lib/site-packages/_qdp/
```
Both build steps exit 0. gfx1201 kernel code confirmed (grep: gfx1201/gfx1250 in _qdp.pyd).

Note on LIB env: `cargo build` (library-only) works with POSIX-style LIB paths, but `cargo test`
(produces .exe test binaries) requires Windows-style semicolon-separated LIB for MSVC link.exe.
Use `cygpath -w` to convert. This is a Windows bash-shell quirk not present on the gfx1151 build
(which may have had LIB set in the Windows environment already).

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1):
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU suites: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total. Matches gfx90a/gfx1100/gfx1151/gfx1101 baseline exactly.

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 301 passed, 12 skipped, 0 failed. Matches gfx1101 baseline exactly.
- Skips: 2 multi-GPU, 1 tensorflow-absent, 1 loader path-timing,
  5 torch_ref (sm_120 on gfx1201 vs sm_-cap list; Triton/torch CUDA reference path, not native engine),
  2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.
- Warnings: PyTorch _select_torch_device warns "sm_120 not in arch list" for fallback-path tests;
  the native HIP engine (_qdp) runs on gfx1201 correctly.

Async-pipeline tests pass: test_amplitude_encoding_async_pipeline,
test_angle_encoding_async_pipeline (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path
(gpu_angle_encoding) -- all via hipMemcpyAsync (non-blocking H2D).

Wave32 verdict: gfx1201 RDNA4 wave32 -- all L2-norm/amplitude tests pass, identical to
gfx1100/gfx1101/gfx1151. warp_id = threadIdx.x / warpSize == >>5 on wave32.
QDP_FULL_WARP_MASK 0xffffffffffffffff upper 32 bits zero on wave32, identical to CUDA's 0xffffffff.

Transition: port-ready -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).

## Validation 2026-06-05 (windows-gfx1101, AMD Radeon PRO V710, TheRock ROCm 7.14) -- PASS

Platform: windows-gfx1101, GPU: AMD Radeon PRO V710 (gfx1101, RDNA3 wave32, warpSize=32),
HIP_VISIBLE_DEVICES=0, Windows 11 (26200), TheRock ROCm 7.14.0a20260604,
torch 2.9.1+rocm7.14.0a20260604, Rust 1.96.0 (msvc target), maturin 1.13.3.
Fork: jeffdaily/mahout @ moat-port HEAD addf01141f64bf09476ce32274ee61481b57e325.

Build commands:
```
ROOT=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
MSVC_DIR=/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64
export CARGO_HOME=/b/develop/moat/agent_space/cargo RUSTUP_HOME=/b/develop/moat/agent_space/rustup
export PATH="$CARGO_HOME/bin:$MSVC_DIR:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1101 QDP_HIPCC="$ROOT/bin/hipcc.exe"
export ROCM_PATH="$ROOT" HIP_VISIBLE_DEVICES=0

# Kernels + core (dev profile, no LTO)
cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip

# Deploy TheRock runtime DLLs next to test binaries (loader prefers exe dir over System32)
cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   projects/mahout/src/qdp/target/debug/deps/

# Python extension wheel (dev profile -- release LTO breaks cdylib on HIP)
VENV=B:/develop/TheRock/external-builds/pytorch/.venv
maturin build --features hip --profile dev \
  --manifest-path projects/mahout/src/qdp/qdp-python/Cargo.toml \
  --out /b/develop/moat/agent_space/mahout_wheels_gfx1101 \
  --interpreter $VENV/Scripts/python.exe
pip install --no-deps --force-reinstall mahout_wheels_gfx1101/qumat_qdp-0.2.0-cp312-cp312-win_amd64.whl

# Deploy DLLs to _qdp package dir for Python import
cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   $VENV/Lib/site-packages/_qdp/
```
Both build steps exit 0. Kernels compile for gfx1101 (1 harmless unused-parameter warning in iqp.cu).

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1):
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU suites: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total. Matches gfx90a/gfx1100/gfx1151 baseline exactly.

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 282 passed, 31 skipped, 0 failed. Matches gfx1151 baseline exactly.
- Skips: 20 Triton AMD backend (triton not installed), 2 multi-GPU, 1 tensorflow-absent,
  1 loader path-timing, 5 torch_ref sm_-arch check (CUDA-only reference path),
  2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.

Windows-specific notes (same as gfx1151):
- TheRock amdhip64_7.dll deployed next to test exes; System32's Adrenalin DLL is broken.
- ROCm bin on PATH before pytest so _qdp.pyd can load amdhip64_7.dll at import.
- -j6 cap does NOT apply to this machine (beefy workstation, 64 cores, used default parallelism).

Wave32 verdict: gfx1101 RDNA3 wave32 -- all L2-norm/amplitude tests pass, identical to
gfx1100/gfx1151. warp_id = threadIdx.x / warpSize == >>5 on wave32.

Transition: port-ready -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).

## PR-prep 2026-06-11 (porter, linux-gfx90a) -- squashed to one commit, all 5 carried forward

All 5 platforms were completed at addf01141. PR-prep done; no rebuild (edits are
comment/doc/attribution only, behavior-preserving).

Jargon scrub: only ONE leak in upstream-visible code/comments --
qdp/qdp-kernels/build.rs:39 said "follower platforms (gfx1100, gfx1151)";
reworded to "other AMD targets (gfx1100, gfx1151)", rationale preserved. The two
original commit messages were otherwise clean (commit 1 referenced "the MOAT
CUDA-to-ROCm effort" once, dropped at squash). No other in-house vocabulary
(lead/follower/Strategy A-B/head_sha/validated_sha/revalidate/curated) anywhere
in the diff or messages.

Attribution (AMD copyright line below the Apache ASF header + `Author: Jeff Daily
<jeff.daily@amd.com>`, matching this tree's file-header convention; ASF puts
ownership in NOTICE and uses no per-file author tags, so a header comment line is
the house-style fit):
- NEW files (plain `Copyright`): qdp-core/src/gpu_rt.rs,
  qdp-kernels/src/device.rs, qdp-kernels/src/kernel_compat.h,
  qdp-kernels/hip_compat/{cuComplex.h,cuda_runtime.h,vector_types.h},
  qdp-python/build.rs.
- Substantially extended (`Portions Copyright`): qdp-kernels/build.rs (the +95
  HIP compile branch), qdp-core/src/gpu/cuda_ffi.rs (the hip_rt FFI mod),
  qdp-kernels/src/amplitude.cu (the wave64 warp-id + 64-bit mask fix).
- SKIPPED as trivial (build-flag/cfg/mechanical only, no AMD-authored logic):
  qdp-core/build.rs (+13 cfg-emit), all ~40 mechanical target_os->qdp_gpu_platform
  rename files, the Cargo.toml feature edits.

Docs (REQUIRED step):
- qdp/DEVELOPMENT.md: added "### AMD GPU build (ROCm / HIP)" after the no-CUDA
  sanity block (the project's single from-source build guide), covering the `hip`
  feature, QDP_USE_HIP=1, QDP_HIP_ARCH_LIST, ROCm prereqs (hipcc, AMD HIP
  runtime), and the dev-profile wheel install, in the file's house style.
- qdp/qdp-python/README.md: added a note that the native `_qdp` engine
  (backend="cuda" route) also runs on AMD via the `hip` feature, distinct from
  the Triton backend, pointing to DEVELOPMENT.md. This README defers from-source
  build steps to DEVELOPMENT.md, so no build block was imposed here.

Arch auto-detect determination: build.rs is already env-driven and correct for
upstream -- QDP_HIP_ARCH_LIST is read and comma-split, gfx90a is only a fallback
when unset, nothing hardcoded overrides the env. This mirrors the existing
QDP_CUDA_ARCH_LIST convention. No change made; a build-time rocminfo auto-detect
would diverge from the project's env convention and add fragility (over-
engineering). Left as-is.

Carry-forward note: `advance-head` on the prep delta flipped all platforms to
revalidate because its classifier treats `.rs` as unknown-file-type (cannot prove
comment-only) and raised a kernel_compat.h __LINE__ line-shift false positive.
Manually verified every changed .rs/.cu/.h line is a `//` comment (copyright/
author headers + one comment reword), zero functional code -- so carried all 5
forward with `carry-forward ... source-class` (the behavior-preserving path).

Prep commit (on top of addf01141, before squash):
a8d63e21764900618a344efb09a3a9ee23a019da.
Squashed (tree-identical collapse, force-with-lease pushed to moat-port):
f3f7db33cc9942f5c1a7ffdbe95aea68c85532f5. `squash-carry-forward` carried all 5
platforms (did not refuse -> tree-identical confirmed). pr-ready=True.
Ready for the user's upstream-PR decision (apache/mahout, moat-port -> main).

## Review fixes (apache/mahout#1399) 2026-06-11 (porter, linux-gfx90a)

Maintainer left 4 inline review comments on the open PR. Fixed as ONE follow-up
commit on top of the validated squash (f3f7db33), NOT an amend. Functional HIP
changes, so `advance-head` flipped the 4 follower AMD platforms to revalidate
(correct); the lead linux-gfx90a was re-validated on real gfx90a GPU at the new
head and stays pr-open.

New fork HEAD: 0b5042e705eff3809fa7c40c1383aa6c3adcc602 (force-with-lease pushed
to jeffdaily/mahout @ moat-port; this updates PR #1399).

### Fix 1 (MERGE-BLOCKER): CudaSlice::drop re-binds the owning device
qdp-kernels/src/device.rs, hip mod, `impl Drop for CudaSlice`.
- Before: `drop` called `hipFree(self.raw_ptr())` with no device bind. hipFree
  frees on the calling thread's CURRENT device, so dropping a slice while a
  different device is current (multi-GPU) freed against the wrong device.
- After: `let _ = self._device.bind();` (best-effort -- Drop cannot return an
  error) before `hipFree`, matching the alloc path (`self.bind()?`) and cudarc.
  The CudaSlice already held `_device: Arc<CudaDevice>`; `bind()` was already a
  private method on CudaDevice. No new FFI needed.

### Fix 2: explicit hipMemoryType mapping (no magic 2)
qdp-core/src/gpu/cuda_ffi.rs, hip_rt::cudaPointerGetAttributes (~line 189).
- Before: reinterpreted the CudaPointerAttributes destination directly as a
  hipPointerAttribute_t and let lib.rs:91 compare memory_type against the CUDA
  constant 2. hipMemoryType enum values are NOT guaranteed equal to CUDA's
  across ROCm releases (hip_runtime_api.h note; older HIP had Host=0/Device=1).
- After: reads a real `HipPointerAttributes` (#[repr(C)] mirror of
  hipPointerAttribute_t), compares its `type` against the named
  HIP_MEMORY_TYPE_DEVICE (hipMemoryTypeDevice) / HIP_MEMORY_TYPE_MANAGED
  (hipMemoryTypeManaged) constants, and translates to the CUDA convention
  (CUDA_MEMORY_TYPE_DEVICE/MANAGED) the caller checks. Other values pass through
  verbatim so the "not device memory" branch still fires. CUDA path unchanged.

### Fix 3: build.rs fails loudly on QDP_USE_HIP vs `hip` feature mismatch
qdp-kernels/build.rs: new `check_hip_consistency()` called early in `main()`.
- Before: `hip_requested()` returned true if EITHER the `hip` feature OR
  QDP_USE_HIP was set, so a default `cargo build` (cuda feature) with
  QDP_USE_HIP=1 built AMD kernels (hipcc) against the cudarc host = silent
  mismatch.
- After: if `qdp_use_hip_env()` and `CARGO_FEATURE_HIP` disagree (either
  direction), the build `panic!`s with a clear message. Verified: with
  QDP_USE_HIP=1 and the hip feature off, `cargo build -p qdp-kernels` aborts
  ("QDP_USE_HIP is set but the `hip` Cargo feature is off ..."). A clean CUDA
  build (neither set) and the HIP build (both set) are unaffected.

### Fix 4: fork_default_stream uses a non-blocking stream
qdp-kernels/src/device.rs, hip mod (~line 392) + FFI.
- Before: `hipStreamCreate(&mut stream)` -- a BLOCKING stream that implicitly
  serializes against the NULL/default stream, defeating copy/compute overlap.
- After: `hipStreamCreateWithFlags(&mut stream, HIP_STREAM_NON_BLOCKING)` (flag
  const = 1), matching cudarc. Swapped the `hipStreamCreate` extern decl for
  `hipStreamCreateWithFlags(*mut *mut c_void, u32)` and added the
  HIP_STREAM_NON_BLOCKING const.
- This EXPOSED a pre-existing latent ordering bug (shared with the CUDA path):
  AmplitudeEncoder::encode_batch_from_gpu_ptr_f32_with_stream
  (qdp-core/src/gpu/encodings/amplitude.rs ~line 877) launched the batch-f32
  norm kernel on the CALLER's stream, then read the result back with a
  default-stream `dtoh_sync_copy` WITHOUT first synchronizing the caller's
  stream. The blocking stream masked it; the non-blocking stream let the
  readback race the zero-initialized norm buffer -> "One or more float32 samples
  have zero or invalid norm" (deterministic). Every other batch path launches
  the norm kernel on the NULL stream, and the single-sample stream path already
  calls `sync_cuda_stream(stream, ...)` before the readback. Fix: added the same
  `sync_cuda_stream(stream, "Norm stream synchronize failed (batch f32)")?`
  before the readback. Arch-unified (correct on wave32 + wave64), and correct on
  the CUDA path too (a non-blocking CUDA stream has the identical hazard). This
  is the ONLY change that touches shared (non-hip-gated) code; it is
  behavior-identical for the common blocking-stream case.

### Validation (gfx90a, MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=3)
```
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=3
cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16   # exit 0
cargo test  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```
All Rust tests PASS, 0 failures, matching the prior baseline exactly:
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack
  9/9, gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32
  2/2, gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- gpu_ptr_encoding::test_encode_batch_from_gpu_ptr_f32_with_stream_success: the
  test that failed on the bare fix-4 stream change (before the sync was added)
  now passes 64/64.

Overlap test (fix 4 specifically), QDP_ENABLE_OVERLAP_TRACKING=1:
- test_amplitude_encoding_async_pipeline, test_angle_encoding_async_pipeline
  (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path
  (gpu_angle_encoding) -- all pass. Non-blocking stream preserves the
  copy/compute overlap.

Mismatch guard (fix 3) sanity check: `QDP_USE_HIP=1 cargo build -p qdp-kernels`
(default cuda feature, hip off) panics with the mismatch message; throwaway
target dir, reverted. Default CUDA path still type-checks:
`cargo check -p qdp-core -p qdp-kernels` (default features, QDP_NO_CUDA=1, no
QDP_USE_HIP) exit 0.

### Byte-for-byte-claim finding (report only, for the PR reply)
The phrase "the NVIDIA build is byte-for-byte identical" appears in BOTH
upstream-visible places:
- PR #1399 body, first paragraph.
- The squashed commit message (f3f7db33) first paragraph: "so the NVIDIA build
  is byte-for-byte identical".
The reviewer is right that it is not strictly true. Pre-existing reasons:
metrics.rs swapped cudarc driver `cuMemcpyDtoH_v2` -> runtime `cudaMemcpy`, and
amplitude.cu `>>5` -> `/warpSize` -- both change the CUDA SASS though behavior is
identical. NEW with this follow-up: the fix-4 `sync_cuda_stream` in
encode_batch_from_gpu_ptr_f32_with_stream is NOT hip-gated (it is under
`#[cfg(qdp_gpu_platform)]`), so it also adds a stream-sync to the CUDA build's
codegen -- another behavior-identical-but-not-byte-identical CUDA delta.
Suggested wording for the reply: the CUDA path is behavior-preserving (no
functional change), not literally byte/SASS-identical.

## Review replies posted (apache/mahout#1399) 2026-06-11
PR body reworded "byte-for-byte identical" -> "behavior-preserving (no functional change)" (reviewer rich7420's nit; the CUDA path has behavior-identical-but-SASS-changing deltas: metrics.rs driver->runtime memcpy, kernel >>5->/warpSize, and the new fix-4 stream sync). Posted the overall reply (issuecomment-4682322151) + 4 threaded inline replies (discussion_r3397247837/8095/8302/8492) -- all four review points fixed in 0b5042e. Awaiting rich7420's response / re-review; on merge run set-pr-merged.
