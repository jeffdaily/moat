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
