# 3 (mumax3) notes

GPU micromagnetics, Go + cgo host, CUDA device kernels dispatched at RUNTIME
via the CUDA Driver API. Ported to HIP/ROCm. ext_type = go-cgo-driver-api (a
new build class).

Fork: https://github.com/jeffdaily/3  branch moat-port (default branch master).
Validated: linux-gfx90a (MI250X), ROCm 7.2.1, HIP_VISIBLE_DEVICES=3.

## Architecture (why this is not a clean hipify)

Device dispatch is 100% the CUDA Driver API, not nvcc-`.cu`-linked or `<<<>>>`:
- `cuda/Makefile` compiled each `.cu` to PTX TEXT with `nvcc -ptx`, one per
  compute capability.
- `cuda/cuda2go.go` embedded each PTX string into `<name>_wrapper.go`.
- At runtime `cuda/fatbin.go` -> `cu.ModuleLoadData(ptx).GetFunction(fn)` and
  `cu.LaunchKernel` -> the hand-written cgo driver-API binding in `cuda/cu/`.
- FFT and RNG are separate hand-written cgo bindings over `<cufft.h>` /
  `<curand.h>`.

So the port is Driver-API plumbing + ROCm-library swap + a build-time
code-object retarget, not a kernel hipify.

## The port, in three stages

STAGE 1 (driver-API plumbing + build-time code objects):
- Rewrote `cuda/cu/*.go` from the CUDA Driver API to the HIP Driver API:
  `<cuda.h>` -> `<hip/hip_runtime.h>`, `-lcuda` -> `-lamdhip64`, every
  `C.cuXxx` -> `C.hipXxx`, the CUresult enum -> hipError_t names, the
  CUdevice_attribute enum -> hipDeviceAttribute_t, the CUfunction_attribute
  enum -> HIP_FUNC_ATTRIBUTE_*. Launch is `hipModuleLaunchKernel` (driver-API
  form), not `hipLaunchKernel`.
- Build-time code objects (plan Option 1): `cuda/Makefile` now compiles each
  `.cu` with `hipcc --genco --offload-arch=<arch>` to a relocatable HIP code
  object (`.co`), and `cuda/cuda2go.go` embeds the code-object BYTES
  (base64-encoded, since they contain NUL) keyed by gfx arch STRING.
  `hipModuleLoadData` accepts a `--genco` image directly (no hiprtc needed).
- `ModuleLoadData` API changed from `string` (NUL-terminated C string) to
  `[]byte` (length-delimited) because code objects are binary.
- `cuda/fatbin.go` arch selection: replaced `determineCC`/`ccIsOK` (highest CC
  trial-load) with `determineArch`, which reads `hipDeviceProp_t.gcnArchName`
  via `cu.Device.ArchName()`, strips the feature suffix
  (`gfx90a:sramecc+:xnack-` -> `gfx90a`), and trial-loads as a fallback.

STAGE 2 (ROCm library swaps, near-mechanical, cu* -> hip* is 1:1):
- `cuda/cufft/*` -> hipFFT: `<cufft.h>` -> `<hipfft/hipfft.h>`,
  `-lcufft` -> `-lhipfft`, `cufftXxx` -> `hipfftXxx`,
  `cudaStream_t` -> `hipStream_t`. Note: on AMD `hipfftHandle` is a POINTER
  (`struct hipfftHandle_t*`), not the `int` that `cufftHandle` is, so the Go
  `Handle` (uintptr) conversions go through `unsafe.Pointer`. Removed the dead
  `mode.go` (CUFFT_COMPATIBILITY_FFTW_PADDING has no hipFFT equivalent and was
  unused).
- `cuda/curand/*` -> hipRAND: `<curand.h>` -> `<hiprand/hiprand.h>`,
  `-lcurand` -> `-lhiprand`, `curandXxx` -> `hiprandXxx`. Only the float
  GenerateNormal path is used (thermal term).

STAGE 3 (device-code correctness):
- `cuda/reduce.h` wave64 fix (PRIMARY): the reduce macro dropped to an
  unrolled warp-synchronous tail (`volatile float* smem; smem[tid] += [tid+32]
  ... +1` with no __syncthreads) once the __syncthreads tree reached 32,
  relying on 32-lane implicit lockstep. Invalid on gfx90a wave64. Fix: let the
  __syncthreads-ed tree run all the way down to `s>0` (same add order,
  block-wide barrier, correct on wave32 AND wave64). Feeds all 6 reduce
  kernels (MaxTorque, solver error control).
- `cuda/atomicf.h`: atomicFmaxabs used int `atomicMax` on float-as-int. Replaced
  with an atomicCAS loop on the float bits (values are non-negative so the
  IEEE-754 bit order is monotonic). atomicCAS is honored on all ROCm memory
  modes, sidestepping the CDNA coarse-grained int-atomicMax-silently-dropped
  class regardless of allocation coherence.

## Build recipe (linux-gfx90a)

Go 1.22.4 is required (go.mod) and is NOT in apt at that patch level; installed
the official tarball to /var/lib/jenkins/goroot.

```
export GOROOT=/var/lib/jenkins/goroot
export PATH=/var/lib/jenkins/goroot/bin:/opt/rocm/bin:$PATH
export GOPATH=/var/lib/jenkins/go
export CGO_ENABLED=1 GOFLAGS=-mod=mod
cd projects/3/src/cuda
make wrappers CUDA_CC=gfx90a          # hipcc --genco per arch -> .co -> cuda2go embeds -> wrappers
cd ..
go install github.com/mumax/3/...      # links mumax3 + tools into $GOPATH/bin
```

Multi-arch (followers, no source change): `make wrappers CUDA_CC="gfx90a gfx1100 gfx1151"`.
The wrapper map is keyed by gfx string and the loader picks by gcnArchName, so
one build serves all archs and a follower only needs to add its arch to the
list.

Kernel compile flags (cuda/Makefile HIPCCFLAGS):
- `-include hip/hip_runtime.h`: the `.cu` rely on nvcc's implicit builtins
  (blockIdx, threadIdx, atomicAdd, float3, ...); hipcc needs them included.
  Sources stay CUDA-spelled and unmodified.
- `-I.` so `#include <cuComplex.h>` resolves to the local compat shim
  `cuda/cuComplex.h` (maps cuComplex names to hipComplex; one .cu uses it).
- `-Wno-bitwise-instead-of-logical`: clang errors on the intentional bitwise-or
  of bools in the topological-charge / magnetoelastic stencils that nvcc
  accepts; -Werror kept for everything else.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=3)

- standardproblem4.go: M.Average() = (-0.98461, 0.12605, 0.04327), all 3
  components OK within TOL=1e-3 (the headline correctness gate and the
  followers' cross-arch diff target).
- standardproblem5.mx3: mx/my/mz all OK within 1e-4.
- `go test ./cuda/...`: PASS (reduce_test directly exercises the reduce.h fix;
  buffer/slice tests run on GPU).
- `go test ./cuda/cu/...`: PASS (module load+launch, memcpy, memset, context).
- Non-GPU regression (`go test ./data/... ./httpfs/...`): PASS.
- `mumax3 -vet *.mx3`: all 176 scripts OK.
- Subset of self-checking .mx3 (demag FFT, exchange, DMI, anisotropy energy
  conservation, cubic anisotropy, thermal/curand): all OK.

## Gotchas / fault classes

- NEW PATTERN: go-cgo-driver-api / runtime-code-object loading. The CUDA path
  embedded PTX TEXT (a NUL-terminated C string) keyed by integer CC. HIP code
  objects from `hipcc --genco` are BINARY (contain NUL), so (a) the embed must
  be length-delimited (base64 const decoded at init into `map[string][]byte`),
  not a C string, and (b) `hipModuleLoadData` takes `unsafe.Pointer(&buf[0])`,
  not `C.CString`. Key the map by gfx arch string and select on gcnArchName,
  not by trial-loading the "highest CC" (no CC concept on AMD).
- `hipCtxSynchronize` and `hipCtxGetApiVersion` (the deprecated HIP context
  API) return `hipErrorNotSupported` on ROCm 7.2.1. mumax uses
  `stream0.Synchronize()` everywhere, so this never bites the runtime; only an
  upstream cu-package test called ApiVersion (made tolerant). Do NOT route
  Sync() through CtxSynchronize on ROCm.
- LockOSThread + hipCtxCreate (legacy/deprecated context API) works fine under
  cgo for the single-GPU stream-0 path; no need for
  hipDevicePrimaryCtxRetain. The deprecation is warnings-only.
- hiprand.h -> rocrand.h does `typedef __half half`, and the AMD `__half` is a
  C++ struct, so the cgo C-preamble parse fails with "unknown type name
  '__half'". Fix: alias `typedef _Float16 __half;` in the cgo preamble before
  `#include <hiprand/hiprand.h>` (only the float generator is used). IMPORTANT
  cgo subtlety: this typedef must be in the contiguous `//`-comment block
  immediately above `import "C"`; put any prose rationale as a SEPARATE Go
  comment with a blank line, or the prose (and its apostrophes) get parsed as
  C and break the build.
- hipFFT R2C/C2R/C2C layout and scaling match cuFFT for the batched real
  transforms mumax uses; the demag self-test passed with no normalization
  change.
- The int atomicMax-on-float coarse-grained-drop class: mumax's atomicFmaxabs
  targets hipMalloc device memory (likely fine), but the atomicCAS-loop
  rewrite is correct regardless and removes the risk entirely.

## Install as a dependency

Not a library other MOAT targets depend on; no dependents.
