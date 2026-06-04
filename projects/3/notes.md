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

## Review 2026-06-04 (reviewer)

Verdict: review-passed. Driver-API rewrite, build-time code-object path, library swaps, and both device-code fixes are correct. One minor cleanup, no blocker.

Findings (problems only):
- cuda/init.go:36,42-43 -- dead code. `M, m := dev.ComputeCapability()` then `_ = M; _ = m`. The CC concept is fully gone from arch selection (gcnArchName-based now) and M/m are read nowhere else; ComputeCapability() is a pure query. Drop the call and the two blank assignments. Minor; not a correctness issue.

Verified sound (so the next agent need not recheck):
- reduce.h wave64 fix: block size REDUCE_BLOCKSIZE=512 (power of 2); the tree `for(s=blockDim.x/2; s>0; s>>=1)` with `if(tid<s)` reads only sdata[tid+s] where tid+s < 2s <= 512, no OOB; all 512 entries seeded before the tree. The dropped unrolled tail's 32-lane implicit lockstep is invalid on wave64, so removing it is the correct fix; the all-__syncthreads tree is correct on wave32 AND wave64. "Same add order" in the comment is approximate for sum (associativity reordering) but reducesum already accumulates via atomicAdd across blocks (nondeterministic), and fmax is exactly associative; headline gates pass at 1e-3/1e-4. Feeds all 6 reduce kernels (MaxTorque, solver error control) -- the key correctness item, and it is right.
- atomicf.h atomicCAS loop: reducemaxabs seeds initVal=0 and loads fabs(src[i]), so accumulator and candidates are non-negative -> IEEE-754 int bits monotonic, integer-max-over-bits semantics preserved. atomicCAS honored on all ROCm coherence modes, sidestepping the CDNA int-atomicMax coarse-grained-drop class. Correct.
- cu/* driver-API mappings: cuXxx->hipXxx, <cuda.h>-><hip/hip_runtime.h>, -lcuda->-lamdhip64 all correct. result.go enum remap complete; only ERROR_NO_BINARY_FOR_GPU and ERROR_INVALID_IMAGE are referenced (fatbin.go archLoads recover), both retained. Dropped enums (NOT_PERMITTED, TOO_MANY_PEERS, LAUNCH_INCOMPATIBLE_TEXTURING, HARDWARE_STACK_ERROR, ...) referenced nowhere. hipMemsetD32 takes int value -> C.int(value) correct. hipMemAllocHost present in ROCm 7.2.1 header (links).
- MemcpyAsync (generic cuMemcpyAsync) -> hipMemcpyDtoDAsync: sole caller (slice.go:58) is DevicePtr->DevicePtr, semantically correct. Memcpy generic -> hipMemcpyDtoD has no remaining callers (harmless). MemcpyPeer/Async signature changed Context->Device but has zero callers (safe).
- Build path: hipModuleLoadData gets unsafe.Pointer(&image[0]) of a length-delimited []byte (not C.CString); base64 const decoded at init. determineArch returns only a present-and-loadable arch or log.Fatalf, so fatbinLoad's codeobjs[arch] is never empty (&image[0] cannot panic on the normal path). Arch keyed on gcnArchName suffix-stripped via regex `^(gfx[0-9a-f]+)`, trial-load fallback retained. Sound; multi-arch CUDA_CC="gfx90a gfx1100 gfx1151" serves followers with no source change.
- hipFFT/hipRAND swaps: complete and 1:1; hipfftHandle pointer routed through unsafe.Pointer in a handle() helper. rocrand.h `typedef __half half` C-parse fix via `typedef _Float16 __half;` placed in the contiguous cgo preamble (generator.go, status.go), prose rationale split into a separate Go comment block -- correct cgo discipline.
- Deprecated HIP context API: Sync() -> stream0.Synchronize() -> hipStreamSynchronize, never hipCtxSynchronize. context_test made tolerant of hipErrorNotSupported on ApiVersion. Correct.
- Commit hygiene: title "[ROCm] Add AMD GPU support via the HIP driver API" (<=72), Test Plan with literal commands, root-cause explanation, Claude disclosed, no noreply trailer. Author jeff.daily@amd.com is his own public email (not an internal account). No MOAT jargon, no non-ASCII, no em-dash in changed hand files. cuComplex.h shim included by exactly one .cu, resolved via -I.

PORTING_GUIDE promotion (recommend to the validator/next prep phase):
- PROMOTE (general, new build class): runtime-code-object loading for a Go+cgo Driver-API project -- embed `hipcc --genco --offload-arch=<arch>` code-object BYTES base64-keyed by gfx arch string (NOT PTX text keyed by CC, NOT "highest CC" trial-load); hipModuleLoadData takes unsafe.Pointer(&buf[0]) of a length-delimited buffer, never C.CString; select on gcnArchName suffix-stripped with trial-load fallback. This is the defining lesson of the go-cgo-driver-api class and recurs for any driver-API embed scheme.
- PROMOTE (general cgo): hiprand.h `typedef __half half` breaks the cgo C-preamble parse (AMD __half is a C++ struct); fix with `typedef _Float16 __half;` in the contiguous comment block immediately above import "C", prose rationale in a SEPARATE Go comment. Applies to any cgo binding over a roc*/hip* header that re-typedefs __half.
- PROMOTE (general): deprecated HIP context API (hipCtxSynchronize, hipCtxGetApiVersion) returns hipErrorNotSupported on ROCm 7.2.1; route synchronization through stream.Synchronize() (hipStreamSynchronize), not the context API.
- KEEP project-local (mumax-specific): the -include hip/hip_runtime.h + -I. cuComplex shim hipcc flags -- a kernel-build detail tied to mumax's CUDA-spelled-unmodified .cu strategy; not broadly general. Note the -include-builtins trick in the guide as a one-liner only if another --genco port needs it.
