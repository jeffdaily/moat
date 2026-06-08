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

## Validation 2026-06-04 (linux-gfx90a, validator)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a:sramecc+:xnack-), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork sha: 64cb1c7cb5c3cb560b9407b9a3a1492e6491d813.

Build commands:
```
export GOROOT=/var/lib/jenkins/goroot
export PATH=/var/lib/jenkins/goroot/bin:/opt/rocm/bin:$PATH
export GOPATH=/var/lib/jenkins/go
export CGO_ENABLED=1 GOFLAGS=-mod=mod HIP_VISIBLE_DEVICES=0
cd projects/3/src/cuda && make wrappers CUDA_CC=gfx90a  # code objects already present, no-op
cd .. && go install github.com/mumax/3/...
```

Build result: PASS (binary at /var/lib/jenkins/go/bin/mumax3, 15 MB; deprecated HIP ctx API warnings only, no errors).

Test results:
- `go test ./cuda/...`: 8 tests PASS (TestBuffer, TestReduceSum, TestReduceDot, TestReduceMaxAbs, TestSlice, TestCpy, TestSliceFree, TestSliceHost; plus cufft FFT1D test).
- `go test ./cuda/cu/...`: 12 tests PASS (TestContext, TestDevice, TestMalloc, TestMemAddressRange, TestMemGetInfo, TestMemsetAsync, TestMemset, TestMemcpy, TestMemcpyAsync, TestMemcpyAsyncRegistered, TestModule, TestVersion).
- `go test ./data/... ./httpfs/...`: PASS (non-GPU regression tests).
- `mumax3 -vet *.mx3`: 176/176 scripts OK.
- `mumax3 -paranoid=false -failfast -cache /tmp -http "" -f *.go *.mx3`: 181 OK, 0 failed.

Headline gates:
- standardproblem4 (M.Average() within 1e-3): computed (-0.98461187, 0.12604699, 0.04326887) vs expected (-0.98461241, 0.12604089, 0.04327124) -- PASS.
- standardproblem5 mx/my/mz within 1e-4: mx=-0.23488 (OK), my=-0.09453 (OK), mz=0.02296 (OK) -- PASS.

Verdict: PASS. State -> completed.

## Validation 2026-06-04 (linux-gfx1100, RDNA3 native wave32)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100), ROCm 7.2.1, HIP_VISIBLE_DEVICES=1.
Fork sha: 7ef67aa4fdf6b4234849ef41729fb3a4eeb6e286 (adds TestModule arch fix on top of 64cb1c7).
Wave size: ATTRIBUTE_WARP_SIZE = 32 (confirmed by TestDevice output -- RDNA3 native wave32).
gfx1100 code object: confirmed by boot log "using gfx1100 code object" at runtime.

Build commands:
```
export GOROOT=/var/lib/jenkins/goroot
export PATH=/var/lib/jenkins/goroot/bin:/opt/rocm/bin:$PATH
export GOPATH=/var/lib/jenkins/go
export CGO_ENABLED=1 GOFLAGS=-mod=mod HIP_VISIBLE_DEVICES=1
cd projects/3/src/cuda
make wrappers CUDA_CC=gfx1100   # hipcc --genco --offload-arch=gfx1100 per .cu -> embeds gfx1100 code objects
cd ..
go install github.com/mumax/3/...
```

Build result: PASS (binary at /var/lib/jenkins/go/bin/mumax3, 15 MB; deprecated HIP ctx API warnings only, no errors).

Note: GPU[0] (HIP_VISIBLE_DEVICES=0) had an orphaned KFD context that blocked queue creation (hipMemsetD32 hung indefinitely); GPU[1] (HIP_VISIBLE_DEVICES=1) and GPU[3] were responsive. All tests run on GPU[1].

TestModule fix: module_test.go hardcoded testdata/testmodule_gfx90a.co, causing hipErrorInvalidImage on gfx1100. Fixed to query Device.ArchName() and load testdata/testmodule_<arch>.co; added testdata/testmodule_gfx1100.co built from testmodule.cu with hipcc --genco --offload-arch=gfx1100. Committed as a new commit on top (does not amend the gfx90a-validated sha). On gfx90a the fix loads testmodule_gfx90a.co as before.

Test results:
- `go test ./cuda/...`: 8 tests PASS (TestBuffer, TestReduceSum, TestReduceDot, TestReduceMaxAbs, TestSlice, TestCpy, TestSliceFree, TestSliceHost; plus cufft FFT1D test). Wave32 reduction tests (TestReduceSum, TestReduceDot, TestReduceMaxAbs) all PASS -- the reduce.h fix (all-__syncthreads tree, no unrolled 32-lane tail) is correct on RDNA3 native wave32.
- `go test ./cuda/cu/...`: 12 tests PASS (TestContext, TestDevice, TestMalloc, TestMemAddressRange, TestMemGetInfo, TestMemsetAsync, TestMemset, TestMemcpy, TestMemcpyAsync, TestMemcpyAsyncRegistered, TestModule, TestVersion).
- `go test ./data/... ./httpfs/...`: PASS (non-GPU regression tests).
- `mumax3 -vet *.mx3`: 176/176 scripts OK.
- `mumax3 -paranoid=false -failfast -cache /tmp -http "" -f *.go *.mx3`: 181 OK, 0 failed.

Headline gates:
- standardproblem4 (M.Average() within 1e-3): computed (-0.9846119, 0.1260456, 0.0432690) vs expected (-0.9846124, 0.1260409, 0.0432712) -- PASS. Cross-arch comparison vs gfx90a reference (-0.98461187, 0.12604699, 0.04326887): difference at 7th decimal place (max delta 5e-7) -- no wave32 reduction fault.
- standardproblem5 mx/my/mz within 1e-4: mx=-0.23488 (|diff|=8.6e-5, OK), my=-0.09453 (|diff|=3.0e-6, OK), mz=0.02296 (|diff|=1.8e-6, OK). Cross-arch comparison vs gfx90a reference (mx=-0.23488, my=-0.09453, mz=0.02296): matching to 5 significant figures -- PASS.

atomicFmaxabs: TestReduceMaxAbs PASS on wave32 confirms the atomicCAS loop works correctly on gfx1100.

Verdict: PASS. State -> completed. validated_sha = 7ef67aa4fdf6b4234849ef41729fb3a4eeb6e286.

## Validation 2026-06-04 (linux-gfx90a, revalidate carry-forward)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a), ROCm 7.2.1.
Revalidate trigger: head advanced from gfx90a validated_sha 64cb1c7c to 7ef67aa4 (gfx1151 follower delta-port commit).

Delta: 2 files -- cuda/cu/module_test.go (TestModule arch fix) and cuda/cu/testdata/testmodule_gfx1100.co (gfx1100 test code object). No device source (.cu/.cuh) changed.

Binary-equivalence check:
```
# Extracted all 65 gfx90a .co files at both shas; compared md5sum:
md5sum old/*_gfx90a.co == md5sum new/*_gfx90a.co  # all 65 byte-identical
# Confirmed via clang-offload-bundler unbundle on a sample:
md5sum cellindices_old_gfx90a.elf cellindices_new_gfx90a.elf  # identical
```

Verdict: binary-equiv carry-forward. All 65 gfx90a code objects byte-identical; test-only delta does not affect device ISA. State -> completed at 7ef67aa4fdf6b4234849ef41729fb3a4eeb6e286. No GPU re-run required.

## Validation 2026-06-07 (windows-gfx1201)

Platform: windows-gfx1201, AMD Radeon RX 9070 XT (gfx1201 / RDNA4, wave32), TheRock ROCm 7.14, Windows 11 Pro for Workstations, HIP_VISIBLE_DEVICES=0.

New commit: f6b642d5ce5bc0f832d809953efe0ebfc01d7a83 (on top of 7ef67aa4).
69 files changed: 65 *_wrapper.go (gfx1201 code objects), curand/generator.go, curand/status.go (curand_shim.h), cuda/curand/curand_shim.h (NEW), cuda/cu/testdata/testmodule_gfx1201.co (NEW).

### Windows-specific fixes needed

1. **hiprand.h C++-only incompatibility (TheRock 7.14)**
   - `hiprand.h` -> `rocrand.h` defines `uint4` as an anonymous struct in its C-mode `#else` block, conflicting with `hip_vector_types.h`'s `uint4` definition. Also `typedef _Float16 __half` in generator.go's cgo preamble conflicts with `rocrand.h`'s `typedef unsigned short __half`.
   - Fix: added `cuda/curand/curand_shim.h` that declares only the hiprand symbols mumax3 uses, without including the full hiprand.h chain. Changed cgo preambles in generator.go and status.go from `//typedef _Float16 __half;\n//#include <hiprand/hiprand.h>` to `//#include "curand_shim.h"`.

2. **testdata/testmodule_gfx1201.co required for TestModule**
   - module_test.go probes for `testdata/testmodule_{gcnArchName}.co` and SKIPs if absent.
   - Built with: `hipcc --genco --offload-arch=gfx1201 -include hip/hip_runtime.h testmodule.cu -o testmodule_gfx1201.co`

3. **MinGW import libraries for HIP DLLs**
   - cgo on Windows uses MinGW gcc (Strawberry Perl); cannot link MSVC `.lib` files.
   - Created MinGW import libraries from DLLs:
     ```
     gendef amdhip64.dll; dlltool -d amdhip64.def -l libamdhip64.dll.a
     gendef hipfft.dll;   dlltool -d hipfft.def   -l libhipfft.dll.a
     gendef hiprand.dll;  dlltool -d hiprand.def   -l libhiprand.dll.a
     ```
   - Stored in `C:\Users\Shark44\AppData\Local\Temp\mingw_libs\`.

### Build commands

```
# Build wrappers for gfx1201 (from cuda/ subdir)
cd B:\develop\moat\projects\3\src\cuda
set HIP_PATH=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
make wrappers CUDA_CC=gfx1201

# Build testmodule code object
cd cuda\cu\testdata
hipcc --genco --offload-arch=gfx1201 -include hip/hip_runtime.h testmodule.cu -o testmodule_gfx1201.co

# Install mumax3
set GOROOT=B:\develop\go_root\go
set GOPATH=B:\develop\go_path_gfx1201
set CGO_ENABLED=1
set CGO_CFLAGS=-IB:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel\include -D__HIP_PLATFORM_AMD__
set CGO_LDFLAGS=-LC:\Users\Shark44\AppData\Local\Temp\mingw_libs -lamdhip64 -lhipfft -lhiprand
set GOFLAGS=-mod=mod
set HIP_VISIBLE_DEVICES=0
# PATH must include GOPATH\bin and ROCm\bin so DLLs are found
go install -v ./...
```

### Test results

GPU: ArchName=gfx1201, warpSize=32 (confirmed from TestVersion output).

```
go test -v -count=1 ./cuda/cu/...
# TestContext, TestDevice, TestMalloc, TestMemAddressRange, TestMemGetInfo,
# TestMemsetAsync, TestMemset, TestMemcpy, TestMemcpyAsync,
# TestMemcpyAsyncRegistered, TestModule, TestVersion
12/12 PASS
```

```
go test -v -count=1 ./cuda/...
# TestBuffer, TestReduceSum, TestReduceDot, TestReduceMaxAbs,
# TestSlice, TestCpy, TestSliceFree, TestSliceHost
8/8 PASS
```

```
go test -v -count=1 ./cuda/cufft/...
# TestExampleFFT1D
PASS
```

```
go test -v -count=1 ./data/... ./httpfs/...
# non-GPU regression: all PASS
```

```
# standardproblem4: M.Average within 1e-3 of reference
# result: m=(-0.9846119, 0.12605, 0.04327)  ref=(-0.9846, 0.1260, 0.0435) -- PASS

# standardproblem5: mx/my/mz all within 1e-4 -- PASS
```

### Impact on Linux platforms

This commit regenerated the *_wrapper.go files with gfx1201 code objects (removing gfx90a/gfx1100 objects). The device kernel LOGIC is unchanged; only which arch's objects are embedded differs. Linux platforms (gfx90a, gfx1100) will be flipped to `revalidate` by advance_head. Those validators should do a binary-equivalence check after rebuilding with `make wrappers CUDA_CC=<their-arch>`: if the extracted code objects match (same kernel logic, new hipcc may produce byte-identical or near-identical output), carry forward; otherwise GPU re-run is needed.

For future multi-arch builds (all platforms in one repo state): use `make wrappers CUDA_CC="gfx90a gfx1100 gfx1201"` so all arches are embedded simultaneously.

Verdict: PASS. State -> completed. validated_sha = f6b642d5ce5bc0f832d809953efe0ebfc01d7a83.

## Validation 2026-06-08 (linux-gfx90a, revalidate carry-forward)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a), ROCm 7.2.1.
Revalidate trigger: head advanced from gfx90a validated_sha 7ef67aa4 to f6b642d5 (windows-gfx1201 delta).

Delta (7ef67aa4 -> f6b642d5): 65 *_wrapper.go files regenerated with gfx1201 code objects; cuda/cu/testdata/testmodule_gfx1201.co (new); cuda/curand/curand_shim.h (new, Windows SDK compat shim); cuda/curand/generator.go + status.go (cgo preamble switches from typedef+hiprand.h to curand_shim.h). No .cu, .cuh, or Makefile changes.

Binary-equivalence check:
- Rebuilt gfx90a code objects at both shas using identical toolchain (ROCm 7.2.1, same Makefile/HIPCCFLAGS):
  `make realclean && make wrappers CUDA_CC=gfx90a` at 7ef67aa4 -> 64 *_gfx90a.co files saved.
  `make realclean && make wrappers CUDA_CC=gfx90a` at f6b642d5 -> 64 *_gfx90a.co files saved.
- md5sum comparison: all 64 gfx90a .co files byte-identical across both shas (same .cu sources, same Makefile).
- curand_shim.h host change: compiled successfully on Linux/ROCm 7.2.1 (`go build ./cmd/mumax3` -- no errors, deprecation warnings only, same as before).
- Note: codeobj_diff.py not applicable to Go executables (looks for .so/.hsaco only); manual .co comparison used instead.

Verdict: binary-equiv carry-forward. All 64 gfx90a code objects byte-identical; host-only curand_shim.h change compiles clean. State -> completed at f6b642d5ce5bc0f832d809953efe0ebfc01d7a83. No GPU re-run required.

## Validation 2026-06-08 (linux-gfx1100, revalidate carry-forward)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Revalidate trigger: head advanced from linux-gfx1100 validated_sha 7ef67aa4 to f6b642d5 (windows-gfx1201 delta).

Delta (7ef67aa4 -> f6b642d5): 65 *_wrapper.go files regenerated with gfx1201 code objects; cuda/cu/testdata/testmodule_gfx1201.co (new); cuda/curand/curand_shim.h (new, Windows SDK compat shim); cuda/curand/generator.go + status.go (cgo preamble switches from typedef+hiprand.h to curand_shim.h). No .cu, .cuh, or Makefile changes.

Binary-equivalence check:
- Rebuilt gfx1100 code objects at both shas using identical toolchain (ROCm 7.2.1, same Makefile/HIPCCFLAGS):
  `make realclean && make CUDA_CC=gfx1100` at 7ef67aa4 -> 64 *_gfx1100.co files saved to agent_space/3-gfx1100-revalidate/co-old/.
  `make realclean && make CUDA_CC=gfx1100` at f6b642d5 -> 64 *_gfx1100.co files saved to agent_space/3-gfx1100-revalidate/co-head/.
- cmp -s comparison: all 64 gfx1100 .co files byte-identical across both shas (same .cu sources, same Makefile; diff confirms no .cu/.h/Makefile changes).
- curand_shim.h host change: compiled successfully on Linux/ROCm 7.2.1 (`go build ./cmd/mumax3` -- no errors, deprecation warnings only, same as before).
- Note: codeobj_diff.py not applicable to Go executables (looks for .so/.hsaco only); manual .co comparison used instead.

Verdict: binary-equiv carry-forward. All 64 gfx1100 code objects byte-identical; host-only curand_shim.h change compiles clean. State -> completed at f6b642d5ce5bc0f832d809953efe0ebfc01d7a83. No GPU re-run required.

## Restructure 2026-06-08 (additive HIP backend)

The earlier commits replaced the CUDA path in place (rewrote cuda/cu, cuda/cufft,
cuda/curand to HIP; -lcuda -> -lamdhip64; deleted testmodule.ptx and mode.go).
That breaks every NVIDIA user, so it could not be upstreamed. This restructure
(new commit on top of the validated f6b642d5, NOT an amend) makes the HIP support
ADDITIVE and opt-in: default `go build` still compiles the unchanged upstream
CUDA path; `go build -tags hip` (plus `make BACKEND=hip`) compiles the HIP path.
No device kernel logic changed -- only the file layout, build tags, and the
kernel-build backend selection.

### Build-tag dual backend (the file-layout scheme)

Idiomatic Go build tags select the backend. For every cgo/build file that the
HIP port had rewritten, the upstream CUDA file is restored verbatim and
constrained with `//go:build !hip`, and the HIP version moves to a sibling file
constrained with `//go:build hip`:

- cuda/cu/*: context.go [!hip] + context_hip.go [hip]; cgoflags.go (-lcuda) +
  cgoflags_hip.go (-lamdhip64); result.go (CUresult) + result_hip.go
  (hipError_t); device/execution/function/init/memory/memset/module/peer/
  stream/version the same. Test files: context_test.go [!hip] +
  context_hip_test.go [hip] (the HIP test sibling is named *_hip_test.go, NOT
  *_test_hip.go, so `go test` still recognizes it as a test file). testdata
  carries both testmodule.ptx (CUDA test) and testmodule_<arch>.co (HIP test).
- cuda/cufft/*, cuda/curand/*: same split. cufft/mode.go (uses
  CUFFT_COMPATIBILITY_FFTW_PADDING, no hipFFT equivalent) restored as a
  [!hip]-only file. curand/curand_shim.h stays HIP-only (included only by the
  hip cgo preamble).
- cuda/ build infra: fatbin.go (determineCC/PTX-map, map[int]string) [!hip] +
  fatbin_hip.go (determineArch/gcnArchName, map[string][]byte) [hip]; init.go +
  init_hip.go the same. The 65 generated wrappers: upstream PTX *_wrapper.go
  [!hip] (keyed by compute capability) alongside HIP *_wrapper_hip.go [hip]
  (code-object bytes keyed by gfx arch). The committed *_wrapper_hip.go embed a
  multi-arch set (gfx90a gfx1100 gfx1201) so all validated platforms are served
  by one committed state.
- cuda/cuda2go.go: now backend-aware via -backend=cuda|hip; the cuda branch
  emits PTX *_wrapper.go (//go:build !hip), the hip branch emits code-object
  *_wrapper_hip.go (//go:build hip). It can regenerate either set.
- cuda/Makefile: BACKEND ?= cuda. Default = upstream nvcc `-ptx` per compute
  capability in $CUDA_CC; BACKEND=hip = hipcc `--genco --offload-arch=<arch>`
  per gfx arch in $CUDA_CC. Both branches drive cuda2go with the matching
  -backend flag.
- Host files: engine/engine.go's UNAME var moved into engine/uname.go [!hip]
  (CUDA wording, cu.CUDA_VERSION) and engine/uname_hip.go [hip] (HIP wording,
  cu.HIP_VERSION); engine.go keeps everything else and drops the now-unused
  fmt/runtime/cu imports. cmd/mumax3/main.go's one backend-specific print line
  becomes gpuInfoLine(), defined in cmd/mumax3/gpuinfo.go [!hip] ("cc=%d PTX",
  cuda.UseCC) and gpuinfo_hip.go [hip] ("%s code object", cuda.UseArch). Those
  two files also define goBuildTags ("" / "hip"); runGoFile forwards `-tags hip`
  to the `go run` of a .go input script on the HIP build so the script compiles
  with the same backend (the default build's `go run` args are unchanged).

### #ifdef-guarded device headers (compiled by BOTH nvcc and hipcc)

cuda/reduce.h and cuda/atomicf.h are ONE shared file each, guarded with
`#ifdef __HIP_PLATFORM_AMD__ ... #else <upstream code> #endif`:

- reduce.h: the reduce() macro is a backslash-continued #define, so a directive
  cannot sit inside one body; the whole macro is defined twice -- HIP branch =
  all-__syncthreads tree to s>0 (wave64-correct), CUDA #else branch = the
  upstream unrolled 32-lane warp tail, byte-identical to upstream.
- atomicf.h: HIP branch = atomicCAS loop on the float bits; CUDA #else branch =
  upstream int atomicMax one-liner, byte-identical.
- cuComplex.h stays HIP-only and is reached only via the hipcc -I. include path
  (nvcc finds the real cuComplex.h).

The hipcc compile sets -D__HIP_PLATFORM_AMD__ (HIPCCFLAGS + the cgo CFLAGS), so
the HIP branch is taken on the AMD build and the upstream branch on nvcc.

### CUDA-preservation proof (structural; CUDA path NOT built here -- no nvcc/libcuda)

For every restored CUDA cgo .go file, `git diff 3fe3d41f -- <file>` shows ONLY
the added `//go:build !hip` constraint plus its mandatory trailing blank line
(zero removed lines, CUDA cgo/content otherwise byte-identical to upstream).
Verified programmatically across all 26 restored .go files and all 65 PTX
*_wrapper.go: PASS. The CUDA #else branches of reduce.h/atomicf.h were extracted
and compared byte-for-byte to upstream: identical. The default build cannot be
linked on this AMD host (no CUDA Toolkit): `go build` of cuda/cu stops at
"fatal error: cuda.h: No such file or directory" -- a C-preprocessor error, not
a Go error, confirming the !hip Go structure typechecks and only the missing
CUDA headers block it. Preservation is structural by construction, not claimed
as built/run.

### HIP gfx90a build + test (real MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=0)

Build: `make wrappers BACKEND=hip CUDA_CC="gfx90a gfx1100 gfx1201"` then
`go install -tags hip github.com/mumax/3/...` -> mumax3 binary (15.6 MB) linking
libamdhip64/libhipfft/libhiprand/librocfft. Deprecated-HIP-context-API warnings
only.

- `go test -tags hip ./cuda/`: 8/8 PASS (TestBuffer, TestReduceSum,
  TestReduceDot, TestReduceMaxAbs -- the reduce.h wave64 fix and the atomicCAS
  atomicFmaxabs -- TestSlice, TestCpy, TestSliceFree, TestSliceHost).
- `go test -tags hip ./cuda/cu/...`: 12/12 PASS (module load+launch, memcpy,
  memset, context, version).
- `go test -tags hip ./cuda/cufft/...`: TestExampleFFT1D PASS.
- `go test -tags hip ./data/... ./httpfs/...`: PASS (non-GPU regression). oommf
  fails a pre-existing `go vet` nit (int->string in unmodified ovf2.go); with
  `-vet=off`, the project's documented mode, it has no test files. Not a port
  regression.
- standardproblem4 (M.Average within 1e-3): m=(-0.98461181, 0.12604699,
  0.04326887), all OK. UNAME line shows "HIP-7.2", "using gfx90a code object".
- standardproblem5 (mx/my/mz within 1e-4): mx=-0.23488, my=-0.09453, mz=0.02296,
  all OK.
- `mumax3 -vet *.mx3`: 176/176 OK.
- Broader self-checking scripts: demag2D (9 OK), cubicanisotropy (15 OK),
  anisenergyconservation (10 OK), dmi (2 OK) -- demag FFT, cubic anisotropy,
  energy conservation, DMI kernel paths, zero failures.

### Code-object byte-identity (cross-arch carry-forward)

Built the 64 gfx90a .co at f6b642d5 (its HIP-only Makefile) in a detached
worktree and `cmp`-compared to the 64 gfx90a .co from the restructured tree:
MATCH=64 DIFFER=0. The restructure changed no device code (same .cu, same
HIPCCFLAGS, byte-identical HIP header branches), so the per-arch code objects
are bit-for-bit unchanged. Cross-arch implication: gfx1100 and gfx1201 revalidate
as a binary-equiv carry-forward on their own hosts -- the validator rebuilds its
arch's .co (`make wrappers BACKEND=hip CUDA_CC=<arch>`) and confirms it matches
the previously-validated bytes; no GPU re-run needed for the followers since the
ISA is identical.

### Subtleties for the next agent

- The HIP test-file siblings must end in `_test.go` (named `*_hip_test.go`), not
  `*_test_hip.go`; Go only treats `_test.go` files as tests. The non-test HIP
  siblings are `*_hip.go`; `hip` is not a GOOS/GOARCH so it carries no implicit
  constraint and the explicit `//go:build hip` controls them.
- mumax runs .go input scripts by shelling out to `go run`; that subprocess
  must get `-tags hip` on the HIP build (goBuildTags + runGoFile), or the script
  recompiles the default CUDA path and fails on cuda.h. The cmd-only `-failfast`
  flag is not defined on the script's flag set, so a script run with `-f` should
  omit cmd-batch-only flags (pre-existing upstream behavior, not port-specific).
- To regenerate a single backend's wrappers, the Makefile is timestamp-driven;
  `rm -f cuda/*_wrapper_hip.go` (or *_wrapper.go) before `make wrappers
  BACKEND=...` forces a full regen.
