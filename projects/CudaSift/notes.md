# CudaSift notes

Possibly the upstream of the SIFT-GPU code in jeffdaily/colmap (branch rocm-sift-gpu); reference that port for the AMD-fault fixes (texture rule-of-five, OOB index clamps, 256-byte texture pitch).

## Porting 2026-05-30 (gfx90a, ROCm 7.2)

Strategy A (colmap minimal-footprint). Result: clean configure + build of all
targets for gfx90a (configure_exit=0, build_exit=0, 0 compile errors). Binaries
embed `hipv4-amdgcn-amd-amdhsa--gfx90a` code objects and link `libamdhip64`.
GPU tests not run here (validator does that) -- compile+link only.

### Files changed
- `cuda_to_hip.h` (new): the single CUDA->HIP compat header. On
  `USE_HIP || __HIP_PLATFORM_AMD__` includes `<hip/hip_runtime.h>` and `#define`s
  every cudaXxx symbol the project uses to its hip equivalent (errors, device
  mgmt, events/streams, malloc/memcpy/memset, constant-memory symbol copies,
  memcpy kinds, cudaArray + channel-format, texture objects + resource/texture
  descs and their enums). Names taken from pytorch hipify
  `cuda_to_hip_mappings.py`. Also defines two device-intrinsic shims (see fault
  classes). On NVIDIA it is a no-op `#include <cuda_runtime.h>`.
- `hip_compat/cuda_runtime.h`, `hip_compat/cuda.h` (new): tiny HIP-only shims
  (each just `#include "cuda_to_hip.h"`) so the few sources that do
  `#include <cuda_runtime.h>` / `<cuda.h>` directly (tests/test_match.cpp,
  tests/benchmark.cpp, match.cu) resolve on ROCm, which ships no CUDA headers.
  Added to the include path for the HIP build only; NVIDIA uses the real header.
- `CMakeLists.txt`: added `option(USE_HIP ... OFF)`. When ON: `project(... HIP)`,
  `DEVICE_LANG=HIP`, mark the `.cu` TUs and the `LANGUAGE CUDA` `.cpp` files
  `LANGUAGE HIP`, set `HIP_ARCHITECTURES gfx90a` on every target, force-include
  the compat header (`-include .../cuda_to_hip.h` for the HIP language), add the
  `hip_compat/` shim dir, define `USE_HIP`, and swap nvcc flags
  (`--use_fast_math -lineinfo`) for clang (`-ffast-math`). The CUDA path
  (`USE_HIP=OFF`) keeps the original behavior.
- `cudaImage.cu`: `CopyToTexture` -- replaced deprecated `cudaMemcpyToArray`
  (removed from current HIP) with `hipMemcpy2DToArray` under a HIP guard.
  Semantics preserved exactly: the array is `hipMallocArray`'d as
  width=`pitch` elems x height rows and the source is contiguous with row
  stride = pitch, so spitch == width == `pitch*sizeof(float)`, height = rows.
  CUDA branch unchanged.
- `cudaSiftH.cu`: the two `cudaMemcpyToSymbolAsync(symbol, src, bytes)` calls
  now pass explicit `0, cudaMemcpyHostToDevice`. CUDA defaults offset/kind/stream;
  HIP's overload only defaults `stream`, so the 3-arg form failed to resolve.
  The explicit args are valid on CUDA too (BC-safe shared-code edit).
- `matching.cu`: `MatchSiftPoints2` -- the two `sift[min(int, unsigned)]` array
  subscripts now cast to `(int)`. HIP's `min` overload resolution produced a
  non-integer subscript (illegal); the other `min(unsigned, int)` sites that
  assign to an `int` were fine and left as-is.

`cudautils.h` and `cudaSiftD.cu` were deliberately NOT modified -- the warp
helpers and the descriptor-layout / orientation-histogram constants stay in
plain CUDA spelling.

### Fault-class issues encountered and fixes
- Compat-header bootstrap (HIP macro chicken-and-egg): clang in HIP mode
  predefines `__HIP__` but NOT `__HIP_PLATFORM_AMD__` -- that macro is only set
  *by* including the HIP runtime header, so it can't gate the include of that
  same header. Fixed by also passing `-DUSE_HIP` to the compiler (the header's
  first guard term), which is what made the force-include actually translate.
- Warp size (wave64): `cudaSiftD.cu` calls `__any_sync(0xffffffff, pred)`
  directly. On gfx90a `__any_sync` is templated and static-asserts that the
  mask is 64-bit, rejecting the CUDA 32-bit full mask. The kernels that use it
  (`FindPointsMulti*`) launch 32-thread blocks and the surrounding reduction is
  explicitly width-32 (`ShiftUp(.,d) for d<32`, `Shuffle(.,31)`). Mapped
  `__any_sync(mask, pred) -> __any(pred)` in the header: HIP's mask-free `__any`
  polls the active wavefront (exactly the 32 active lanes here), matching the
  original "any active lane" intent. The cudautils.h `ShiftDown/Up/Shuffle`
  helpers take the non-`_sync` `__shfl_*(var, delta, width)` branch under HIP
  (HIP does not define `CUDART_VERSION`); width=32 is honored as a sub-group
  split on wave64, so the 4-partial-sum (`sums[idx/32]`, idx in {0,32,64,96})
  descriptor reduction keeps CUDA semantics. NUMERIC PARITY STILL NEEDS THE GPU
  VALIDATOR -- this analysis says it should match, on-device confirmation pending.
- Missing intrinsic: HIP has no round-toward-zero `__fmul_rz` (used only in the
  RANSAC homography residual test). Mapped to a plain `(a)*(b)` in the header;
  rounding mode is immaterial to an inlier threshold test, and round-to-nearest
  is the faithful default.
- Deprecated array-copy API: `cudaMemcpyToArray` unavailable in HIP -> rewrote
  to `hipMemcpy2DToArray` (see cudaImage.cu above). Texture backing is a
  `cudaArray` (not a pitched 2D bind), so the 256-byte-pitch fault class does
  not apply; the sampler clamps (`cudaAddressModeClamp`) so edge `tex2D`
  fetches at +-cos/+-sin are in-bounds by construction (no OOB clamp needed).
- Texture rule-of-five: `texObj` is default-initialized to 0, created with
  `cudaCreateTextureObject` and unconditionally destroyed with
  `cudaDestroyTextureObject` in the same scope (`ExtractSiftOctave`); the
  `cudaArray t_data` in CudaImage is default-init NULL with a guarded
  `cudaFreeArray` in the dtor. No double-free / uninitialized-handle hazard, so
  no hardening needed beyond what upstream already does.

### Build commands (run from /var/lib/jenkins/moat)
```
utils/timeit.sh CudaSift compile -- cmake -S projects/CudaSift/src \
  -B projects/CudaSift/src/build-hip -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build projects/CudaSift/src/build-hip -j
```
Full clean configure+build ~9s. Targets: cudasift_lib, cudasift, test_extract,
test_match, test_homography, benchmark (+ demo_extract/match/video).

### Remaining warnings (non-fatal)
- ~152 `-Wunused-value`: ignoring `nodiscard` `hipError_t` returns from
  `cudaEvent*` / memcpy calls in TimerGPU and elsewhere. Pre-existing CUDA code
  pattern; CUDA headers don't mark these `nodiscard`, HIP does. Harmless; left
  untouched to keep the diff minimal.
- 4 `-Wliteral-conversion`, 2 `-Wnan-infinity-disabled` (from `-ffast-math`).

### BC note
`USE_HIP=OFF` configure fails on this host only because there is no CUDA
toolkit (`Failed to find nvcc` at the `project(... CUDA)` line) -- identical to
the unmodified upstream on a CUDA-less host, not a regression. The CUDA-path
CMake logic mirrors the original.
