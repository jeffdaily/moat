# aihwkit notes

IBM Analog Hardware Acceleration Kit. PyTorch library with a standalone
CMake/scikit-build CUDA backend (RPUCuda) exposed via a pybind11 module
(`rpu_base`). Strategy A HIP port: one compat header plus a USE_HIP CMake
option; Torch is found for headers/linkage only (NOT torch cpp_extension).

## Environment (linux-gfx90a)
- ROCm 7.2.1 at /opt/rocm; hipBLAS, hipRAND, hipCUB present.
- ROCm PyTorch in conda env py_3.12: torch 2.13.0a0, torch.version.hip 7.2.x,
  device AMD Instinct MI250X / MI250 (gfx90a, wave64).
- This newer Torch requires C++20 (its headers use the `requires` keyword), so
  the build sets RPU_CXX_STANDARD=20. Default stays 17 to match older Torch.

## The port
- New: src/rpucuda/cuda/cuda_to_hip.h -- the only HIP-aware header. Included
  once from cuda_util.h before the CUDA runtime / cuBLAS / cuRAND block,
  guarded by USE_HIP (CUDA build byte-identical). Pulls <cstring>/<cstdlib>
  before <hip/hip_runtime.h>, then aliases the cuda*/cublas*/curand* spellings
  the project uses to hip*/hipblas*/hiprand*.
- New: cmake/dependencies_hip.cmake -- enable_language(HIP),
  find_package(hip/hipblas/hiprand/hipcub), defines RPU_USE_CUDA and USE_HIP.
- CMakeLists.txt: option(USE_HIP); the GPU library/test/pybind branches now
  fire on USE_CUDA OR USE_HIP. Under HIP the .cu sources get LANGUAGE HIP, link
  roc::hipblas hip::hiprand hip::hipcub, HIP_ARCHITECTURES from
  CMAKE_HIP_ARCHITECTURES (default gfx90a when unset; no literal hardcode), IPO
  forced OFF. RPU_CXX_STANDARD cache var replaces the hardcoded CXX_STANDARD 17.
- rpu_cub.h: under USE_HIP include <hipcub/hipcub.hpp> and set
  RPU_CUB_NS_QUALIFIER to `hipcub::` (hipCUB ignores CUB_NS_PREFIX wrapping).
- src/aihwkit/simulator/CMakeLists.txt: the pybind TUs get LANGUAGE HIP under
  HIP (they inherit RPU_GPU's --offload-arch usage requirement, which only the
  HIP/clang driver understands), and IPO OFF on the module.
- rpu_base_tiles_cuda.cpp: under USE_HIP skip the raw `cuda.h` include and use
  <c10/hip/HIPStream.h> + a `using c10::cuda::getCurrentCUDAStream` in the
  at::cuda namespace, instead of <ATen/cuda/CUDAContext.h> which transitively
  pulls hipSOLVER/hipSPARSE headers not in this build.
- rpu_linearstep_device.h: declare the ctor as `SoftBoundsRPUDeviceMetaParameter()`
  not `...<T>()` -- a template-id constructor name is ill-formed under C++20
  (latent upstream bug, exposed only because the newer Torch forces C++20).

### Library swaps (mechanical, via the compat header)
- cuBLAS -> hipBLAS: Sgemm/Dgemm/Hgemm, Sgemv/Dgemv, Sger/Dger, Sscal/Dscal,
  Scopy/Dcopy, Snrm2/Dnrm2; handle + SetStream + Get/SetPointerMode (HOST and
  DEVICE pointer modes on the host handle -- hipBLAS supports both); op N/T and
  status enums. CUBLAS_STATUS_LICENSE_ERROR has no hipBLAS analogue -> mapped to
  HIPBLAS_STATUS_UNKNOWN so the CUBLAS_CALL switch still compiles. The
  device-side cuBLAS path (kernelCublasCreateDevice/getDeviceHandle, device API)
  is fully behind RPU_WITH_CUBLAS_DEVICE (OFF by default) and is never reached;
  hipBLAS has no device API so leaving it gated off is correct.
- cuRAND -> hipRAND: host generator (curandGenerator_t + create/destroy/
  generate/seed/SetStream) and device per-thread state (curand_init/normal/
  uniform). GOTCHA: in CUDA, curandState, curandState_t and curandStateXORWOW
  are the SAME XORWOW type; hipRAND declares hiprandState and
  hiprandStateXORWOW_t as DISTINCT structs. The project mixes the spellings
  (CudaArray<curandState_t> at call sites, explicit instantiation of
  CudaArray<curandStateXORWOW>), so all three are aliased to one type
  (hiprandState_t). Mapping them to different hip types caused a missing
  template instantiation -> ImportError: undefined symbol
  CudaArray<hiprandState>.
- CUB -> hipCUB: DeviceReduce/DeviceScan/DeviceSegmentedReduce + BlockScan
  (block-width templated, adapts to wave64 automatically). The single BlockScan
  TempStorage is used once (no back-to-back reuse) so no wave64 TempStorage race.

### bit_line_maker.cu warp-size fix (the risk path -- dietgpu-class)
The stochastic pulse train is a warp-size-coupled SERIALIZED bit format:
__ballot_sync(0xFFFFFFFF, ...) packed into 32-bit words (nK32 = (Kplus1+31)/32),
laneId = threadIdx.x & 0x1f; the pulsed-weight-updater kernels (pwu_kernel.h)
read it back word-by-word with >>5 / &0x1f / __popc. The on-device word layout
must stay byte-identical regardless of wavefront width. HIP 7.2 provides real
__ballot_sync/__shfl_sync that static_assert a 64-bit mask, and a native
__ballot on wave64 spans all 64 lanes -- which would merge the two independent
32-lane logical warps a wave64 wavefront holds into one 64-bit value and corrupt
the format. Fix (in cuda_to_hip.h): redefine __ballot_sync/__shfl*_sync as
width-32 logical-warp ops. __ballot_sync uses a helper that takes the full
__ballot() and shifts down by (__lane_id() & 0x20) so each 32-lane subgroup
extracts ONLY its own 32 lanes as a 32-bit word (wave32: shift always 0; wave64:
lanes 32-63 select the high word). The shuffles use width 32. The producer's
per-subgroup leader (laneId==0 -> physical lanes 0 and 32 on wave64) and
per-subgroup sourceId (threadIdx.x >> 5) are already correct in the surrounding
code, so each logical warp writes its own 32-bit word exactly as CUDA does. The
word width stays 32 (NOT widened to 64); the consumer is unchanged. The launch
geometry already assumes 32-lane logical warps (numwarpsperblock = threads/32).

Byte-identity evidence: the multi-arch fat-binary check
(-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100") emits BOTH gfx90a (wave64) and
gfx1100 (wave32) code objects from the one wave-agnostic source for
bit_line_maker.cu (llvm-objdump --offloading confirms both). The pulsed-update
tile tests passing on wave64 (test_specific_tiles, InferenceCuda program_weights)
prove the packed stream is read back correctly on wave64. A literal byte-for-byte
diff against a CUDA build cannot be run here (no NVIDIA GPU on the host); the
format-preservation is structural (32-bit-per-32-lane-subgroup, consumer
unchanged).

## Build recipe (gfx90a)
Direct CMake (bring-up / fastest device-code iteration):
```
cd projects/aihwkit/src
TORCH_CMAKE=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/share/cmake
cmake -S . -B build_hip -GNinja \
  -DUSE_HIP=ON -DUSE_CUDA=OFF -DRPU_CXX_STANDARD=20 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;/opt/rocm" \
  -DRPU_BLAS=OpenBLAS -DBUILD_TEST=OFF -DRPU_USE_TORCH_BUFFERS=OFF
cmake --build build_hip -j16
```
Python package in-place (what the pytest suite imports):
```
cd projects/aihwkit/src
USE_HIP=ON USE_CUDA=0 python setup.py build_ext -j16 --inplace \
  -DUSE_HIP=ON -DUSE_CUDA=OFF -DRPU_CXX_STANDARD=20 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;/opt/rocm" \
  -DRPU_BLAS=OpenBLAS -DRPU_USE_TORCH_BUFFERS=OFF -DCMAKE_BUILD_TYPE=Release
```
This drops rpu_base.cpython-*.so into src/aihwkit/simulator/. Run tests with
PYTHONPATH=src (the editable package is not pip-installed here).

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=0)
CPU-tile-vs-CUDA-tile comparisons are tolerance-based (assertTensorAlmostEqual
decimal=4); test_specific_tiles uses no fixed cuRAND bitstream, so it is a valid
cross-arch gate. Results:
- tests/test_specific_tiles.py: 18 passed (the bit_line_maker + pulsed-update
  warp-size path; 9 cuda-parametrized cases). CRITICAL gate -- PASS.
- tests/test_simulator_tiles.py + tests/test_bindings_tiles.py: 530 passed,
  1 failed, 56 skipped. The single failure was TileTest_Inference (the CPU
  tile, not CUDA) test_program_weights -- a stochastic 5%-tolerance
  weight-programming convergence test sensitive to global RNG ordering. It
  passes 15/15 in isolation; the CUDA variant InferenceCuda passes 8/8. Not a
  port regression (pre-existing stochastic-test fragility on the CPU path).
- tests/test_torch_tiles.py + tests/test_inference_tiles.py: 406 passed, 55
  skipped.
- tests/test_layers_linear.py + tests/test_layers_convolution.py: 566 passed,
  216 skipped.
Multi-arch code-object check: fat binary for "gfx90a;gfx1100" builds and emits
both code objects for bit_line_maker.cu (the warp-size TU).

Pin a GCD with HIP_VISIBLE_DEVICES=0 (host has 4 GCDs shared with another
session).

## Known follow-ups (not blocking the lead port)
- FP16/bfloat16 tile support (RPU_USE_FP16) is experimental and OFF by default;
  left unported (hip_fp16/hip_bfloat16 is a future delta if a half tile test
  needs it).
- RPU_USE_TORCH_BUFFERS=ON (RPU_TORCH_CUDA_BUFFERS) was OFF for this validation
  to keep bring-up focused; the CUDA-only --expt-relaxed-constexpr/-Xcudafe
  flags are now gated to NOT USE_HIP, so turning torch buffers ON under HIP is a
  straightforward follow-up to verify.
- BUILD_EXTENSION (aihwkit_extension) GPU ops left on the CUDA path only
  (BUILD_EXTENSION OFF by default).

## Install as a dependency
Not a base library for other MOAT projects; no dependents.
