# kaldifeat notes

## Environment (Linux gfx90a lead)
- Host arch: gfx90a (MI250X / CDNA2, wave64), ROCm 7.2.1.
- ROCm PyTorch env (shared with k2): `/opt/conda/envs/py_3.12`
  - torch 2.13.0a0+git8f9a6c8, `torch.version.hip` = 7.2.53211
  - `torch.cuda.is_available()` = True (ROCm), `torch.cuda.device_count()` = 4 (MI250X)
  - `torch.utils.hipify.__version__` = 2.0.0 (hipify v2). NOTE: not used by this build (custom CMake extension, not torch CUDAExtension) and no c10 stream sites, so TORCH_HIPIFY_V2 handling is N/A here.
- Build interpreter MUST be `/opt/conda/envs/py_3.12/bin/python` so `cmake/torch.cmake` resolves the ROCm torch (it runs `import torch` via PYTHON_EXECUTABLE).

## Build facts
- Pure CMake (Strategy A). `setup.py` uses a custom `cmake_extension`/`BuildExtension` (cmake/cmake_extension.py) that shells out to cmake+make; NOT torch's CUDAExtension. So torch build-time source-hipify never runs.
- `.cu`/`.cuh` count = 0. No CUDA/c10::cuda/CUDAStream/cublas/cufft/thrust/cub sites in csrc. GPU compute is delegated entirely to libtorch tensor ops; device is a `torch::Device` from a user string ("cuda"). Confirmed: on ROCm torch, `cuda:0` maps to "AMD Instinct MI250X / MI250" via the HIP backend, so the existing code runs on the AMD GPU unchanged.

## Source change required (small, build-only)
A pure rebuild was NOT quite enough: the C++ unit-test executables (`-Dkaldifeat_BUILD_TESTS=ON`) needed two ROCm-only build fixes. The library/python module itself needs no source change. All fixes are guarded on `torch.version.hip` (a ROCm torch) so CUDA/CPU builds are untouched.

1. cmake/torch.cmake -- detect a ROCm torch (`torch.version.hip` non-empty) and `find_library` the HIP runtime (`amdhip64`) and `rocprofiler-sdk`. On ROCm, `${TORCH_LIBRARIES}` drags in the static `libkineto.a` whose ROCm profiler references `hipRuntimeGetVersion@hip_4.2` (in `libamdhip64.so`); the shared torch libs do not re-export it, so the test exes fail to link without the HIP runtime on their own link line.
2. kaldifeat/csrc/CMakeLists.txt --
   - Link the HIP runtime into the shared `kaldifeat_core` (PUBLIC) so the test exes inherit it.
   - For the gtest executables, compile the core sources directly via an OBJECT library (`kaldifeat_core_obj`) instead of linking the shared `libkaldifeat_core.so`. Root cause: at process startup the ROCm profiler (`librocprofiler-sdk`, pulled in by `libkineto.a`) recursively `dlopen`s the executable's new shared dependency (`libkaldifeat_core.so`) during its own one-time `pthread_once` static init -> `__gnu_cxx::recursive_init_error`, aborting before any test runs. (gdb backtrace: `__hipRegisterFatBinary` in librocsparse -> `__gxx_personality_v0` throw inside rocprofiler-sdk's pthread_once.) Compiling the core into the exe removes the startup dlopen; the static-core exe then also needs `rocprofiler-sdk` on its link line (the shared `.so` had resolved it transitively).
   - The shared `kaldifeat_core` (used by the python module `_kaldifeat.so` and the install) is unchanged; the python module never hits the recursive-dlopen abort because it is imported into an already-initialized process.

These C++ gtests are CPU-only unit tests (feature-window math, RecyclingVector) and upstream CI does NOT build them (`kaldifeat_BUILD_TESTS` defaults OFF; the upstream CUDA workflow does not override it). They are not the GPU validation gate but are made to pass on ROCm for completeness.

## Version tag cosmetic note
`get_version.py:with_cuda()` keys on `nvcc` being on PATH. On ROCm there is no nvcc, so the wheel/egg local-version tag is `+cpu.torch2.13.0a0` even though the build is GPU-capable. This is cosmetic (packaging name only); the build runs on the AMD GPU regardless. Left as-is to avoid changing the PyPI naming scheme; a downstream consumer should not key on the `+cpu` suffix.

## Build commands (gfx90a, verified)
```
PYI=/opt/conda/envs/py_3.12/bin/python
cmake -S projects/kaldifeat/src -B projects/kaldifeat/src/build \
  -DPYTHON_EXECUTABLE=$PYI -Dkaldifeat_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/kaldifeat/src/build -j
```
Python install (the supported consumer path):
```
cd projects/kaldifeat/src
KALDIFEAT_MAKE_ARGS="-j" /opt/conda/envs/py_3.12/bin/python setup.py install --prefix <prefix>
```
No `CMAKE_HIP_ARCHITECTURES` / `enable_language(HIP)` -- no device code is compiled here; GPU arch comes from the linked ROCm torch at runtime, so the build is arch-portable with no source change (followers just need a matching ROCm torch).

## Tests (verified on gfx90a, GPU exercised)
Pin one GCD: `HIP_VISIBLE_DEVICES=0` (then `device_count()`=1, so `get_devices()` returns `[cpu, cuda:0]`).
- ctest (`-Dkaldifeat_BUILD_TESTS=ON`): 12/12 pass. = 2 C++ gtests (feature-window-test, online-feature-test) + 10 registered python tests (test_fbank, test_fbank_options, test_frame_extraction_options, test_mel_bank_options, test_mfcc, test_mfcc_options, test_plp, test_plp_options, test_spectrogram, test_spectrogram_options). Each python test loops over `get_devices()` and compares the cuda:0 result against the precomputed CPU reference (`test_data/*.txt`) and the CPU run via `torch.allclose`.
- Whisper tests (NOT ctest-registered upstream, run standalone): test_whisper_fbank, test_whisper_v3_fbank -- both PASS. Self-contained (vendored mel-bank headers, no network/model download).
- GPU proof: `torch.cuda.get_device_name(0)` = "AMD Instinct MI250X / MI250"; a fbank computed on cuda:0 returns a tensor on cuda:0 with ~80 MB GPU memory allocated during compute; `torch.version.hip` non-None. Not a silent CPU fallback.
```
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib \
ctest --test-dir build --output-on-failure
```

## Install as a dependency
k2/torch consumes `jeffdaily/kaldifeat @ moat-port` via `cmake/kaldifeat.cmake`. Verified recipe:
```
# 1. Clone the ROCm port branch
git clone -b moat-port https://github.com/jeffdaily/kaldifeat _deps/kaldifeat/src

# 2. Build + install against the ROCm torch env (produces _kaldifeat.so,
#    libkaldifeat_core.so, and kaldifeatConfig.cmake)
cd _deps/kaldifeat/src
KALDIFEAT_MAKE_ARGS="-j" /opt/conda/envs/py_3.12/bin/python setup.py install \
  --prefix $(pwd)/../install
```
This installs (verified paths under `<prefix>/lib/python3.12/site-packages/kaldifeat-*.egg/`):
- `_kaldifeat.cpython-312-x86_64-linux-gnu.so` (python extension)
- `kaldifeat/lib/libkaldifeat_core.so` (C++ core shared lib)
- `kaldifeat/share/cmake/kaldifeat/kaldifeatConfig.cmake` + `kaldifeatConfigVersion.cmake` (CMake package)

For a CMake consumer (e.g. k2/torch), point `find_package(kaldifeat)` at the installed
`kaldifeatConfig.cmake` (`-Dkaldifeat_DIR=<prefix>/.../kaldifeat/share/cmake/kaldifeat`)
instead of re-fetching upstream in `cmake/kaldifeat.cmake` (k2 pins FetchContent at tag
v1.25.4; redirect it to the installed jeffdaily build). For a python consumer, put the egg
dir on `PYTHONPATH` or `pip install .` into the env. The build env MUST be the ROCm torch
(`/opt/conda/envs/py_3.12`); kaldifeat inherits its GPU backend from the linked torch.
</content>
</invoke>
