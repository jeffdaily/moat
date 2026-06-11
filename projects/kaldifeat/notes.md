# kaldifeat notes

## Environment (Linux gfx90a lead)
- Host arch: gfx90a (MI250X / CDNA2, wave64), ROCm 7.2.1.
- ROCm PyTorch env (shared with k2): `/opt/conda/envs/py_3.12`
  - torch 2.13.0a0+git8f9a6c8, `torch.version.hip` = 7.2.53211
  - `torch.cuda.is_available()` = True (ROCm), `torch.cuda.device_count()` = 4
  - `torch.utils.hipify.__version__` = 2.0.0 (hipify v2). NOTE: not used by this build (custom CMake extension, not torch CUDAExtension) and no c10 stream sites, so TORCH_HIPIFY_V2 handling is N/A here.
- Build interpreter MUST be `/opt/conda/envs/py_3.12/bin/python` so `cmake/torch.cmake` resolves the ROCm torch (it runs `import torch` via PYTHON_EXECUTABLE).

## Build facts
- Pure CMake (Strategy A). `setup.py` uses a custom `cmake_extension`/`BuildExtension` (cmake/cmake_extension.py) that shells out to cmake+make; NOT torch's CUDAExtension.
- `.cu`/`.cuh` count = 0. No CUDA/c10::cuda/CUDAStream/cublas/cufft/thrust/cub sites in csrc. GPU compute is delegated entirely to libtorch tensor ops; device is a `torch::Device` from a user string ("cuda").
- Links only `${TORCH_LIBRARIES}`.
- Standalone config: `cmake -S src -B src/build -DPYTHON_EXECUTABLE=$PYI -Dkaldifeat_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release`
- Python install: `KALDIFEAT_MAKE_ARGS="-j" $PYI setup.py install`
- No `CMAKE_HIP_ARCHITECTURES` / `enable_language(HIP)` -- no device code is compiled here; GPU arch comes from the linked ROCm torch at runtime, so the build is arch-portable with no source change.

## Tests
- GPU-exercising (run cpu + cuda:0/cuda:1 via tests/utils.get_devices): test_fbank, test_mfcc, test_plp, test_spectrogram, *_options, test_whisper_fbank, test_whisper_v3_fbank. Compare against test_data/*.txt + CPU run via torch.allclose.
- CPU-only regression: C++ gtests feature-window-test, online-feature-test (build with -Dkaldifeat_BUILD_TESTS=ON, run via ctest).

## Install as a dependency
(Porter: verify and finalize the exact commands after the first successful build. k2/torch consumes jeffdaily/kaldifeat @ moat-port via cmake/kaldifeat.cmake.)
Draft recipe:
```
git clone -b moat-port https://github.com/jeffdaily/kaldifeat _deps/kaldifeat/src
cd _deps/kaldifeat/src
KALDIFEAT_MAKE_ARGS="-j" /opt/conda/envs/py_3.12/bin/python setup.py install --prefix _deps/kaldifeat/install
```
Point k2/torch's cmake/kaldifeat.cmake at the installed kaldifeatConfig.cmake instead of re-fetching upstream.
