# OCTproZ notes

## Build instructions (HIP/ROCm)

### Dependencies
- ROCm 7.2.1+
- Qt 5.15+ (qtbase5-dev qtbase5-dev-tools qt5-qmake libqt5opengl5-dev)
- hipFFT (hipfft-dev)
- OpenGL (libgl1-mesa-dev libglu1-mesa-dev)

### Build
```bash
cd projects/OCTproZ/src/octproz_project
mkdir build_hip && cd build_hip
qmake CONFIG+=USE_HIP HIP_ARCHITECTURES=gfx90a ../octproz_project.pro
make -j$(nproc)
```

The executable is at `build_hip/octproz/OCTproZ`.

### Notes
- The port uses a cuda_to_hip.h compat header approach
- hip.pri is the HIP equivalent of cuda.pri
- USE_HIP qmake config selects the HIP build path
- HIP_ARCHITECTURES can be set to gfx90a, gfx1100, etc.

## Port details
- Ported following Strategy A (compat header approach)
- cuFFT -> hipFFT (1:1 mapping)
- CUDA-OpenGL interop -> HIP-OpenGL interop
- No warp-size issues (kernels use block size 128, no warp intrinsics)
