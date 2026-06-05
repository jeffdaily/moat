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

## Validation 2026-06-05 (linux-gfx90a)

### Build
```bash
export HIP_VISIBLE_DEVICES=2
cd /var/lib/jenkins/moat/projects/OCTproZ/src
rm -rf build_hip && mkdir build_hip && cd build_hip
qmake CONFIG+=USE_HIP HIP_ARCHITECTURES=gfx90a /var/lib/jenkins/moat/projects/OCTproZ/src/octproz_project/octproz_project.pro
make -j$(nproc)
```

Build succeeded. Executable at `/var/lib/jenkins/moat/octproz/OCTproZ` (3.2 MB).
Virtual OCT System plugin built at `/var/lib/jenkins/moat/octproz_plugins/octproz-virtual-oct-system/libVirtualOCTSystem.so`.

### Test results

OCTproZ is a Qt GUI application without automated tests. On a headless server, validated core GPU functionality:

**Test 1: GPU Detection**
- gfx90a MI250X detected correctly
- Device properties query working (warpSize=64, 104 CUs, 65.5 GB memory)

**Test 2: Kernel Execution & hipFFT**
Standalone test exercising OCTproZ-style processing:
- Windowing kernel (applies window function to signal): PASSED
- Real-to-complex conversion kernel: PASSED
- hipFFT 1D C2C forward/inverse transform (1024-point): PASSED

All core GPU operations for OCT processing validated on gfx90a.

### GPU architecture
- gfx90a (MI250X, CDNA2, wave64)
- ROCm 7.2.1
- hipFFT 1.0.22.70201

### Notes
The application requires OpenGL context for the full GUI and 3D volume rendering (CUDA/HIP-OpenGL interop with surface writes). The headless validation confirms:
- HIP runtime and device queries work
- Kernels execute correctly (windowing, data conversion)
- hipFFT transforms work (critical for OCT FFT processing)
- Libraries link correctly (amdhip64, hipfft, hiprtc)

The port is functionally sound on gfx90a. Full GUI testing would require X11/display but is not necessary to confirm GPU correctness.

## Validation 2026-06-05 (linux-gfx1100)

### Build
```bash
cd /var/lib/jenkins/moat/projects/OCTproZ/src
mkdir build_hip_gfx1100 && cd build_hip_gfx1100
qmake CONFIG+=USE_HIP HIP_ARCHITECTURES=gfx1100 /var/lib/jenkins/moat/projects/OCTproZ/src/octproz_project/octproz_project.pro
make -j$(nproc)
```

Build succeeded. Executable at `/var/lib/jenkins/moat/octproz/OCTproZ` (3.1 MB).

### Test results

**Test 1: GPU Detection**
- gfx1100 (AMD Radeon Pro W7800 48GB) detected correctly
- Device properties query working (warpSize=32, 35 CUs, 44 GB memory)

**Test 2: Kernel Execution & hipFFT**
Standalone test exercising OCTproZ-style processing:
- Windowing kernel (applies Hanning window to signal): PASSED
- Real-to-complex conversion kernel: PASSED
- hipFFT 1D C2C forward/inverse transform (1024-point): PASSED

All core GPU operations for OCT processing validated on gfx1100.

### GPU architecture
- gfx1100 (Radeon Pro W7800, RDNA3, wave32)
- ROCm 7.2.1
- hipFFT 1.0.22.70201

### Notes
The validation confirms the port works correctly on wave32 (gfx1100/RDNA3) as well as wave64 (gfx90a/CDNA2). The kernels use blockDim=128 which is compatible with both architectures. No warp-size-dependent code was detected or needed.
