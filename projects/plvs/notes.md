# plvs notes

## Port Status: BLOCKED

**Blocked Reason**: Missing required dependencies. PLVS requires OpenCV 4.x with CUDA/HIP support and many third-party libraries (Pangolin, DBoW2, g2o, PCL, etc.) that need to be built first. The host does not have OpenCV installed.

### Port Progress
Strategy A CMake port structure is in place. Core changes verified to compile (Cuda.cu compiles successfully with HIP).

### Files Modified
- `CMakeLists.txt` - Added USE_HIP option and HIP build support
- `src/cuda/cuda_to_hip.h` - New compat header with CUDA->HIP aliases
- `src/cuda/Allocator_gpu.cu` - Include cuda_to_hip.h
- `src/cuda/Cuda.cu` - Include cuda_to_hip.h (VERIFIED: compiles with HIP)
- `src/cuda/Orb_gpu.cu` - Include cuda_to_hip.h
- `src/cuda/Fast_gpu.cu` - Include cuda_to_hip.h
- `include/cuda/helper_cuda.h` - Added HIP-specific implementation section
- `Thirdparty/libsgm/` - Multiple files updated for HIP support
- `Thirdparty/libsgm/src/utility.hpp` - Wave64-aware WARP_SIZE
- `Thirdparty/libelas-gpu/` - cuda_to_hip.h compat header added

### Blocking Issues
1. **OpenCV not installed**: The main CUDA files use OpenCV CUDA headers (`opencv2/core/cuda.hpp`, `opencv2/core/cuda/reduce.hpp`). Need OpenCV 4.x built with HIP support.
2. **Third-party libraries not built**: Pangolin, DBoW2, g2o, PCL, volumetric_mapping, open_chisel, chisel_server, voxblox, line_descriptor all need to be built first.
3. **libsgm SIMD intrinsics**: libsgm uses CUDA SIMD intrinsics (`__vcmpgtu2`, `__vminu2`, `__vmaxu2`, `__vcmpgtu4`, `__vminu4`, `__vmaxu4`) that have no direct HIP equivalent. Requires rewriting with portable scalar code or HIP/AMD-specific intrinsics.
4. **libsgm kernel launch syntax**: Has spaces in `<< <` which HIP rejects.

### Verified Working
- Basic HIP build infrastructure (CMakeLists.txt changes)
- cuda_to_hip.h compat header structure
- helper_cuda.h HIP section
- Cuda.cu compiles successfully with HIP

### Build Commands (HIP, once dependencies are installed)
```bash
# Install OpenCV 4.x with HIP support first
# Build third-party libraries per build_thirdparty.sh

# Then build PLVS without stereo libraries
mkdir build-hip && cd build-hip
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DWITH_LIBSGM=OFF -DWITH_LIBELAS=OFF -DWITH_CUDA=OFF
make -j8
```

### Next Steps to Unblock
1. Install OpenCV 4.x with HIP support (`-DWITH_HIP=ON` when building OpenCV)
2. Build required third-party libraries (run `build_thirdparty.sh` with HIP modifications)
3. Continue with main PLVS build
4. Address libsgm SIMD intrinsic rewrites if stereo functionality needed

### Dependencies Required
- ROCm 7.x (installed)
- OpenCV 4.x with HIP support (MISSING)
- Eigen3, Boost, Pangolin, PCL, GLOG, octomap, Protobuf, GLFW3 (need to check/install)
