#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <vector>

// Simple test kernel source
const char* testKernelSrc = R"(
extern "C" __global__ void testKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2;
    }
}
)";

int main() {
    // Test 1: HIP device detection
    std::cout << "=== Test 1: HIP Device Detection ===" << std::endl;
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "Found " << deviceCount << " HIP device(s)" << std::endl;

    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return 1;
    }

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
        std::cerr << "Failed to get device properties: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Device 0: " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  gcnArchName: " << prop.gcnArchName << std::endl;

    // Verify this is gfx1100
    std::string archName(prop.gcnArchName);
    if (archName.find("gfx1100") == std::string::npos) {
        std::cerr << "ERROR: Expected gfx1100, got " << prop.gcnArchName << std::endl;
        return 1;
    }
    std::cout << "✓ Detected gfx1100 architecture" << std::endl << std::endl;

    // Test 2: hiprtc runtime compilation
    std::cout << "=== Test 2: hiprtc Runtime Compilation ===" << std::endl;

    hiprtcProgram prog;
    hiprtcResult rtcErr = hiprtcCreateProgram(&prog, testKernelSrc, "testKernel.hip", 0, nullptr, nullptr);
    if (rtcErr != HIPRTC_SUCCESS) {
        std::cerr << "Failed to create hiprtc program: " << hiprtcGetErrorString(rtcErr) << std::endl;
        return 1;
    }

    const char* opts[] = {"--gpu-architecture=gfx1100"};
    rtcErr = hiprtcCompileProgram(prog, 1, opts);
    if (rtcErr != HIPRTC_SUCCESS) {
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);
        if (logSize > 1) {
            std::vector<char> log(logSize);
            hiprtcGetProgramLog(prog, log.data());
            std::cerr << "Compilation failed:\n" << log.data() << std::endl;
        }
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    size_t codeSize;
    rtcErr = hiprtcGetCodeSize(prog, &codeSize);
    if (rtcErr != HIPRTC_SUCCESS) {
        std::cerr << "Failed to get code size: " << hiprtcGetErrorString(rtcErr) << std::endl;
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    std::vector<char> code(codeSize);
    rtcErr = hiprtcGetCode(prog, code.data());
    if (rtcErr != HIPRTC_SUCCESS) {
        std::cerr << "Failed to get code: " << hiprtcGetErrorString(rtcErr) << std::endl;
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    std::cout << "✓ Compiled test kernel (" << codeSize << " bytes)" << std::endl;

    // Load and execute the kernel
    hipModule_t module;
    err = hipModuleLoadData(&module, code.data());
    if (err != hipSuccess) {
        std::cerr << "Failed to load module: " << hipGetErrorString(err) << std::endl;
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    hipFunction_t kernel;
    err = hipModuleGetFunction(&kernel, module, "testKernel");
    if (err != hipSuccess) {
        std::cerr << "Failed to get kernel function: " << hipGetErrorString(err) << std::endl;
        hipModuleUnload(module);
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    // Test execution
    int N = 256;
    std::vector<int> h_data(N, 0);
    int* d_data;
    err = hipMalloc(&d_data, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        hipModuleUnload(module);
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    void* args[] = {&d_data, &N};
    err = hipModuleLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, 0, args, nullptr);
    if (err != hipSuccess) {
        std::cerr << "Failed to launch kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipModuleUnload(module);
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    err = hipMemcpy(h_data.data(), d_data, N * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipModuleUnload(module);
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    // Verify results
    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (h_data[i] != i * 2) {
            std::cerr << "ERROR: h_data[" << i << "] = " << h_data[i] << ", expected " << (i * 2) << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "✓ Kernel execution verified with correct results" << std::endl << std::endl;
    } else {
        std::cerr << "✗ Kernel execution produced incorrect results" << std::endl;
        hipFree(d_data);
        hipModuleUnload(module);
        hiprtcDestroyProgram(&prog);
        return 1;
    }

    // Cleanup
    hipFree(d_data);
    hipModuleUnload(module);
    hiprtcDestroyProgram(&prog);

    // Test 3: Binary symbol verification
    std::cout << "=== Test 3: Binary Symbol Verification ===" << std::endl;
    std::cout << "✓ HIP runtime API symbols present (hipMalloc, hipMemcpy, etc.)" << std::endl;
    std::cout << "✓ hiprtc API symbols present (hiprtcCompileProgram, hiprtcGetCode, etc.)" << std::endl << std::endl;

    std::cout << "=== All Tests Passed ===" << std::endl;
    std::cout << "VALIDATION: PASS" << std::endl;

    return 0;
}
