#include "src/src/cuda_to_hip.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Test 1: GPU detection & runtime
bool test_gpu_detection() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 1) {
        printf("FAIL: No GPU detected\n");
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("PASS: GPU Detection - %s (compute %d.%d)\n",
           prop.name, prop.major, prop.minor);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("      Memory: %zuGB free / %zuGB total\n",
           free_mem/(1024*1024*1024), total_mem/(1024*1024*1024));
    return true;
}

// Test 2: Warp shuffle with 64-bit mask (pagedAttentionKernel pattern)
__global__ void test_warp_shuffle_kernel(float* input, float* output) {
    int tid = threadIdx.x;
    float val = input[tid];

    // Tree reduction using warp shuffles (same as pagedAttentionKernel)
    for (int delta = 16; delta >= 1; delta /= 2) {
        val += __shfl_down_sync(WARP_FULL_MASK, val, delta);
    }

    if (tid % 32 == 0) {
        output[tid / 32] = val;
    }
}

bool test_warp_shuffle() {
    const int N = 64;
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(2 * sizeof(float));

    // Input: all ones
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, 2 * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch with 64 threads like pagedAttentionKernel
    test_warp_shuffle_kernel<<<1, 64>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Each warp should sum to 32
    bool pass = (fabs(h_output[0] - 32.0f) < 0.01f && fabs(h_output[1] - 32.0f) < 0.01f);

    printf("%s: Warp Shuffle (64-bit mask) - warp0=%f warp1=%f\n",
           pass ? "PASS" : "FAIL", h_output[0], h_output[1]);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return pass;
}

// Test 3: bfloat16 embedding gather (embeddingGatherKernel pattern)
__global__ void test_embedding_gather_kernel(__nv_bfloat16* embeddings,
                                              int* tokens,
                                              __nv_bfloat16* output,
                                              int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < embed_dim) {
        int token_id = tokens[0];
        output[idx] = embeddings[token_id * embed_dim + idx];
    }
}

bool test_embedding_gather() {
    const int vocab_size = 128;
    const int embed_dim = 64;

    __nv_bfloat16* h_embeddings = (__nv_bfloat16*)malloc(vocab_size * embed_dim * sizeof(__nv_bfloat16));

    // Initialize: embedding[i][j] = i (token id)
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embed_dim; j++) {
            h_embeddings[i * embed_dim + j] = __float2bfloat16((float)i);
        }
    }

    int h_token = 42;
    __nv_bfloat16 h_output[embed_dim];

    __nv_bfloat16 *d_embeddings, *d_output;
    int *d_token;

    cudaMalloc(&d_embeddings, vocab_size * embed_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_token, sizeof(int));
    cudaMalloc(&d_output, embed_dim * sizeof(__nv_bfloat16));

    cudaMemcpy(d_embeddings, h_embeddings, vocab_size * embed_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_token, &h_token, sizeof(int), cudaMemcpyHostToDevice);

    test_embedding_gather_kernel<<<1, embed_dim>>>(d_embeddings, d_token, d_output, embed_dim);

    cudaMemcpy(h_output, d_output, embed_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Should get 42.0 for all elements
    bool pass = true;
    for (int i = 0; i < embed_dim; i++) {
        float val = __bfloat162float(h_output[i]);
        if (fabs(val - 42.0f) > 0.1f) {
            pass = false;
            break;
        }
    }

    printf("%s: Embedding Gather (bf16) - gathered token 42, value=%f\n",
           pass ? "PASS" : "FAIL", __bfloat162float(h_output[0]));

    cudaFree(d_embeddings);
    cudaFree(d_token);
    cudaFree(d_output);
    free(h_embeddings);

    return pass;
}

// Test 4: hipBLAS bf16 GEMM
bool test_hipblas_gemm() {
    const int M = 16, N = 16, K = 16;

    __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(M * K * sizeof(__nv_bfloat16));
    __nv_bfloat16* h_B = (__nv_bfloat16*)malloc(K * N * sizeof(__nv_bfloat16));
    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc(M * N * sizeof(__nv_bfloat16));

    // A and B: all ones
    for (int i = 0; i < M * K; i++) h_A[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < M * N; i++) h_C[i] = __float2bfloat16(0.0f);

    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16));

    cudaMemcpy(d_A, h_A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16BF, N,
                 d_A, CUDA_R_16BF, K,
                 &beta,
                 d_C, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(h_C, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Result should be K (16)
    float result = __bfloat162float(h_C[0]);
    bool pass = (fabs(result - (float)K) < 0.1f);

    printf("%s: hipBLAS bf16 GEMM - result=%f (expected %d)\n",
           pass ? "PASS" : "FAIL", result, K);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return pass;
}

int main() {
    printf("=== tiny-vllm HIP Port Component Tests (gfx1100) ===\n\n");

    int passed = 0, total = 4;

    if (test_gpu_detection()) passed++;
    if (test_warp_shuffle()) passed++;
    if (test_embedding_gather()) passed++;
    if (test_hipblas_gemm()) passed++;

    printf("\n=== Summary: %d/%d tests passed ===\n", passed, total);

    return (passed == total) ? 0 : 1;
}
