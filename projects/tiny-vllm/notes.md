# tiny-vllm notes

## Build

```bash
cd projects/tiny-vllm/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -G Ninja
cmake --build build
```

For other architectures (e.g., gfx1100 for RDNA3):
```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -G Ninja
cmake --build build
```

## Port summary (gfx90a)

Strategy A (pure CMake, compat-header model) applied:

1. Created `src/cuda_to_hip.h` compat header with:
   - bfloat16 type mappings (`__nv_bfloat16` -> `__hip_bfloat16`)
   - CUDA runtime -> HIP runtime symbol aliases
   - cuBLAS -> hipBLAS symbol aliases
   - 64-bit warp mask constant for `__shfl_down_sync`

2. CMakeLists.txt changes:
   - Added `USE_HIP` option
   - Conditional HIP vs CUDA language enablement
   - Both main.cpp and kernels.cu compiled as HIP (bfloat16 types require HIP compiler)
   - hipBLAS linking on HIP path

3. Source changes:
   - kernels.cu: Added compat header include, replaced `0xffffffff` mask with `WARP_FULL_MASK`
   - kernels.cuh: Platform-conditional include for bfloat16 headers
   - main.cpp: Replaced CUDA/cuBLAS headers with compat header

## Key technical notes

- The `__shfl_down_sync` calls used a 32-bit mask (0xffffffff). HIP requires 64-bit masks (the runtime static_asserts `sizeof(MaskT)==8`), so we defined `WARP_FULL_MASK` as `0xffffffffffffffffULL` for HIP.

- main.cpp must be compiled as HIP (not plain CXX) because it uses `__nv_bfloat16` types throughout, and the HIP bfloat16 header (`hip/hip_bf16.h`) uses clang-specific builtins that GCC cannot compile.

- The paged attention kernel uses 64 threads per block (HEAD_DIM=64) with warp shuffles for a tree reduction. The pattern does two 32-thread warp reductions then combines via shared memory and `__syncthreads()`. This is wave-size agnostic because it uses logical 32-wide shuffles and block synchronization, working correctly on both wave64 (gfx90a) and wave32 (gfx1100).

## GPU detection test (gfx90a)

```
Device: AMD Instinct MI250X / MI250
Compute capability: 9.0
Global memory: 65520 MB
SM count: 104
Max threads per block: 1024
Free memory: 63GB, total memory: 63GB
```

The HIP runtime initializes correctly and detects the MI250X GPU.

## Validation dependency

Full inference validation requires Llama 3.2 1B Instruct model weights (`model.safetensors`). This model is gated on HuggingFace and requires authentication + license acceptance from Meta. Without the model file, the binary exits with "Can't open model.safetensors file" after successful GPU detection.

To validate with the model:
1. Log in to HuggingFace: `hf auth login`
2. Accept the Llama 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
3. Download the model:
   ```bash
   cd projects/tiny-vllm/src
   python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='meta-llama/Llama-3.2-1B-Instruct', filename='model.safetensors', local_dir='.')"
   ```
4. Run inference:
   ```bash
   HIP_VISIBLE_DEVICES=0 ./build/tiny-vllm
   ```
   This should produce token-by-token output for the 4 hardcoded prompts.

5. For deterministic comparison with reference:
   ```bash
   ./full_test.sh > output.txt
   # Compare generated tokens with reference.txt
   ```

The reference output is in `reference.txt` for comparison.

## Review 2026-06-05

### Commit Hygiene
**MOAT jargon in upstream-visible text**:
- `src/cuda_to_hip.h:4`: Comment says "Strategy A" which is MOAT internal vocabulary. Reword to describe what it does without the label (e.g., "keep CUDA spellings in source, alias to HIP on AMD").
- Commit message body contains "Uses Strategy A (compat-header model)" -- per CLAUDE.md, MOAT vocabulary must not appear in upstream-visible text. Reword to describe the approach without the label.

### Verdict
**Request Changes** -- the code is functionally correct and follows the porting approach properly. The only issue is the MOAT jargon ("Strategy A") appearing in the commit message body and a code comment. These are upstream-visible and must be reworded before the port can proceed.

## MOAT jargon fix (2026-06-05)

Fixed both instances of "Strategy A" per reviewer feedback:
1. `src/cuda_to_hip.h` line 4: Changed "Strategy A: keep CUDA spellings..." to "Keeps CUDA spellings in source and aliases them to HIP on AMD GPUs"
2. Commit message: Changed "Uses Strategy A (compat-header model): a cuda_to_hip.h header..." to "A cuda_to_hip.h header aliases CUDA spellings to HIP..."

Rebuilt and verified compilation still passes. Pushed 4297b8c to moat-port.

## Review 2026-06-05 (re-review after jargon fix)

Re-reviewed the port after jargon fixes. Both instances of "Strategy A" have been removed:
- `src/cuda_to_hip.h` line 4: Now says "Keeps CUDA spellings in source and aliases them to HIP on AMD GPUs"
- Commit message body: Jargon removed

Verified all fault classes:
- 64-bit lane masks: WARP_FULL_MASK correctly 0xffffffffffffffffULL for HIP
- The `threadIdx.x % 32` in ropeKernel/ropeKernelDecode (lines 95, 287) are HEAD_DIM frequency index math, not warpSize
- The `thread_id == 32` in pagedAttentionKernel is part of a wave-size-agnostic reduction: the 16/8/4/2/1 shuffle tree with width=64 correctly reduces lanes 0-31 to thread 0 and lanes 32-63 to thread 32, then combines via shared memory + __syncthreads()
- No textures, streams, events (no rule-of-five concerns)
- cuBLAS -> hipBLAS mappings correct
- Build system properly guarded (USE_HIP option, default OFF)
- Commit hygiene clean (no noreply, no MOAT jargon, [ROCm] title, Claude mentioned)

**Verdict: Approve** -- ready for validation. Validator needs HuggingFace access for Llama 3.2 1B model weights.
