# faster-gaussian-splatting notes

## Build (linux-gfx90a)

```bash
conda activate py_3.12  # ROCm torch env
HIP_VISIBLE_DEVICES=2 pip install -e projects/faster-gaussian-splatting/src/FasterGSCudaBackend --no-build-isolation
```

## Test

```python
import sys
sys.path.insert(0, 'projects/faster-gaussian-splatting/src/FasterGSCudaBackend')
import torch
from FasterGSCudaBackend.torch_bindings.rasterization import rasterize, RasterizerSettings

n = 500; device = 'cuda'; torch.manual_seed(42)
means = torch.randn(n, 3, device=device)
scales = torch.randn(n, 3, device=device)
rotations = torch.randn(n, 4, device=device)
opacities = torch.randn(n, device=device)
sh_0 = torch.randn(n, 3, device=device)
sh_rest = torch.randn(n, 15, 3, device=device)

w2c = torch.eye(4, device=device).unsqueeze(0)[:, :3, :]
settings = RasterizerSettings(
    w2c=w2c, cam_position=torch.zeros(3, device=device),
    bg_color=torch.zeros(3, device=device), active_sh_bases=1, width=256, height=256,
    focal_x=200, focal_y=200, center_x=128, center_y=128, near_plane=0.1, far_plane=100,
    proper_antialiasing=True
)

image = rasterize(means, scales, rotations, opacities, sh_0, sh_rest.view(n, -1), settings, to_chw=True, clamp_output=True)
print(f'Output: {image.shape}, range=[{image.min():.4f}, {image.max():.4f}]')
```

## Gotchas

1. **C++20 required**: PyTorch 2.x uses C++20 concepts/requires in headers. The extension must use `-std=c++20`.

2. **helper_math.h vector ops conflict**: HIP's HIP_vector_type provides all float2/3/4 operators. NVIDIA's helper_math.h defines the same operators, causing "ambiguous overload" errors on HIP. Fixed by guarding the operators with `HELPER_MATH_SKIP_VECTOR_OPS`.

3. **std::lerp conflict**: C++20 adds std::lerp(float,float,float). The helper_math.h scalar lerp conflicts with it when `using namespace std` is in effect (via PyTorch headers). Fixed by conditionally compiling the scalar lerp based on C++ version.

4. **cub::DoubleBuffer not mapped by hipify**: torch hipify maps `cub::Device*` functions to `hipcub::`, but not the `cub::DoubleBuffer` type. Fixed with `namespace cub = hipcub;` on HIP.

5. **rsqrt not available on host**: The rsqrt intrinsic is CUDA device-only. Host code in kernels_mcmc.cuh was calling it; changed to `1/std::sqrt()`.

6. **Backward pass gradient shape issue**: The diff_rasterize backward pass has a shape mismatch for opacity gradients (returns [N,1] but expects [N]). This is a pre-existing bug in the original code, not related to the ROCm port.

## Wave64 safety

The code uses `cg::tiled_partition<32>` for warp-level operations. This creates a 32-lane logical tile regardless of hardware wave width. The `constexpr warp_size = 32` matches the tile size, not the wavefront. This is arch-agnostic and safe on wave64 (gfx90a).
