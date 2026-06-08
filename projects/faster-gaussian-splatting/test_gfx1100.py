#!/usr/bin/env python3
"""Validation test for faster-gaussian-splatting on gfx1100."""
import sys
sys.path.insert(0, '/var/lib/jenkins/moat/projects/faster-gaussian-splatting/src/FasterGSCudaBackend')
import torch
from FasterGSCudaBackend.torch_bindings.rasterization import rasterize, RasterizerSettings

def test_rasterize(n_gaussians, width, height, sh_bases=1, seed=42):
    """Test rasterization with given configuration."""
    device = 'cuda'
    torch.manual_seed(seed)

    means = torch.randn(n_gaussians, 3, device=device)
    scales = torch.randn(n_gaussians, 3, device=device)
    rotations = torch.randn(n_gaussians, 4, device=device)
    opacities = torch.randn(n_gaussians, device=device)
    sh_0 = torch.randn(n_gaussians, 3, device=device)
    sh_rest = torch.randn(n_gaussians, 15, 3, device=device)

    w2c = torch.eye(4, device=device).unsqueeze(0)[:, :3, :]
    settings = RasterizerSettings(
        w2c=w2c,
        cam_position=torch.zeros(3, device=device),
        bg_color=torch.zeros(3, device=device),
        active_sh_bases=sh_bases,
        width=width,
        height=height,
        focal_x=200,
        focal_y=200,
        center_x=width//2,
        center_y=height//2,
        near_plane=0.1,
        far_plane=100,
        proper_antialiasing=True
    )

    image = rasterize(means, scales, rotations, opacities, sh_0, sh_rest.view(n_gaussians, -1),
                      settings, to_chw=True, clamp_output=True)

    # Check output validity
    assert image.shape == (3, height, width), f"Unexpected shape: {image.shape}"
    assert not torch.isnan(image).any(), "Output contains NaN"
    assert not torch.isinf(image).any(), "Output contains Inf"
    assert image.min() >= 0.0 and image.max() <= 1.0, f"Output not in [0,1]: [{image.min():.4f}, {image.max():.4f}]"

    return image

def main():
    print("faster-gaussian-splatting validation on gfx1100")
    print(f"PyTorch: {torch.__version__}, HIP: {torch.version.hip}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    tests = [
        # (n_gaussians, width, height, sh_bases, name)
        (10, 128, 128, 1, "tiny-128x128"),
        (100, 128, 128, 1, "small-128x128"),
        (500, 256, 256, 1, "medium-256x256"),
        (1000, 256, 256, 1, "large-256x256"),
        (500, 512, 512, 1, "medium-512x512"),
        (500, 800, 600, 1, "medium-800x600"),
        (500, 256, 256, 4, "medium-256x256-sh4"),
        (500, 256, 256, 8, "medium-256x256-sh8"),
        (500, 256, 256, 16, "medium-256x256-sh16"),
        (5000, 256, 256, 1, "xlarge-256x256"),
        (10000, 256, 256, 1, "huge-256x256"),
    ]

    passed = 0
    failed = 0

    for n, w, h, sh, name in tests:
        try:
            img = test_rasterize(n, w, h, sh)
            print(f"PASS {name}: n={n}, res={w}x{h}, sh={sh}, range=[{img.min():.4f}, {img.max():.4f}]")
            passed += 1
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failed += 1

    # Determinism test: same seed should give same result
    print("\nDeterminism check...")
    img1 = test_rasterize(500, 256, 256, 1, seed=42)
    img2 = test_rasterize(500, 256, 256, 1, seed=42)
    if torch.equal(img1, img2):
        print("PASS determinism: bit-exact results across runs with same seed")
        passed += 1
    else:
        print(f"FAIL determinism: max diff = {(img1 - img2).abs().max()}")
        failed += 1

    # Different seed should give different result
    img3 = test_rasterize(500, 256, 256, 1, seed=99)
    if not torch.equal(img1, img3):
        print("PASS different-seed: different seeds produce different outputs")
        passed += 1
    else:
        print("FAIL different-seed: different seeds produced identical outputs")
        failed += 1

    # Multi-run consistency (same seed across multiple runs)
    consistent = True
    for i in range(3):
        img_run = test_rasterize(500, 256, 256, 1, seed=42)
        if not torch.equal(img1, img_run):
            print(f"FAIL multi-run consistency check {i}: outputs differ")
            consistent = False
            break
    if consistent:
        print("PASS multi-run: consistent across 3 runs with same seed")
        passed += 1
    else:
        failed += 1

    print(f"\n{'='*60}")
    print(f"Total: {passed + failed} tests, {passed} PASS, {failed} FAIL")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
