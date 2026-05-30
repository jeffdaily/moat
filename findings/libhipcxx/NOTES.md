# ROCm/libhipcxx -- the libcu++ / `cuda::std` gap-filler for ROCm

`ROCm/libhipcxx` is AMD's header-only port of NVIDIA's libcu++/libcudacxx. It
provides the `cuda::std::` namespace (and `hip::std::` aliases) for HIP device
code, filling the gap that the ROCm 7.2.x release leaves: there is no libcu++
(`<cuda/std/*>`) shipped with ROCm 7.2.1, which is what deferred gsplat's 3DGUT
path (needs `cuda::std::optional`) and forced the MPPI-Generic `cuda::std -> std`
workaround.

Upstream: https://github.com/ROCm/libhipcxx

## What works on this stack (ROCm 7.2.1 / gfx90a / MI250X)

- Branch/commit validated: **`amd-develop`** at commit `fa4ccc6` ("Get rid of
  timing related error and emit warning instead (#22)"). The README says it is
  tested for ROCm 7.2.0; it compiles and runs cleanly on the host's ROCm 7.2.1
  hipcc toolchain, so the `release/2.2.x` / `v2.2.0` fallbacks were not needed.
- Header-only: nothing to build. Headers live in `include/`. Consume by adding
  `-I<clone>/include` to the hipcc compile.
- Namespace: `cuda::std::` (e.g. `cuda::std::optional`, `cuda::std::nullopt`,
  `cuda::std::numeric_limits`, `cuda::std::is_floating_point_v`). `hip::std`
  aliases the same symbols (`namespace hip = cuda;` in
  `include/cuda/std/__internal/namespaces.h:57`).
- Standards: C++17 and C++20 both compile + run here (README only tests C++17).
- No special macros required (confirmed: no `-D` define needed to enable it).
- Supported arches per README: gfx90a (MI210/MI250), gfx942 (MI300),
  gfx950 (MI355).

## NOT supported (per README "Limitations" table)

- `<cuda/ptx>` / the `cuda::ptx` namespace -- NOT ported.
- `<cuda/std/latch>`, `<cuda/std/barrier>`, `<cuda/std/semaphore>` and the
  extended `<cuda/latch|barrier|semaphore|pipeline>` -- NOT ported.
- `<cuda/annotated_ptr>` -- NOT ported.
- No NVIDIA-hardware/CUDA-backend support; no Windows.

## Install steps (reusable)

    # 1. Clone (amd-develop validated on ROCm 7.2.1; header-only, no build)
    git clone --depth 1 --branch amd-develop \
        https://github.com/ROCm/libhipcxx.git <dest>/libhipcxx
    # Fallbacks if amd-develop ever fails to compile on a newer/older toolchain:
    #   --branch release/2.2.x   then   --branch v2.2.0   (a tag)

    # 2. Add the include dir to the hipcc compile (header-only -- that's it)
    hipcc -std=c++17 -I<dest>/libhipcxx/include foo.cpp -o foo
    # C++20 also works: -std=c++20

NOT in the ROCm 7.2.x release, so vendor it from a clone (as above) rather than
expecting it under `/opt/rocm`.

## Validated `cuda::std::optional` smoke (gfx90a)

Source: `agent_space/libhipcxx_smoke.cpp`. Exercises exactly the facilities
gsplat's 3DGUT path uses -- `cuda::std::optional` / `cuda::std::nullopt` passed
as a `__global__` kernel arg (engaged and disengaged), `cuda::std::numeric_limits`
and `cuda::std::is_floating_point_v` on-device, plus a host-side optional and the
`hip::std` alias.

    cd agent_space
    HIP_VISIBLE_DEVICES=1 hipcc -std=c++17 --offload-arch=gfx90a \
        -I libhipcxx/include libhipcxx_smoke.cpp -o libhipcxx_smoke
    HIP_VISIBLE_DEVICES=1 ./libhipcxx_smoke

Output:

    engaged[0..2] = 0.0 3.0 6.0 (expect 0.0 3.0 6.0)
    cuda::std::optional/nullopt/limits/type_traits on gfx90a: PASS

Both `-std=c++17` and `-std=c++20` compile and the program returns 0 (PASS).

## Reusable include + flags (the payoff)

- Include path: `<clone>/include` (e.g.
  `/var/lib/jenkins/moat/agent_space/libhipcxx/include`).
- hipcc flag: `-I<clone>/include`. No macros, no link flags (header-only).
- For a torch CUDAExtension (Strategy B): add `-I<clone>/include` to the device
  (`nvcc`-key / `extra_cuda_cflags`) compile args, USE_ROCM-guarded. `<cuda/std/*>`
  and the `cuda::std::` namespace pass through torch's hipify UNCHANGED (hipify
  has no `cuda/std` or `cuda::std` mapping -- verified against
  `torch.utils.hipify.cuda_to_hip_mappings`), so no source edits are needed for
  the `cuda::std` symbols themselves; only the include path must be added.

## Consumers in MOAT

- gsplat 3DGUT (the concrete payoff): re-enabled on ROCm by pointing
  `LIBHIPCXX_INCLUDE` at this clone's include/ and building with `BUILD_3DGUT=1`.
  `gsplat/cuda/build.py` adds `-I$LIBHIPCXX_INCLUDE` to the device-compile flags
  USE_ROCM-guarded. The 3 FromWorld/ProjectionUT 3DGUT `.cu` files that
  `#include <cuda/std/optional>` and use `cuda::std::optional`/`nullopt` now
  COMPILE on gfx90a -- the headers resolve and pass through hipify unchanged.
  Result: extension builds (exit 0), `gsplat.has_3dgut()` is True, the 3DGUT ops
  (`projection_ut_3dgs_fused`, `intersect_tile_lidar`,
  `rasterize_to_pixels_from_world_3dgs_fwd`) register, and the device-side 3DGUT
  test suites pass on gfx90a (see projects/gsplat/notes.md for counts). One latent
  port fix was needed beyond the include: the FromWorld kernels' uncast
  `cudaFuncSetAttribute` (fault #9, never hit before because BUILD_3DGUT=0). The
  eval3d tests still fail only because their pure-torch REFERENCE needs `nerfacc`
  (absent on ROCm) -- not a libhipcxx or kernel defect.
  GOTCHA: gsplat's Config.h treats "explicitly enable ANY module" as "disable all
  unspecified modules", so build the full set together:
  `BUILD_3DGUT=1 BUILD_3DGS=1 BUILD_2DGS=1 BUILD_ADAM=1 BUILD_RELOC=1 BUILD_LOSSES=1`.
- MPPI-Generic: does NOT actually use `cuda::std` in its source (verified by grep),
  so libhipcxx is not needed there. Its only std-related fix is a libstdc++
  `std::void_t` C++17-backfill collision, unrelated to libcu++. Note added to
  projects/MPPI-Generic/notes.md for the record.
