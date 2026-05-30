# [Bug] Layered cudaArray: texture fetch does not observe surface writes from a prior kernel launch (cross-launch texture-cache not invalidated) on gfx90a / CDNA2

## Component
HIP runtime / clr (texture + surface over `hipMalloc3DArray` layered arrays)

## Environment
- GPU: AMD Instinct MI250X / MI250 (`gfx90a:sramecc+:xnack-`, CDNA2)
- ROCm: 7.2.1
- HIP version: 7.2.53211-e1a6bc5663
- hipconfig --version: 7.2.53211-e1a6bc5663
- HSA Runtime Version: 1.18
- Kernel (amdgpu): 6.16.6
- Compiler: AMD clang 22.0.0git (roc-7.2.1)

## Summary
For a **layered** `hipArray` allocated with `hipArrayLayered | hipArraySurfaceLoadStore`,
a `tex2DLayered` fetch in kernel B does **not** observe values written by
`surf2DLayeredwrite` in a **separate, earlier** kernel A on the same stream. The
texture read returns the array's pre-write contents (zero), even though:
- a `surf2DLayeredread` of the same location in kernel B returns the fresh value, and
- an explicit `hipDeviceSynchronize()` between A and B does not help, and
- creating a brand-new texture object *after* the write (before B) does not help.

The identical pattern on a **non-layered 2D** array (`hipMallocArray` +
`surf2Dwrite` / `tex2D`) returns the fresh value correctly. So the defect is
specific to layered arrays.

Per both the CUDA C Programming Guide and AMD's own HIP documentation, the
texture/surface cache must be coherent **across** kernel-launch boundaries (it is
only undefined to read writes from the *same* launch). HIP's docs state
explicitly: "Since surfaces are also cached in the read-only texture cache, the
changes written back to the surface can't be observed in the same kernel. A new
kernel has to be launched in order to see the updated surface."
(https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.2/how-to/hip_runtime_api/memory_management/device_memory.html)
A new kernel *is* launched here, yet the update is not observed.

## Expected vs Actual
- Expected: kernel B's `tex2DLayered` returns 161.7 (the value kernel A wrote).
- Actual: kernel B's `tex2DLayered` returns 0.0 (stale, pre-write contents).

## Minimal reproducer
See `repro.cpp` (attached). Build and run:
```
hipcc --offload-arch=gfx90a -O2 repro.cpp -o repro
./repro
```
Observed output (deterministic, reproduced on two separate MI250X dies):
```
[A] Layered: tex created BEFORE write, read via TEXTURE, no sync between launches
  read via tex2DLayered (reused tex)             ... -> STALE (all == 0)
[B] Layered: read via SURFACE (surf2DLayeredread), no sync between launches
  read via surf2DLayeredread                     ... -> FRESH (all == magic)
[C] Layered: tex created AFTER write (fresh tex), read via TEXTURE
  read via tex2DLayered (fresh tex)              ... -> STALE (all == 0)
[D] Layered: reused tex, explicit hipDeviceSynchronize between launches
  read via tex2DLayered (reused tex, synced)     ... -> STALE (all == 0)
[E] Layered: reused tex, two rounds (write v1->read, write v2->read)
  round1 (write magic, read tex)                 ... -> STALE (all == 0)
  round2 (write 42, read SAME tex)               ... -> MIXED
[F] Non-layered 2D: tex created BEFORE write, read via TEXTURE
  read via tex2D (reused tex, 2D)                ... -> FRESH (all == magic)
```
The contrast between [A] (layered texture, STALE) and [F] (2D texture, FRESH),
plus [B] (layered surface read, FRESH), isolates the defect to the layered-array
texture-fetch path: the array backing store does contain the written value
([B], [F]), but the layered `tex2DLayered` path serves stale data and is not
invalidated at the kernel-launch boundary ([A], [C], [D]).

## Notes on root cause
- [C] rules out a stale image/texture descriptor snapshot taken at
  `hipCreateTextureObject` time: a texture object created *after* the write still
  reads stale data.
- [D] rules out host-side ordering: an explicit device sync between the launches
  does not change the result.
- [F] shows the cross-launch invalidation works for non-layered 2D arrays.
The most likely cause is that the image/texture L1 cache (or the layered image's
descriptor/metadata) is not invalidated at the kernel-launch boundary for layered
images on gfx90a, whereas it is for non-layered images.

## Real-world impact
Found while porting popsift (CUDA SIFT) to gfx90a. popsift builds its Gaussian
pyramid by writing blurred octave levels into a layered array via
`surf2DLayeredwrite` (kernel `gauss::absoluteSource::vert`) and then, in a
separate kernel launch on the same stream (`gauss::make_dog`), reading them back
via `tex2DLayered`. Because the texture read returns stale zeros, the entire
Difference-of-Gaussian pyramid is zero, no extrema are found, and 0 features are
produced. The application is in-spec per the CUDA programming guide (the write and
read are separate launches).
