<!-- DRAFT - held until RDNA validation completes. Before opening:
       (1) fill the Testing arch list with the Windows gfx1101 (RDNA3) and gfx1201 (RDNA4) results;
       (2) confirm the gfx1201 realtime-counter rate (currently assumed 100 MHz);
       (3) decide whether to also flip the UNSUPPORTED markers (see "Test enablement" note) in this PR;
       (4) delete this comment block.
     Title (conventional-commit style, matching this repo):
       feat(semaphore): support <cuda/std/semaphore> and <cuda/semaphore> on HIP/AMD
-->

## Summary

This enables `<cuda/std/semaphore>` and the extended `<cuda/semaphore>` on AMD/HIP by removing the `__HIP_PLATFORM_AMD__` `#error` from the two public umbrella headers. The counting and binary semaphore implementation is already present in the tree and builds entirely on the `cuda::std::atomic` wait/notify machinery that HIP already supports, so this adds and rewrites no semaphore logic; the umbrella `#error` was the only thing making the existing implementation unreachable on AMD.

This restores a capability that was available in 2.2.0 (the semaphores compiled before the umbrella gate was introduced) and resolves #10. Code that includes these headers, such as oneflow (`cuda::binary_semaphore<cuda::thread_scope_device>` in its embedding LRU cache) and the hipCollections and dgl stacks, pulls the semaphore types through the same umbrellas.

## What changed

- `include/cuda/std/semaphore`: remove the AMD `#error` so the existing `__semaphore/*` implementation is included.
- `include/cuda/semaphore`: remove the AMD `#error` (this extended header includes `<cuda/std/semaphore>`, so both must be ungated).
- `README.rst`: move semaphore to the supported set, with the forward-progress note below.

No implementation files are touched. The remaining `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700` guard is inert on AMD, where `__CUDA_ARCH__` is undefined.

## Forward-progress note

A blocking `acquire()` waits by polling, like the rest of libcu++'s blocking synchronization primitives. AMD GPUs do not provide independent forward progress for threads within a single wavefront, which is the same property that makes libcu++ require sm_70+ Independent Thread Scheduling for these waits on NVIDIA. Consequently a blocking `acquire()` whose matching `release()` is issued by another thread of the SAME wavefront is not guaranteed to make progress: whether it completes depends on compiler block ordering and should not be relied upon.

Supported patterns:
- Coordinate across separate wavefronts or blocks using `cuda::thread_scope_device` or `cuda::thread_scope_system`, where the producing and consuming wavefronts are scheduled independently. This is the common case (for example oneflow's device-scope usage) and works.
- Within a single wavefront, use the non-blocking `try_acquire()`, or the timed `try_acquire_for` / `try_acquire_until`, which return instead of spinning so the releasing thread can run.

## Testing

Built and run on real AMD GPUs (ROCm 7.2.1):
- gfx90a (CDNA2, wave64)
- gfx1100 (RDNA3, wave32)

The `version`, `max`, `try_acquire`, `acquire`, `release`, and `heterogeneous` semaphore conformance tests pass. A producer/consumer using `cuda::binary_semaphore<cuda::thread_scope_device>` across two blocks acquires and releases correctly, as does `cuda::thread_scope_system`. The realtime-counter rate used by `<cuda/std/chrono>` (100 MHz on RDNA) is already defined in the tree, so `try_acquire_for` and `try_acquire_until` resolve correctly on these targets.

`std/thread/thread.semaphore/timed.pass.cpp` launches a producer and a consumer as concurrent agents within a single wavefront, so on AMD it reflects the single-wavefront forward-progress property described above: the timed acquire reaches its deadline and returns false rather than acquiring. It is left `UNSUPPORTED: hipcc` with a note pointing at the forward-progress section.

## Test enablement

The six conformance tests above currently carry `UNSUPPORTED: hipcc`. They pass on AMD and the `hipcc` token can be removed so CI exercises them; `timed.pass.cpp` should keep `UNSUPPORTED: hipcc` with a comment referencing the forward-progress note, and `hiprtc` remains unsupported (not exercised here).

## Notes

This work was prepared with the assistance of Claude (an AI assistant by Anthropic), which did the analysis, the gate removal, and the validation.
