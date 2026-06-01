# cudf notes (ROCm/HIP port)

## Validation 2026-06-01 (linux-gfx1100, ROCm 7.2.1) -- VALIDATION-FAILED

Device: 4x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). ROCm 7.2.1 HIP clang 22.0.0. HIP_VISIBLE_DEVICES=0.

### Build

Fork clone: jeffdaily/cudf @ moat-port (c4b9b5bf7c522c9a4c6acba0de38f1d7e4abb3d4).

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/cudf/src/cpp -B projects/cudf/build-hip-gfx1100 -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/projects/rmm/install-gfx1100;/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DCUDF_CUCO_SOURCE_DIR=/var/lib/jenkins/moat/projects/cuCollections/src \
  -DCUDF_NVTX3_INCLUDE_DIR=/var/lib/jenkins/pytorch/third_party/NVTX/c/include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=ON
cmake --build projects/cudf/build-hip-gfx1100 -j16
```

Configure output: "CUDF HIP: arch=gfx1100 sources=204 (scoped core)". Build: 207/207 targets, 545 s (9.1 min), zero errors (warnings only). libcudf.so = 583,112,280 B (556.2 MB), well under 2 GiB.

Code-object verification: `llvm-objdump --offloading build-hip-gfx1100/libcudf.so` -> 151 embedded code objects, ALL `hipv4-amdgcn-amd-amdhsa--gfx1100`, zero non-gfx1100 (--offload-compress working, no multi-arch bloat).

### Smoke test (in-scope subset)

```
export LD_LIBRARY_PATH="/var/lib/jenkins/moat/projects/rmm/install-gfx1100/lib:${CONDA_PREFIX}/lib:/opt/rocm/lib"
HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/gtests/MOAT_CORE_SMOKE_TEST
```

Run 1: 18/18 PASS (8446 ms). Run 2: 18/18 PASS (6956 ms). All 18 tests: CountSetBitsWave64, SegmentedCountSetBits, CreateNullMask, CountIsDeterministic, SortRadixFloatTupleKeyShim, SortedOrderRadixFloatTupleKeyShim, DistinctCucoStaticSet, ApplyBooleanMask, ReduceSumMinMaxProduct, ReduceFloatingMean, HashInnerJoin, HashGroupbySum, QuantileLinear, DistinctCount, RepeatTable, TileTable, ListsDistinct, TDigestReducePercentile. No regression vs gfx90a (same 18 tests, same pass). Deterministic (two runs identical).

### Targeted wave32 ballot check -- FAIL: HSA hardware exception

The reviewer-flagged test: exercise `valid_if_kernel` with a non-multiple-of-32 null mask on gfx1100. `valid_if_kernel` (valid_if.cuh:58) uses `tiled_partition<32>::ballot(pred)` inside a `while (i < size)` per-lane loop. For a partial tail (size % 32 != 0), exited tail lanes do NOT call ballot -- the ballot call is divergent within the 32-lane tile.

Root cause isolated by a standalone harness (`agent_space/cudf_valid_if_harness/ballot_diverge.cu`):

- `tiled_partition<32>::ballot(pred)` called from CONVERGED code (all 32 tile members call it): PASS (correct result 0x55555555).
- `tiled_partition<32>::ballot(pred)` called from DIVERGENT code (`if (i < n)`): CRASH -- `HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception` (code 0x1016).

This is a behavioral difference from gfx90a:
- On gfx90a (wave64): 32-lane tile is a sub-wave of the 64-lane hardware wavefront. The hardware executes all 64 lanes in lockstep. Even "exited" tile members participate in the ballot at the hardware level (masked to 0 in the result). No divergence within the hardware wave, so no crash.
- On gfx1100 (wave32): 32-lane tile IS the hardware wavefront. When some lanes have exited the while loop (different PC from the ballot caller), `tile.ballot()` is called divergently. This is invalid for a cooperative groups collective and causes a hardware exception.

The CUDA path avoids this with `__ballot_sync(active_mask, ...)` which carries the `active_mask` across loop iterations. The HIP replacement `tile.ballot(pred)` inside a per-lane while loop does NOT carry a mask, so the partial tail case is structurally broken on gfx1100.

Confirmed: `valid_if_kernel` as written will crash on gfx1100 for any input size not a multiple of 32 (or block_size*grid_stride, more precisely: whenever the last tile in the grid stride has fewer than 32 active lanes).

### Result

- Build: PASS (gfx1100, 556 MB, 151 gfx1100 code objects, --offload-compress confirmed).
- In-scope smoke tests: 18/18 PASS (matches gfx90a, deterministic). None of the 18 tests exercise `valid_if_kernel` with a partial null mask.
- Targeted wave32 valid_if_kernel partial-tail check: FAIL -- HSA hardware exception on divergent `tiled_partition<32>::ballot()` call.
- State: validation-failed. Back to porter (opus).

### Fix sketch (for porter)

`valid_if_kernel` needs to restructure the loop so that ALL tile members call `tile.ballot()` at the same convergent point. The correct HIP pattern for a grid-stride ballot loop:

```cpp
// All threads in the tile call ballot together; out-of-range threads vote false.
while (any_in_tile_active) {
    bool active = (i < size);
    bool vote = active && p(*(begin + i));
    bitmask_type ballot = cudf::detail::ballot_32(vote);
    if (lane_id == leader_lane && active) {
        output[cudf::word_index(i)] = ballot;
        warp_valid_count += __popc(ballot);
    }
    // Advance; check whether ANY tile member has work remaining
    i += stride;
    bool any = tile.any(i < size);
    if (!any) break;
}
```

The key: `ballot_32` is called OUTSIDE any divergent branch -- ALL tile members call it, with out-of-range members voting `false`. The output write is guarded by `active` so only tile members that had valid work write to the output. The `tile.any(i < size)` determines when to exit.

Or equivalently: the existing per-lane `while (i < size)` loop can be replaced by a tile-level loop where the tile stays in the loop as long as ANY member has work. Within the loop body, each lane votes its `(i < size && pred)` result to the ballot, then only lane 0 writes the word if its own `i` is in range (the ballot already has the correct bits because out-of-range lanes vote false).

This fix would keep the 18/18 smoke tests passing and also make the valid_if partial-tail harness pass.

## Review 2026-05-31 (reviewer, linux-gfx90a, fork c4b9b5b vs upstream 6cea374)

Verdict: review-passed. ROCm fault-class-aware review via /pr-review (local-branch). Code, build system, export contract, and commit hygiene are sound; the CUDA path is unchanged (every edit is USE_HIP/__HIPCC__-guarded or a strict generalization valid on nvcc). The offload-compress single-.so decision and the find_package(cudf) dependency contract were verified directly against the built/installed artifacts. One documentation-accuracy finding (does not block).

FINDING (documentation, not a code defect): the "EXACTLY 2" undefined cudf-internal symbol claim is undercounted -- the built libcudf.so has 6.
- status.json blocked_reason and notes.md (the State header at the time of review, ~line 108 and ~line 176 "EXACTLY 1"/"EXACTLY 2") both state the cudf-internal undefined closure is "EXACTLY 2" (the binary_operation JIT pair). `nm -D -u build-hip/libcudf.so | c++filt` (cross-checked with `nm -DC`) shows 6: the 2 binary_operation overloads PLUS cudf::strings::detail::scan_inclusive<DeviceMin|Max> and cudf::structs::detail::scan_inclusive<DeviceMin|Max>. The latter 4 are defined in cpp/src/strings/scan/scan_inclusive.cu and cpp/src/structs/scan/scan_inclusive.cu, both scoped OUT of cpp/cmake/hip/cudf_hip_sources.cmake, referenced by the in-scope reductions/scan/scan_inclusive.cu dispatch. They are the SAME class as the binary_operation pair (header-declared dispatch entry, .cu scoped out), so the conclusion (tolerated via --unresolved-symbols=ignore-in-shared-libs) is unchanged; but the headline count is wrong and internally inconsistent (notes.md "Install as a dependency" already lists the scan_inclusive symbols). Action: correct the count to 6 (2 binary_operation + 4 scan_inclusive) in status.json blocked_reason and the notes State header so the downstream cugraph/cuml porter expects all six.

OFFLOAD-COMPRESS (sound, verified against the binary): cudf_hip.cmake:176-179 applies --offload-compress as PRIVATE + $<COMPILE_LANGUAGE:HIP>-scoped (gfx90a-only via CMAKE_HIP_ARCHITECTURES, default gfx90a). Build is non-RDC (no -fgpu-rdc anywhere in cpp/cmake/hip/), so the per-TU compile is the only place the device bundle forms -- single flag site is correct and complete (no separate device-link to also flag). `llvm-objdump --offloading build-hip/libcudf.so`: 151 embedded code objects, ALL hipv4-amdgcn-amd-amdhsa--gfx90a, no other arch (no multi-arch fatbin bloat); host image 586,646,592 B; no -mcmodel=large anywhere. The single-.so vs multi-.so-split choice is the right one and is realized.

DEPENDENCY-EXPORT CONTRACT (complete + verified end-to-end against installed artifacts in _deps/cudf/install): cudf-config.cmake does find_dependency(hip)+find_dependency(rmm) then includes the targets; cudf-targets.cmake exports `add_library(cudf::cudf SHARED IMPORTED)` with INTERFACE_LINK_LIBRARIES "rmm::rmm" only (PRIVATE cuco::cuco + hip::host correctly absent from a shared lib's interface); IMPORTED_LOCATION -> lib/libcudf.so; installed .so RUNPATH $ORIGIN:/opt/rocm-7.2.1/lib, NEEDED librmm.so + librapids_logger.so + libamdhip64. Minor: notes say RUNPATH is /opt/rocm/lib; the actual embeds the resolved versioned /opt/rocm-7.2.1/lib (functionally equivalent, /opt/rocm is the symlink). Consumer constraint (downstream TU must compile as HIP) is documented and correct.

BINARY_OPERATION-JIT DEFERRAL RULING (acceptable documented follow-on; conditional, see caveat): binary_operation (and the 4 scan_inclusive) are genuine separate-phase deferrals -- binary_operation is the jitify->hipRTC JIT path (jitify_hiprtc_plan.md), scan_inclusive is a not-yet-listed module, neither structurally blocked. They do NOT block the dependency contract for a consumer that does not CALL them: libcudf.so is a shared lib and these are undefined IN it, so they only become a consumer link error if the consumer's own TUs reference them, and the documented consumer linkage (--unresolved-symbols=ignore-in-shared-libs -z lazy) tolerates that. CAVEAT for the validator/downstream: this is conditional on cugraph/cuml not actually invoking cudf::binary_operation or cudf::strings|structs::scan_inclusive at their link. If a future cugraph/cuml port references binary_operation at link time, the binaryop JIT module (or a non-JIT bring-up) becomes a prerequisite then -- flag at that point. For THIS cudf port it is an acceptable, now-correctly-counted (6, not 2) follow-on.

FAULT CLASSES (all correct):
- wave64 null-mask ballot: all in-scope sites (valid_if.cuh x2, copy_range.cuh, copy_if_else.cuh, copying/concatenate.cu x2, strings/copying/concatenate.cu, replace/replace.cu, replace/nulls.cu) route through detail::ballot_32 (warp_primitives.cuh) = cg::tiled_partition<32>::ballot, tile-relative on any wave width; warp_size stays 32 (the bitmask word width, NOT the hardware wave). The CUDA active_mask grid-stride dance is correctly dropped on HIP. valid_if_kernel (valid_if.cuh:58) is the only genuinely divergent-loop site (per-lane `while (i < size)`); analyzed correct -- for the partial final word the exited tail lanes contribute 0 to tile.ballot(), matching the CUDA active_mask result, and the leader (lane 0 = lowest index in the word) is in-range whenever the word is written. NOTE for the validator: the 18 smoke tests exercise valid_if_kernel transitively (quantile/cast_ops/shift are in scope) but mostly over no-null / multiple-of-32 inputs; the divergent partial-tail path with a non-multiple-of-32 null mask is thinly stressed. Analysis holds; worth a targeted check in the broader suite. valid_if_n_kernel and copy_range_kernel loops are block-uniform (not per-lane divergent), so their tile ballots are fully converged.
- cast_functor.cuh:45-62 dangling-ref fix: same-type overload returns BY VALUE under USE_HIP (was T&& copy-elide -> dangling ref through transform_iterator feeding hipCUB DeviceReduce); CUDA keeps the T&& forward form. Correct, and UB-correct on CUDA too.
- radix_tuple_key.cuh re-ties cuda::std::tuple lvalue refs into rocprim::tuple via rocprim::tie (refs into the live key, safe across return); applied at the 2 float decomposer sites (sort_radix.cu, sorted_order_radix.cu) under USE_HIP. sorted_order_radix also returns thrust::tuple (not cuda::std::pair) from its zip transform on HIP. Sound.
- tdigest_aggregation.cu: file-local zip_tuple/zip_get/zip_make_tuple alias (thrust:: on HIP, cuda::std:: on CUDA) applied ONLY at zip-iterator/reduce_by_key boundaries; the operator()-attribute move (10 sites, __device__/CUDF_HOST_DEVICE before operator()) is arch-unified and mechanical. Correct.
- device_span::operator[] -> CUDF_HOST_DEVICE on HIP (span.hpp:406): justified -- generate_cluster_limit (tdigest_aggregation.cu:462, CUDF_HOST_DEVICE) indexes device_spans and is called host-side at :669 for small group counts. The change only ADDS host-callability (device pass unchanged); does not wrongly expose device-only behavior (the host call path operates over host-addressable/pinned spans). CUDA keeps the device-only form.
- cuco_helpers.hpp cuco_allocator: HIP subclass adds the 2-arg allocate/deallocate(num, cuda::stream_ref) overloads + rebind + converting ctor, retains 1-arg via using base::; CUDA keeps the plain alias. Correct.

COMMIT HYGIENE (clean): title "[ROCm] HIP build for AMD GPUs (USE_HIP): libcudf core + algorithms" (66 chars, <=72, [ROCm] prefix); body discloses Claude authorship, has a Test Plan with literal commands, gives a review order for the large change; NO Co-Authored-By noreply trailer; no non-ASCII / em-dash in added lines; no AMD-internal account refs; fork remote is jeffdaily; fork main stays a clean upstream mirror (port on moat-port). Single curated commit on REL v25.08.00.

BUILD SYSTEM (correct): cpp/CMakeLists.txt:17-29 USE_HIP option (default OFF) + early include/return guard bypasses rapids-cmake; enable_language(HIP); arch flows from CMAKE_HIP_ARCHITECTURES (gfx90a default); ROCm deps gated behind the HIP build; CUDA build path untouched.

CUDA-PATH PRESERVATION: the 10 unguarded edits (device_operators.cuh +const; device_atomics.cuh/cast_ops.cuh typename; reduction_operators.cuh ::template; gather.cuh + row_operators.cuh __device__->CUDF_HOST_DEVICE; null_mask.cuh/integer_utils.hpp added includes; list_device_view.cuh public:; type_dispatcher.hpp CUDF_HOST_DEVICE; ewm.cu explicit narrowing casts) are each strict generalizations -- nvcc-valid, no numeric/default change -- the one bc-guidelines-permitted form of shared edit. The macro root fix (types.hpp:19 __CUDACC__ -> ||__HIPCC__) and the 3 other genuine __CUDACC__-device-guard sites are additive.


Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated
RAPIDS). `depends_on: [rmm, cuCollections]` (both COMPLETED/ported). cudf is
itself a dependency of cugraph and cuml.

Pinned upstream: tag `v25.08.00` (commit 6cea374, in `projects/cudf/src`,
gitignored), matching the rmm/raft/cuCollections v25.08 line.

Fork: jeffdaily/cudf @ `moat-port`. The port is a standalone HIP CMake build
reached by a top-of-CMakeLists `if(USE_HIP) include(cmake/hip/cudf_hip.cmake)
return()` guard; the CUDA build is untouched.

## RESOLVED: the libcudf.so relocation wall -- `--offload-compress` shrinks the device fatbin ~55x

The prior "NEXT WALL" (reductions/aggregation/join/groupby do not fit a single
libcudf.so without -mcmodel=large) is RESOLVED, and NOT by a structural split.
Root cause was that the HIP offload bundle is stored UNCOMPRESSED by default, so
each rocPRIM reduce TU embeds a ~159 MB device fatbin (the full type x op ISA
matrix); 15+ reductions push the host image past the +/-2 GiB x86-64 relocation
reach. nvcc fatbins are compact, which is why the CUDA build never hits this.

`--offload-compress` (a documented ROCm 7.2.1 clang HIP flag: "Compress offload
device binaries (HIP only)") stores the embedded device bundle in the compressed
CCOB format; the HIP runtime decompresses on load. Controlled A/B measurement on
`reductions/sum.cu` @ gfx90a (same compile line, flag stripped vs present):

| metric        | uncompressed | compressed | shrink |
|---------------|--------------|------------|--------|
| object (.o)   | 207.8 MB     | 52.1 MB    | 3.99x  |
| `.hip_fatbin` | 158.6 MB     | 2.89 MB    | 54.9x  |

(fatbin drops from 76.3% to 5.6% of the object; 155.7 MB removed per TU). The
device-bundle 55x is the headline: the rocPRIM type x op ISA matrix is hugely
redundant, so it compresses extremely well. Wired in cudf_hip.cmake as
`target_compile_options(cudf PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-compress>)`.
cudf compiles each TU non-RDC (no -fgpu-rdc device-link), so the per-TU compile
is the only place the bundle is formed -- there is no separate device-link to
also flag. This is what lets the broad algorithm surface (reductions +
aggregation + join + groupby) link the SINGLE libcudf.so with NO -mcmodel=large
and NO multi-.so split.

### LINK VERDICT (fork bec85ee): YES -- single libcudf.so, no split, no -mcmodel=large

With --offload-compress the broad algorithm surface (reductions + aggregation +
hash/sort groupby + hash/sort-merge join, ~44 new TUs on top of the core) links
into ONE libcudf.so: host image ~540 MB (566,964,928 B), far under the +/-2 GiB
x86-64 relocation reach. `llvm-objdump --offloading libcudf.so` shows exactly
144 embedded code objects, ALL `hipv4-amdgcn-amd-amdhsa--gfx90a` and nothing
else (gfx90a-only, no multi-arch fatbin bloat). The three documented relocation
overflow classes (PC32, TLS, .eh_frame) do not appear -- the compressed bundle
keeps the image small enough that none is reached. The single-.so approach jeff
wanted is achieved; NO structural split, NO -mcmodel=large.

Undefined cudf-internal symbols grew 1 -> 17 (the broad surface references more
deferred-dispatch entries): the 2 binary_operation overloads (JIT), plus
quantile/group_quantiles, the tdigest group+reduce set, repeat/tile,
lists::distinct + distinct_count, and scan_inclusive<DeviceMax/Min> for
strings/structs. None is exercised by the validated tests; each is the next
module to bring up. The smoke test links with --unresolved-symbols=
ignore-in-shared-libs so these are tolerated (not referenced at runtime).

### GPU VALIDATION (gfx90a / MI250X, GCD 0): 12/12 PASS

The 8-test baseline still passes (no regression) PLUS 4 new broad-surface gates:
- ReduceSumMinMaxProduct: cudf::reduce sum/min/max/product over int32 == host.
- ReduceFloatingMean: cudf::reduce mean/sum/max over float + double == host.
- HashInnerJoin: cudf::inner_join (cuco static_multimap) matched (l,r) pairs ==
  host inner join of the key sets.
- HashGroupbySum: cudf::groupby hash SUM per INT32 key == host fold (arbitrary
  group order, compared as a key->sum map).
So reductions, hash join, AND hash groupby all execute correctly on real gfx90a;
the compressed device bundle decompresses and runs.

### Reduction correctness bug found + fixed: cast_fn dangling reference (cast_functor.cuh)

The first GPU run of the new reduction tests FAILED in a precise pattern: int
min returned INT32_MIN, int max returned INT32_MAX (the seeded identities), and
double sum/mean/max returned nan/0 -- while int sum and int product (and the
groupby SUM) were correct. The discriminator was exactly `OutputType ==
ElementType`: min/max keep the input type, and a double-column sum outputs
double; an int32 column summed to INT64 (output != element) worked. Root cause:
`cudf::detail::cast_fn<T>` (the element transformer applied to dcol->begin<T>()
before every simple reduction) had a same-type overload `operator()(T&&) ->
T&&` that returns a reference to elide a copy. When OutputType==ElementType that
overload is selected; as the transform of a thrust::transform_iterator feeding
hipCUB DeviceReduce::Reduce it returns a reference into EXPIRED iterator
storage, and rocPRIM block-loads the dangling reference -> garbage. This is UB
that nvcc/CUB tolerate. Fix (USE_HIP-guarded, correct on CUDA too): the same-type
overload returns BY VALUE.

Localization method (worth reusing): reduce the SAME int32 column three ways in
the test -- (1) cudf::reduce MIN, (2) raw-pointer hipCUB DeviceReduce::Min, (3)
thrust::reduce with cudf::DeviceMin over dcol->begin<int32>(). (2) and (3) both
returned -4 (correct) while (1) returned INT32_MIN, which isolated the bug to
the cudf reduce wrapper (the cast_fn transform + DeviceReduce::Reduce with a
custom op), NOT the column iterator, the DeviceMin operator, or rocPRIM. A
standalone microbench of DeviceReduce::Reduce + transform_iterator + the
dangling cast_fn did NOT reproduce it (the dangle only manifests through cudf's
nested column_device_view iterator instantiation), so the in-test three-way
probe was the decisive tool. The five edits the prior session staged for the
reductions bring-up (ewm.cu narrowing casts, device_atomics typename T::duration,
compute_aggregations SetType::template ref_type, the join/groupby copy_if
stencil predicates, sort_merge_join thrust::get) are all still needed and are in
bec85ee; the cast_fn fix is the new one that makes the reductions CORRECT, not
just compiling.

## Validation 2026-05-31 (validator, linux-gfx90a, fork c4b9b5b)

Device: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0. ROCm 7.2.1.

Build reused (intact build-hip/libcudf.so, 586,646,592 B, 151 gfx90a code objects confirmed via llvm-objdump --offloading). git rev-parse HEAD == c4b9b5bf7c522c9a4c6acba0de38f1d7e4abb3d4.

Undefined cudf-internal symbols confirmed: 6 total (2 binary_operation JIT overloads + 4 scan_inclusive<DeviceMin|Max> for strings/structs). The `nm -D -u | grep 'U cudf'` filter misses the 4 scan_inclusive symbols because their demangled form starts with `std::unique_ptr<cudf::column>` not `cudf::` -- use `grep "scan_inclusive\|binary_operation" | grep "cudf::"` to see all 6.

GPU test run (x2 for determinism):
```
source /var/lib/jenkins/moat/agent_space/cudf_build_env.sh
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/cudf/build-hip/gtests/MOAT_CORE_SMOKE_TEST
```
Run 1: 18/18 PASS (9047 ms)
Run 2: 18/18 PASS (9107 ms)

All 18 tests: CountSetBitsWave64, SegmentedCountSetBits, CreateNullMask, CountIsDeterministic, SortRadixFloatTupleKeyShim, SortedOrderRadixFloatTupleKeyShim, DistinctCucoStaticSet, ApplyBooleanMask, ReduceSumMinMaxProduct, ReduceFloatingMean, HashInnerJoin, HashGroupbySum, QuantileLinear, DistinctCount, RepeatTable, TileTable, ListsDistinct, TDigestReducePercentile.

valid_if divergent-tail disposition: valid_if_kernel (valid_if.cuh:58, HIP path `while (i < size)`) is correctly coded -- exited tail lanes contribute 0 to tile.ballot(), matching the CUDA active_mask result. The 18 smoke tests do NOT directly exercise valid_if_kernel with a partial null mask (the tdigest path that calls valid_if only fires when nulls are present, which none of the tests trigger; distinct/apply_boolean_mask use their own cuco/stencil paths). The partial-tail path is thinly stressed by this suite; analysis of the code is sound. Noted as a follow-on coverage gap for a broader suite; does not block completion.

find_package(cudf) consumer confirmed:
```
HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="/var/lib/jenkins/moat/_deps/cudf/install/lib:$LD_LIBRARY_PATH" /var/lib/jenkins/moat/agent_space/cudf_consumer/build/consumer
# -> cudf consumer linked; column size=8 (exit 0)
```
cudf::cudf target resolves, libcudf.so loads, make_numeric_column runs on gfx90a.

Deferred-symbol count corrected from "EXACTLY 2" to 6 (2 binary_operation JIT + 4 scan_inclusive<DeviceMin|Max>). cugraph/cuml porter should expect all 6 in libcudf.so.

State: completed. validated_sha = c4b9b5bf7c522c9a4c6acba0de38f1d7e4abb3d4.

## State: linux-gfx90a -> completed (fork c4b9b5b). Core fully links: 6 cudf-internal deferred symbols (2 binary_operation JIT + 4 scan_inclusive<DeviceMin|Max>); find_package(cudf) export contract GPU-verified

This session (fork ae6edc8 -> c4b9b5b) brought up the remaining ~15 non-JIT
modules behind the prior 17-symbol residual and drove the cudf-internal
undefined closure to EXACTLY 6 (2 binary_operation JIT pair + 4 scan_inclusive<DeviceMin|Max>). Modules added to
cudf_hip_sources.cmake (9 new TUs): stream_compaction/distinct_count.cu,
quantiles/quantile.cu, quantiles/tdigest/{tdigest.cu, tdigest_aggregation.cu,
tdigest_column_view.cpp}, groupby/sort/group_quantiles.cu, filling/repeat.cu,
reshape/tile.cu, lists/stream_compaction/distinct.cu. libcudf.so = 559.46 MB
(586,646,592 B), 151 embedded code objects ALL gfx90a, no -mcmodel=large, no
split. GPU 18/18 on MI250X (GCD 0): the prior 12 (no regression) + QuantileLinear,
DistinctCount, RepeatTable, TileTable, ListsDistinct, TDigestReducePercentile,
stable over repeated runs. find_package(cudf) export contract added + verified
(see "Install as a dependency"). This unblocks cugraph/cuml (which can now build
on jeffdaily/cudf @ moat-port).

### Three new arch-unified fault-class fixes (tdigest bring-up), all CUDA-unchanged

All in quantiles/tdigest/tdigest_aggregation.cu + one core header:
1. **rocThrust zip-iterator / rocPRIM reduce_by_key tuple interop.** rocThrust's
   zip_iterator is parameterized on thrust::tuple and its reference neither
   matches nor assigns a cuda::std::tuple (CCCL thrust does; rocThrust does not),
   and rocPRIM reduce_by_key stores its accumulator in a thrust/rocprim tuple.
   The tdigest `centroid` value type and the min/max zip-iterator functors used
   cuda::std::tuple, so (a) `make_zip_iterator(cuda::std::make_tuple(...))` did
   not form an iterator and (b) `output = op(...)` (cuda::std::tuple) and
   rocPRIM's internal `get<1>(reduction)` assignment had no viable `=`. Fix: a
   file-local `zip_tuple`/`zip_get`/`zip_make_tuple` alias = thrust:: on HIP,
   cuda::std:: on CUDA, applied ONLY at the zip-iterator boundaries (centroid
   reduce_by_key + its merge_centroids get<>; the min/max thrust::transform;
   tdigest_min/max read). Per-site bodies unchanged; CUDA byte-for-byte
   identical. Same fault family as the sorted_order_radix thrust::tuple fix.
2. **clang rejects an attribute between operator() and its parameter list.**
   tdigest_aggregation.cu wrote `operator() __device__(args)` and
   `operator() CUDF_HOST_DEVICE(args)` (10 sites); nvcc tolerates it, clang does
   not ("'operator()' cannot be the name of a variable"). Moved the attribute
   before operator() (`__device__ ret operator()(args)`), which both compilers
   accept. Arch-unified (no guard needed).
3. **device_span::operator[] made CUDF_HOST_DEVICE on HIP** (include/cudf/
   utilities/span.hpp). It was `__device__`-only, but cudf has CUDF_HOST_DEVICE
   functors (cumulative_*/centroid_group_info/scalar_group_info_grouped) that
   index a device_span and are instantiated for the host CPU path of tdigest
   `generate_cluster_limit` (over pinned, host-addressable spans). clang forbids
   a __host__ __device__ function calling a __device__-only callee (nvcc warns
   only). USE_HIP-guarded: HIP gets host+device (matching the host span's
   indexer; device pass unchanged), CUDA keeps the device-only form unchanged.
   This is a core force-included header, so the change recompiles most TUs but
   is low-risk (only ADDS host-callability) and likely prevents the same error
   in cugraph/cuml device_span use.

(Superseded baseline below; the broad-surface + these modules subsume it.)

(Superseded baseline below, kept for the fault-class history; the 107-TU core is
now subsumed by the broad-surface build above at fork bec85ee.) A scoped libcudf
core (107 TUs, fork HEAD 5be894c) compiles for gfx90a and links libcudf.so with
a real gfx90a code object. The focused GPU gtest passes 8/8 on MI250X (stable,
repeated runs): the 4 wave64 null-mask tests, the 2 tuple-key radix-sort tests,
PLUS 2 cuco-backed tests (DistinctCucoStaticSet, ApplyBooleanMask).

THE HEADLINE RESULT this session: the cudf-INTERNAL type-erased dispatch
link-closure collapsed from ~209 undefined symbols (at the 83-TU core) to
EXACTLY 1. Bringing up ~24 modules (table row_operators +
primitive_row_operators -> the experimental::row machinery; stream_compaction
distinct/apply_boolean_mask/distinct_helpers/stable_distinct; the full hashing
set; search contains_column/table; structs/lists utilities + lists dremel +
scatter_helper + dictionary set_keys + transform encode) DEFINED essentially
every symbol the foundational core referenced. Verified by `nm -D -u
libcudf.so | c++filt | grep 'U cudf'`: the ONLY remaining cudf-internal
undefined symbol is `cudf::detail::binary_operation(column, scalar)` -- the
jitify->hipRTC binaryop JIT path (the next phase). The prior-flagged symbols
(dictionary::detail::match_dictionaries, lists::detail::
build_lists_child_column_recursive, experimental::row::lexicographic::
preprocessed_table::create) are all DEFINED now (nm shows T). Every other
undefined symbol in the .so is the normal external shared-library surface
(rmm, rapids_logger, libstdc++, libc, hip runtime), resolved at load.

cuCollections refresh: the cuco dep MUST be the CURRENT moat-port HEAD
(47ae24d, with include/cuco/detail/hip_device_select.cuh -- the retrieve_all
DeviceSelect adaptor), NOT the older 57a1c1a. DistinctCucoStaticSet exercises
distinct() -> cuco::static_set insert_and_find + retrieve_all on real GPU
through that adaptor; GPU-validated.

### [RESOLVED by --offload-compress -- see the top section] the libcudf.so x86-64 relocation scale limit

NOTE: this wall is RESOLVED at fork bec85ee. The analysis below is the original
characterization (kept for the record). The fix was NOT a split or
-mcmodel=large; it was --offload-compress, which shrinks each reduce TU's
embedded device bundle ~55x (158.6 MB -> 2.89 MB), so the whole broad-surface
libcudf.so stays at ~540 MB, well under 2 GiB. See the top "RESOLVED" + "LINK
VERDICT" sections.

Reductions + aggregation were attempted on top (build_2/build_3) and REVERTED:
they do not fit. The reduction TUs are pathologically large on rocPRIM --
reductions/std.cu and var.cu are 234 MB EACH, mean/sum/product/sum_of_squares
~200 MB each, dominated by a ~190 MB `.hip_fatbin` per TU (rocPRIM instantiates
the cub/rocPRIM `reduce_impl` over the FULL type x DeviceSum/Min/Max matrix;
the device ISA for every instantiation is embedded). Adding the 15 simple
reductions + aggregation pushed the cudf.dir objects to 2.5 GB and the linked
libcudf.so host image past the +/-2 GiB reach of x86-64 32-bit relocations.
Three distinct relocation classes overflow, in escalating order:
  1. R_X86_64_PC32 (`.text` -> `.rodata.str`/`.data`): the FIRST overflow.
     FIXABLE with `-mcmodel=large` on the HIP host pass (64-bit data refs); the
     amdgcn device pass is unaffected. Confirmed: with -mcmodel=large the
     libcudf.so relink PASSED (ninja [121]->[122], 0 PC32 errors).
  2. BUT -mcmodel=large then exposes R_X86_64_TLSGD / R_X86_64_TLSLD /
     R_X86_64_PLT32-to-__tls_get_addr in stream_pool.cpp + host_memory.cpp
     (the thread_local `event_for_thread()::thread_events` and rmm's
     `stream_ordered_memory_resource::get_event()::events_tls`). The large CODE
     model does NOT imply a large TLS model, and clang has no large-TLS x86-64
     model that covers a >2 GiB TLS-block-relative access, so these still
     overflow.
  3. AND `.eh_frame` "PC offset is too large" in column.cu (the unwinder's
     32-bit PC-relative FDE offsets).
So -mcmodel=large alone is NOT sufficient; reductions need a structural fix:
either SPLIT libcudf into multiple .so (e.g. libcudf_reductions.so) so no single
image exceeds 2 GiB, or shrink the per-TU device fatbin (limit the reduction
type matrix). This is the concrete next wall -- a build-scale problem, not a
HIP-correctness one. The cuco-backed join/groupby modules will hit the same
scale ceiling and likely need the same split. The 107-TU core (5be894c) stays
the validated, pushed baseline; it links FINE without -mcmodel=large (it is
under 2 GiB: libcudf.so ~408 MB).

### This session cleared the 3 documented "next walls" (rocPRIM tuple, array, ballot)

- **rocPRIM radix sort over cuda::std::tuple keys -- SOLVED with a reusable
  shim, NO rocPRIM modification.** cudf's float fast-sort
  (sort/sort_radix.cu SortKeys, sort/sorted_order_radix.cu SortPairs) packs
  {int32 nan-bias, float} into a struct and sorts with a "decomposer" returning
  `cuda::std::tuple<size_type&, F&>`. hipCUB forwards the decomposer to rocPRIM,
  whose radix codec for a CUSTOM key only recognizes `rocprim::tuple`
  (`rocprim/type_traits.hpp` codec<Key,Desc,false>: `is_tuple_of_references` /
  `for_each_in_tuple` / `rocprim::tuple_size` / the internal
  `extract_digit_from_key_impl(const rocprim::tuple<Args...>&)` ALL pattern-match
  rocprim::tuple; `type_traits_functions.hpp` `is_tuple_of_references` primary
  even `static_assert(sizeof(T)==0,"only implemented for rocprim::tuple")`). Fix:
  `cpp/include/cudf/detail/utilities/hip/radix_tuple_key.cuh` -- a decomposer
  ADAPTOR that calls the user decomposer then re-`rocprim::tie`s the SAME lvalue
  references into a `rocprim::tuple<Refs...>` (the references point into the live
  key, safe across the wrapper). rocPRIM's existing, validated codec then drives
  the sort. Applied at the 2 float-path decomposer sites under `#if USE_HIP`
  (CUDA byte-for-byte unchanged). The header is library-agnostic (knows only
  cuda::std::tuple + rocprim::tuple, not cudf) -- LIFTABLE VERBATIM into
  cuCollections retrieve_all and raft, which hit the identical gap. GPU-proven:
  the standalone shim sorts 100000 floats correctly via
  hipcub::DeviceRadixSort::SortKeys, and the in-tree GPU gtest sorts FLOAT32
  columns asc+desc (sizes 1..100000) == host std::sort. So the rocPRIM tuple-key
  wall is NOT infeasible -- the shim clears it without touching rocPRIM, and the
  per-project packed-key fallback is NOT needed.
  - sorted_order_radix.cu ALSO needed a second fix: its zip-iterator transform
    functor `float_to_pair_and_seq` returned `cuda::std::pair<float_pair,size_type>`,
    and rocThrust's zip-iterator reference (a thrust::tuple) does not assign from
    a cuda::std::pair (CCCL thrust does; rocThrust does not). Changed to return
    `thrust::tuple` on HIP (USE_HIP-guarded).
- **cuda::std::array<unsigned,100> -- NOT a real wall (the prior finding was
  stale).** strings/case.cu uses `__constant__ cuda::std::array<uint32_t,100>`
  with an aggregate initializer; the type IS fully supported by the current
  vendored ROCm/libhipcxx (a standalone __constant__ + thrust::binary_search
  probe compiles). case.cu simply never `#include`d `<cuda/std/array>` (it relied
  on CCCL pulling it transitively, which libhipcxx does not). Fix: add the
  explicit include (correct on CUDA too -- same fault class as the cuda/std/bit
  missing-transitive-include).
- **wave64 ballot in strings/copying/concatenate.cu -- converted** to the
  tile-relative `ballot_32` helper (warp_primitives.cuh), identical arch-unified
  treatment as the in-scope valid_if sites. `fused_concatenate_string_offset_kernel`
  dropped the carried `active_mask` dance (tiled_partition<32>::ballot() ballots
  only active lanes). COMPILES; this specific kernel is not yet exercised by the
  GPU test (covered by the identical, GPU-validated valid_if pattern).

### Validation gotcha burned this session (cost ~hours): OOB in the TEST, not the port

The first version of the sort gtest crashed nondeterministically (~70% SIGSEGV/
SIGABRT) -- but ONLY in the gtest binary, never in any standalone libcudf-linked
repro (sort_radix/sorted_order_radix called hundreds of times = clean). Root
cause was a HEAP BUFFER OVERFLOW in the TEST's own input helper: `mixed_floats(n)`
wrote `h[10]/h[20]/h[100]/h[200]` unconditionally, which is OOB of the
std::vector for the small sizes (n=1,33,64,65) in the size loop, smashing the
host heap -> later malloc/free or the ROCm-runtime atexit finalizer crashed.
The cudf sort + shim were CORRECT throughout (which is exactly why every
standalone repro that built its input array correctly was clean). LESSON: when a
GPU test crashes intermittently in libamdhip64/libhsa host alloc or
__cxa_finalize but a standalone repro of the same library calls is clean,
suspect the TEST harness (an OOB host write, a bad index) before the port; guard
every fixed index against the parametrized size. Do NOT paper over it with an
rmm pool or _exit() (both "fixed" the symptom by changing heap layout and masked
the real bug).

## MACRO-CASCADE VERDICT (jeff's question): the 1-line fix DID clear it

Ground truth confirmed: `CUDF_HOST_DEVICE` is DEFINED once
(`cpp/include/cudf/types.hpp:23`, gated `#ifdef __CUDACC__`) and USED at 331
sites across 43 files. Changing that ONE definition to
`#if defined(__CUDACC__) || defined(__HIPCC__)` restored `__host__ __device__`
at all use sites and **cleared the entire column_device_view.cu -> string_view
"call to __host__ from __device__" cascade (the 71-header wall) in one edit**.
Verified by compiling `cpp/src/column/column_device_view.cu` standalone as HIP:
before, the cascade; after the 1-line fix, only a SEPARATE, smaller issue
remained -- `type_dispatcher.hpp` `base_type_to_id` explicit specializations
(the `CUDF_TYPE_MAPPING` macro + the `char` specialization) are declared
WITHOUT `CUDF_HOST_DEVICE` while the primary now HAS it; clang (unlike nvcc)
rejects the host/device-attribute mismatch ("no function template matches").
Adding `CUDF_HOST_DEVICE` to those 2 specialization definitions cleared it, and
column_device_view.cu then compiled with ZERO errors. So: the macro cascade was
real and the 1-line root fix dissolved it; the prior "46 guard site cascade"
framing was an overcount (it conflated the single definition with its 331 uses
plus the unrelated nv-pragma guards). Do NOT blanket-`#define __CUDACC__` (the
prior dead-end: it wrongly activates CUDA template paths in type_dispatcher and
elsewhere).

### How the __CUDACC__ / __CUDA_ARCH__ sites were actually handled

- `CUDF_HOST_DEVICE` / `CUDF_KERNEL` definition gate (types.hpp:19): root fix
  `|| defined(__HIPCC__)`. This is the whole cascade.
- Real `__device__`-code `#ifdef __CUDACC__` guards (4 sites, genuine device
  functions): `|| defined(__HIPCC__)` so the device fn is declared on HIP --
  `utilities/bit.hpp` (set_bit), `column/column_device_view_base.cuh`
  (set_valid), `src/jit/span.cuh` (is_valid_nocheck/is_valid).
- NVCC-only `#pragma nv_exec_check_disable` guards (the OTHER ~11 `#ifdef
  __CUDACC__` sites in type_dispatcher.hpp / column_view.hpp / aggregation.hpp /
  device_scalar.hpp / quantiles_util.hpp): LEFT `__CUDACC__`-only. clang neither
  has nor needs that pragma; adding `|| __HIPCC__` would make clang choke on an
  unknown pragma. This per-site discrimination is exactly why the blanket define
  fails.
- `__CUDA_ARCH__` (30 sites, all presence/absence -- NO numeric `>= N` gates in
  cudf): the raft fault-class-1 fix -- define `__CUDA_ARCH__ 800` in the HIP
  DEVICE pass only (compat header, guarded by `__HIP_DEVICE_COMPILE__`). Every
  `#ifndef __CUDA_ARCH__` (host-only error-throw branch) and `#ifdef
  __CUDA_ARCH__` (device direct-memory branch) then selects correctly in each
  pass. No NVIDIA intrinsics are newly activated (cudf has no `__CUDA_ARCH__ >=
  N` comparisons), so no per-site `&& !defined(USE_HIP)` guards were needed
  (unlike raft, which had __dp4a/__nanosleep/PTX behind numeric gates).

## Architecture of the HIP build

- `cpp/CMakeLists.txt`: 6-line `option(USE_HIP)` + guard at the very top.
- `cpp/cmake/hip/cudf_hip.cmake`: standalone HIP build. find_package(hip),
  find_package(rmm) (carries vendored libhipcxx cuda::std/cuda::mr +
  rapids_logger + the rmm compat force-include + USE_HIP/__HIP_PLATFORM_AMD__/
  LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE defs), add_subdirectory the
  ported cuCollections with USE_HIP (gives cuco::cuco header-only INTERFACE; its
  BUILD_TESTS is force-OFF so cudf's BUILD_TESTS does not pull cuco's Catch2
  tests), NVTX3 from a vendored header dir, gtest from conda. Generates
  version_config.hpp + logger_macros.hpp (rapids-logger template; logger ns is
  `cudf::default_logger()`, NOT `cudf::detail::`). Marks the scoped sources
  LANGUAGE HIP, force-includes the cudf compat header, links rmm::rmm +
  cuco::cuco + hip::host.
- `cpp/cmake/hip/cudf_hip_sources.cmake`: the curated in-scope source list (the
  knob that grows the core). Files NOT listed are scoped out, not deleted.
- `cpp/cmake/hip/cudf_hip_tests.cmake`: builds the focused GPU gtest.
- `cpp/include/cudf/detail/hip/cuda_to_hip.h`: the cudf compat header
  (force-included on every HIP TU). Adds the 4 cuda* runtime symbols rmm's
  compat does not cover (cudaGetSymbolAddress, cudaPeekAtLastError,
  cudaStreamDefault, cudaOccupancyMaxActiveBlocksPerMultiprocessor) and the
  `__CUDA_ARCH__ 800` device-pass define. Most of cudf's cuda* runtime surface
  is already aliased by rmm's compat (force-included first via rmm::rmm).
- `cpp/hip_compat/`: forwarding shims on the HIP include path (BEFORE) --
  `cuda_runtime.h`/`cuda_runtime_api.h`/`cuda.h` -> the compat header; `cub/*`
  -> hipCUB (cub.cuh + the 12 specific cub headers cudf includes, each
  `namespace cub = hipcub;`); `cooperative_groups.h` + `cooperative_groups/
  reduce.h` -> HIP CG + a tile-relative cg::reduce shim (copied from the cuco
  port). On NVIDIA this dir is absent so the real toolkit headers win.
- `cpp/include/cudf/detail/utilities/hip/warp_primitives.cuh`: the wave64
  ballot helper (below). Included by `detail/utilities/cuda.cuh` under USE_HIP.

## Fault classes fixed (all arch-unified; NVIDIA path byte-for-byte unchanged)

1. **wave64 null-mask ballot** (the central correctness gate). cudf builds a
   32-bit validity word per warp via `__ballot_sync(0xFFFF'FFFFu, pred)` and has
   the lane-0 leader write it to `output[word_index(i)]`. On AMD this (a) does
   not COMPILE (HIP `__ballot_sync` static_asserts a 64-bit mask) and (b) is
   semantically wrong on a 64-lane wavefront (64 lanes straddle two 32-bit
   words). cudf's bitmask granularity is genuinely 32 bits/word, so the fix
   KEEPS the logical warp at 32 lanes: `ballot_32(pred)` uses
   `cg::tiled_partition<32>(block).ballot(pred)` -- tile-relative on any wave
   width (the cuCollections/gsplat lesson), so the two 32-lane tiles of a wave64
   wavefront each produce their own 32-bit word matching `word_index`. Routed at
   all in-scope sites: `detail/valid_if.cuh` (x2), `detail/copy_range.cuh`,
   `detail/copy_if_else.cuh`, `src/copying/concatenate.cu` (x2),
   `src/replace/replace.cu`, `src/replace/nulls.cu`, each `#if defined(USE_HIP)`
   guarded. `warp_size` stays 32 (do NOT bump to 64 -- it is the bitmask word
   width, not the hardware wave width). GPU-validated: see Test results.
2. **HIP cooperative_groups gaps**: `grid_group::block_rank()` does not exist in
   HIP CG -> replaced (USE_HIP) with the explicit row-major linear block index
   (`blockIdx.x + blockIdx.y*gridDim.x + ...`), identical to CUDA's value
   (null_mask.cu set_null_masks_kernel). `cg::reduce`/`cg::plus` absent -> the
   tile-relative shfl_xor shim (hip_compat/cooperative_groups/reduce.h).
3. **clang-vs-nvcc strictness** (clang stricter): (a) hash-functor explicit
   specializations must match the constexpr primary's constexpr-ness
   (murmurhash3_x86_32/x64_128, xxhash_32/64 -- added `constexpr`; the two
   unreachable list_view/struct_view specializations also need a
   `-Winvalid-constexpr` pragma + an unreachable `return`, HIP-only); (b)
   `typename` on dependent type names (`cast_ops.cu`
   `cuda::std::chrono::floor<typename TargetT::duration>`); (c) a nested type's
   forward-declaration access must match its definition (list_device_view.cuh
   pair_accessor/pair_rep_accessor -- made the definitions public to match the
   public forward decls); (d) thrust::iterator_facade hooks called from
   `__host__ __device__` helpers must themselves be host+device
   (`row_operators.cuh` strong_index_iterator increment/advance/equal/
   dereference/distance_to -> CUDF_HOST_DEVICE); (e) gather.cuh bounds_checker
   ctor was `__device__`-only but thrust constructs it host-side ->
   CUDF_HOST_DEVICE.
4. **missing transitive includes that only libhipcxx exposes**: `cuda/std/bit`
   for `cuda::std::popcount`/`countl_zero` (null_mask.cuh, null_mask.cu --
   CUDA's CCCL pulled it transitively, libhipcxx does not); assert.cuh for
   `CUDF_UNREACHABLE` (integer_utils.hpp, now reachable in the device pass since
   __CUDA_ARCH__ is defined there). All are correct on CUDA too (explicit deps).
5. **missing cub headers**: shimmed (block_load, device_histogram,
   device_merge_sort, device_radix_sort, device_segmented_sort, warp_reduce in
   addition to the initial set).

## What COMPILES + GPU-validates (historical: the 107-TU core, fork 5be894c)

(SUPERSEDED at fork bec85ee, which adds reductions + aggregation + groupby +
join on top and GPU-validates 12/12 -- see the top sections. This section is the
107-TU-core record.) (+24 over the prior 83.) The 83-TU foundational core PLUS, this session: table
row_operators + primitive_row_operators (experimental::row lexicographic/
equality machinery); stream_compaction apply_boolean_mask + distinct +
distinct_helpers + stable_distinct (cuco static_set + retrieve_all);
the full hashing set (md5, murmurhash3_x86_32/x64_128, sha1/224/256/384/512,
xxhash_32/64); search contains_column + contains_table; structs/utilities;
dictionary set_keys; lists utilities + dremel + copying/scatter_helper;
transform/encode. Plus the original 83: column, table, scalar, bitmask,
utilities, unary, copying, filling, replace, search (contains_scalar/
search_ordered), sort (incl. the rocPRIM tuple-key radix path), and the
foundational nested-type column views + factories + detail ops.

libcudf.so links with a real gfx90a code object and leaves exactly ONE
cudf-internal symbol undefined (binary_operation, the JIT path). GPU 8/8:
DistinctCucoStaticSet + ApplyBooleanMask added to the prior 6.

GPU-validation helpers added to the gtest (core_smoke_test.cu): make_int32_column
and device_int32_to_host (host<->device INT32 round-trip via rmm::device_buffer
ctor + hipMemcpyAsync).

## Scoped OUT (guarded via the source list; files NOT deleted)

- **all of src/io** (89 files: cuIO csv/json/parquet/orc/avro/text + comp) --
  home of the unportable **nvCOMP** (NVIDIA proprietary binary) and **KvikIO**
  (GPUDirect Storage). Cleanly separable (0 nvCOMP includes outside io/).
- **jitify/NVRTC JIT** kernels (binaryop-JIT, transform-JIT, rolling-JIT). The
  jitify->hiprtc plan is a separate effort (jitify_hiprtc_plan.md, owned
  elsewhere). `binaryop/binaryop.cpp` pulls jitify (15 includes).
- **reductions + aggregation + hash/sort groupby + hash/sort-merge join are NOW
  IN SCOPE** (fork bec85ee) -- they link the single .so (--offload-compress) and
  GPU-validate. STILL scoped out: full lists/strings/dictionary/structs
  algorithm sets, text, interop-to-arrow, transform (non-JIT parts), rolling,
  datetime, quantiles/tdigest, lists::distinct. These extend the API surface;
  none is gated by a structural wall now (the SCALE wall is resolved), only by
  not-yet-being-in the source list. group_quantiles is the one groupby TU
  omitted (it pulls quantiles/, scoped out) -- groupby/sort/group_quantiles.cu.

## The next concrete walls (precise blocked_reason material)

The dispatch-closure wall (prior wall 1, "~209 deferred symbols") is CLOSED to
1 (see the State header). What remains:

1. **The libcudf.so x86-64 relocation SCALE wall -- RESOLVED (fork bec85ee).**
   --offload-compress shrinks the per-TU device bundle ~55x, so reductions +
   aggregation + join + groupby all fit the single libcudf.so (~540 MB, gfx90a
   only) with no split and no -mcmodel=large; the three relocation classes never
   trigger. GPU-validated 12/12. See the top "RESOLVED" / "LINK VERDICT" /
   "GPU VALIDATION" sections.
2. then the **JIT** wall (binaryop/transform/rolling) -- the jitify->hipRTC plan
   is written (jitify_hiprtc_plan.md) and the hipRTC PoC is proven; it is a
   separate phase. binary_operation (2 of the 17 remaining undefined symbols)
   lives here. The other 15 (quantile/tdigest/repeat/tile/lists::distinct/
   strings+structs scan_inclusive) are non-JIT modules simply not yet in the
   source list -- straightforward next bring-ups, gated by nothing structural.

### cuco-backed bring-up adaptations that WORKED (reusable, in 5be894c)

- **cuco_allocator stream-aware adaptor** (detail/cuco_helpers.hpp, USE_HIP):
  the ROCm cuco port is a NEWER cuco generation than cudf v25.08 pins on CUDA,
  and drives stream-ordered storage allocation through the CCCL 2-arg
  `allocate(num, cuda::stream_ref)` / `deallocate(ptr, num, cuda::stream_ref)`
  interface; rmm's stream_allocator_adaptor exposes only the single-arg standard
  C++ form (fixed stream bound at construction). Fix: on HIP make cuco_allocator
  a thin subclass of rmm::mr::stream_allocator_adaptor<polymorphic_allocator<T>>
  that adds the 2-arg overloads (forwarding the caller's stream to the underlying
  polymorphic_allocator via rmm::cuda_stream_view{stream.get()}) while keeping the
  1-arg form via `using base::allocate/deallocate`. CUDA keeps the plain alias.
- **mixed_multimap_type modern spelling** (join/join_common_utils.hpp, USE_HIP):
  the ROCm cuco port carries only `cuco::static_multimap` (modern template order
  Key,T,Extent,Scope,KeyEqual,ProbingScheme,Allocator,Storage), NOT
  `cuco::legacy::static_multimap`. Re-spell the alias for the modern type so
  headers that pull it parse; the mixed-join TUs that instantiate it stay scoped
  out anyway (scale wall).
- **const operator() on the reduction binary functors** (device_operators.cuh
  DeviceSum/Min/Max/Product/Count; cast_functor.cuh cast_functor_fn): rocPRIM's
  DeviceReduce invokes the binary op through a const wrapper
  (convert_binary_result_type_wrapper) where CUB tolerates a non-const operator;
  const is correct on CUDA too. (Compiled fine; only the reductions LINK failed
  on the scale wall.)
- **dependent-template `::template`** (reduction_operators.cuh): clang two-phase
  lookup needs `typename Derived::template transformer<R>` /
  `Derived::template intermediate<R>` (nvcc accepts the bare form). Same family
  as the existing typename-on-dependent-type fixes.
  These four header edits are coupled to the reductions bring-up; they are
  REVERTED in 5be894c along with the reductions (kept here so the re-attempt,
  once the .so is split, does not re-derive them).

## rmm-as-a-dependency: WORKED (refreshed)

The cudf-rmm dep MUST be the CURRENT jeffdaily/rmm @ moat-port (now d22baff),
NOT the older 1473ffc: the older install exported a malformed force-include path
(`<prefix>//rmm/...`, missing `include/`, because CMAKE_INSTALL_INCLUDEDIR was
empty at export) that broke every cudf HIP TU. d22baff uses
`$<INSTALL_PREFIX>/include/rmm/...`. Recipe (per projects/rmm/notes.md):

```
# rmm dep clone updated to fork HEAD, rebuilt+installed into _deps/cudf-rmm/install
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S _deps/cudf-rmm/src/cpp -B _deps/cudf-rmm/build -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/cudf-rmm/install
cmake --build _deps/cudf-rmm/build --target install -j16
```

## Vendored deps a build needs (stable locations)

- rmm install: `_deps/cudf-rmm/install` (jeffdaily/rmm @ moat-port, built above).
- cuCollections: `_deps/cudf-cuco/src` (`git clone --depth 1 -b moat-port
  https://github.com/jeffdaily/cuCollections`). Header-only; passed via
  `-DCUDF_CUCO_SOURCE_DIR`.
- libhipcxx: `agent_space/libhipcxx/include` (ROCm/libhipcxx amd-develop).
- rapids_logger: `agent_space/rapids_logger` (rapidsai/rapids-logger 0.2.0).
- NVTX3: `agent_space/nvtx3_include` (vendored from pytorch's third_party/NVTX;
  pure host C++, no CUDA device content). Passed via `-DCUDF_NVTX3_INCLUDE_DIR`.

## Build script (lead, gfx90a)

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/cudf/src/cpp -B projects/cudf/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/cudf-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DCUDF_CUCO_SOURCE_DIR=/var/lib/jenkins/moat/_deps/cudf-cuco/src \
  -DCUDF_NVTX3_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/nvtx3_include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=ON
cmake --build projects/cudf/build-hip -j 16   # CAP at -j16: C++20 templates are
                                              # memory-hungry; OOM risk otherwise
```

Arch flows from `CMAKE_HIP_ARCHITECTURES` (defaults to gfx90a only when unset),
so a follower (gfx1100/gfx1151) builds the scoped core with only
`-DCMAKE_HIP_ARCHITECTURES=<arch>` and no source/CMake edit.

## Validation (gfx90a / MI250X, ROCm 7.2.1; run on a FREE GCD)

```
# pick a free GCD from `rocm-smi` (4 GCDs 0-3 on this box; siblings may use
# 1/2). The bec85ee broad-surface run used GCD 0 and passed 12/12.
HIP_VISIBLE_DEVICES=0 ctest --test-dir projects/cudf/build-hip --output-on-failure
# (LD_LIBRARY_PATH must include $CONDA_PREFIX/lib for libspdlog and
#  _deps/cudf-rmm/install/lib for librmm/librapids_logger -- sourced via
#  agent_space/cudf_build_env.sh)
```

libcudf.so built with a gfx90a code object (roc-obj-ls:
`hipv4-amdgcn-amd-amdhsa--gfx90a`). **MOAT_CORE_SMOKE_TEST: 6/6 PASS** on real
gfx90a (stable over 20 repeated runs; use a free GCD -- rocm-smi showed 0,1,2,3,
this session used HIP_VISIBLE_DEVICES=2):
- `CountSetBitsWave64`: device `count_set_bits` over known masks == host count,
  sizes {1,31,32,33,63,64,65,127,128,129,1000,4096,100000} (crossing 32-bit-word
  AND 64-lane-wavefront boundaries). The wave64 ballot gate.
- `SegmentedCountSetBits`: segmented reduce over uneven segments == host.
- `CreateNullMask`: ALL_VALID -> n set bits, ALL_NULL -> 0, on device.
- `CountIsDeterministic`: 3x repeated device count bit-identical.
- `SortRadixFloatTupleKeyShim`: `detail::sort_radix` on no-null FLOAT32 columns
  (the decomposer path through the rocPRIM tuple-key shim), asc AND desc, device
  result == host std::sort, sizes 1..100000. The tuple-key shim gate.
- `SortedOrderRadixFloatTupleKeyShim`: `detail::sorted_order_radix` produces the
  sort permutation; gathering the input by it == host std::sort (the SortPairs
  decomposer path + the thrust::tuple zip-iterator functor).

The smoke test links libcudf.so (cudf is a SHARED lib) with
`--unresolved-symbols=ignore-in-shared-libs` + `-z lazy`: the .so carries the
deferred dispatch-closure symbols (above), the test never calls them, and this
linkage tolerates unresolved symbols FROM the .so while still erroring on any
unresolved symbol in the test's own objects (tighter than the old ignore-all). A
broader GPU suite (gather/groupby/join/etc.) needs the remaining link-closure
(wall 1) resolved first.

## Install as a dependency

cudf is consumed by cugraph/cuml. cudf_hip.cmake (fork c4b9b5b) exports the
`find_package(cudf)` contract: `install(TARGETS cudf EXPORT cudf-exports)` ships
the HIP-built libcudf.so + the public `include/cudf` tree + the two generated
headers (version_config.hpp, logger_macros.hpp), and a generated
`cudf-config.cmake` (`find_dependency(hip)` + `find_dependency(rmm)`, then the
exported `cudf::cudf` target) lets a downstream resolve the ROCm build via
`-DCMAKE_PREFIX_PATH`. The install layout: `lib/libcudf.so`,
`lib/cmake/cudf/{cudf-config.cmake,cudf-config-version.cmake,cudf-targets.cmake,
cudf-targets-release.cmake}`, `include/cudf/...`.

Recipe (build + install into _deps/cudf/install). Build rmm first (see the rmm
recipe above), then:

```
export CONDA_PREFIX=/opt/conda/envs/py_3.12
cmake -S projects/cudf/src/cpp -B projects/cudf/build-hip -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/cudf-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DLIBHIPCXX_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/libhipcxx/include \
  -DCUDF_CUCO_SOURCE_DIR=/var/lib/jenkins/moat/_deps/cudf-cuco/src \
  -DCUDF_NVTX3_INCLUDE_DIR=/var/lib/jenkins/moat/agent_space/nvtx3_include \
  -DRAPIDS_LOGGER_SOURCE_DIR=/var/lib/jenkins/moat/agent_space/rapids_logger \
  -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/cudf/install
cmake --build projects/cudf/build-hip --target install -j 16
```

A dependent consumes it with `find_package(cudf REQUIRED)` and links
`cudf::cudf`, passing `-DCMAKE_PREFIX_PATH=".../_deps/cudf/install;
.../_deps/cudf-rmm/install;/opt/rocm;$CONDA_PREFIX"`. The cudf-including TU MUST
be compiled as HIP (the PUBLIC cudf headers pull rocPRIM device intrinsics, e.g.
`__builtin_amdgcn_wavefrontsize`, which a plain-CXX TU cannot compile). The
installed libcudf.so RUNPATH is `$ORIGIN:/opt/rocm/lib` and NEEDED is librmm.so +
librapids_logger.so + the HIP runtime, so it loads with rmm's install/lib on the
path (0 unresolved).

CONSUMABILITY VERIFIED (fork c4b9b5b): a standalone `find_package(cudf)` project
(agent_space/cudf_consumer) configured (resolved cudf via cudf-config ->
find_dependency hip+rmm), compiled `main.cpp` as HIP, linked cudf::cudf
(ldd resolves libcudf.so + librmm.so + librapids_logger.so), and RAN on gfx90a:
`cudf::make_numeric_column` returned an INT32 column of size 8, exit 0.

NOTE for cugraph/cuml: libcudf.so still carries the documented scoped-out
deferred-dispatch symbols (the 2 binary_operation JIT overloads, and -- because
the strings/structs segmented-scan and a few other modules are header-declared
but their .cu is scoped out -- symbols like strings/structs
`scan_inclusive<DeviceMin/Max>` that a downstream HIP TU may instantiate from the
headers). A downstream that does not call them links cleanly with
`--unresolved-symbols=ignore-in-shared-libs -z lazy` (the same option the smoke
test uses). To call them, bring up the corresponding module in
cudf_hip_sources.cmake (none is structurally blocked; only the 2 binaryop JIT
overloads need the jitify->hipRTC effort).
