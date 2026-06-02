# cuvs notes (ROCm/HIP port)

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated
RAPIDS). cuvs is the next RAPIDS layer after rmm + raft; cuml depends on cuvs.
The `## Install as a dependency` section is the contract cuml consumes.

Pinned upstream: tag `v25.08.00` (commit 9ce11a0f, in `projects/cuvs/src`,
gitignored). CRITICAL version pin: upstream `main` is 26.08.00, but the ported
rmm and raft are 25.08.00 and cuvs CMake requires an exact-major.minor raft. cuvs
MUST stay at v25.08.00 so it links our 25.08 rmm/raft ABI.

depends_on: **rmm** (completed) + **raft** (completed). cuvs consumes both via
find_package against the installed `_deps/raft/install` (raft) and
`_deps/raft-rmm/install` (rmm); raft::raft transitively pulls rmm::rmm, and both
force-include their cuda_to_hip.h on every HIP TU. The installed raft has
`RAFT_COMPILE_LANCZOS=OFF` (the B6 lanczos fix is staged, not on the fork), so
cuvs is built WITHOUT the spectral/lanczos path (see Deferred).

## Existing AMD support

None. No ROCm/HIP branch or fork upstream. NVIDIA-only (CUDA + CUTLASS +
cuBLAS/cuSOLVER/cuSPARSE/cuRAND). From-scratch CUDA-to-HIP, Strategy A.

## Build classification: Strategy A (pure-CMake RAPIDS library)

cuvs is a pure-CMake RAPIDS library (rapids-cmake bootstrap + CPM, same shape as
rmm/raft), NOT a torch extension. Strategy A: a top-of-file `option(USE_HIP)` +
early `include(cmake/hip/cuvs_hip.cmake); return()` guard bypasses the
rapids-cmake/CPM bootstrap (which would fetch NVIDIA CCCL/CUTLASS/raft/rmm). The
standalone HIP build resolves rmm + raft via find_package, marks the stock `.cu`
LANGUAGE HIP (never renamed), and force-includes a cuVS compat header on HIP TUs.

## Scope delivered (lead, gfx90a): the DISTANCE slice

This pass delivers and GPU-validates the cuVS **pairwise-distance subsystem**
(`cuvs::distance::detail`, the CUTLASS->SIMT slice), the proven raft DISTANCE
template re-namespaced into cuVS. The library is brought up in stages via the
`CUVS_HIP_SLICE` cache var (mirroring raft's RAFT_TEST_*/RAFT_COMPILE_LANCZOS
gating) so a not-yet-ported stage never blocks the validated slice. The shipped
slice (`CUVS_HIP_SLICE=distance`) compiles 43 distance `.cu` into libcuvs.so:
every pairwise-distance metric x dtype (float/half/double, int/int64 index) +
fused-distance-NN + the public `distance.cu`/`pairwise_distance.cu`.

CUTLASS does NOT port to ROCm (PORTING_GUIDE/jeff). On HIP the distance + fused-NN
dispatchers force the legacy SIMT/sm60 path -- mathematically identical to the
CUTLASS path, and exactly the fallback cuVS uses on pre-Ampere NVIDIA. (A CK MFMA
fast path, as raft delivered via dispatch_ck.cuh, is a documented Layer-B
follow-on; Layer A SIMT is correct and is the validated deliverable here.)

### Build infrastructure (new files, HIP-only; CUDA path byte-identical)

- `cpp/CMakeLists.txt`: the ONLY edit to the stock file -- a top-of-file
  `option(USE_HIP)` + `if(USE_HIP) include(cmake/hip/cuvs_hip.cmake) return()`
  before the rapids includes.
- `cpp/cmake/hip/cuvs_hip.cmake` (new): standalone HIP build. enable_language(HIP);
  arch from CMAKE_HIP_ARCHITECTURES (defaulted gfx90a only when unset);
  find_package(rmm)+find_package(raft); regenerates version_config.h; builds the
  CUVS_HIP_SLICE source group marked LANGUAGE HIP; force-includes the cuVS compat
  header; `--offload-compress` + -ffunction/-fdata-sections + --gc-sections (the
  cudf large-fatbin x86-64 relocation lever); exports cuvs::cuvs + a minimal
  cuvs-config.cmake.
- `cpp/cmake/hip/cuvs_hip_tests.cmake` (new): per-group CUVS_TEST_* options;
  builds DISTANCE_TEST from the pure pairwise-distance metric tests.
- `cpp/include/cuvs/util/hip/cuda_to_hip.h` (new): the cuVS compat force-include.
  Minimal -- cuVS's CUDA spellings are a subset of raft's, so rmm's + raft's
  compat headers (force-included ahead of this via rmm::rmm/raft::raft) cover the
  runtime + fp16/bf16 + math-lib + __CUDA_ARCH__ surface, and raft's hip_compat/
  shim dir redirects cuVS's `#include <cublas_v2.h>`/<cub/...>/<cooperative_groups.h>.

### Guarded source edits (all `#if defined(USE_HIP)`; CUDA path byte-identical)

- `pairwise_matrix/dispatch_sm80.cuh`: `#if !defined(USE_HIP)` the cutlass include
  + body; provide an unreachable `pairwise_matrix_sm80_dispatch` stub on HIP (the
  forward-decl in dispatch-inl.cuh and the generated `.cu` still include it).
- `pairwise_matrix/dispatch-inl.cuh`: `constexpr bool cutlass_op_unavailable =
  true` on HIP -> always take the SIMT/sm60 path (kernel_virtual_arch reads
  ptxVersion, meaningless on HIP).
- `pairwise_matrix/kernel_sm60.cuh`: `#if !defined(USE_HIP)` the
  `SM_compute_arch()` runtime-arch early-exit. raft's SM_compute_arch::value() is
  __device__-only and static_asserts (P2593 dummy) in the host pass, but
  clang-as-hipcc parses this __global__ body in BOTH passes -> the assert fires at
  compile time. cuVS targets a single HIP arch so the sm60 kernel is always the
  one launched; skipping the early-exit is correct and arch-unified.
- `distance_ops/l_inf.cuh`: `acc = raft::max(acc, static_cast<AccT>(diff))`.
  raft::max has no (float, __half) host-pass overload (only `#ifdef __CUDA_ARCH__`)
  and static_asserts; clang parses this DI body in the host pass. Casting to AccT
  matches the device-pass __half2float conversion -> value-identical on CUDA.
- `fused_distance_nn.cuh`, `fused_distance_nn/{fused_l2_nn,fused_cosine_nn,
  helper_structs}.cuh`: `#if !defined(USE_HIP)` the `cutlass_base.cuh` includes
  (which pull <cuda/semaphore> + <cutlass/...>) and the cutlass dispatch branch;
  force the SIMT fused-NN kernel on HIP (same fallback cuVS uses below SM_80).
- `fused_distance_nn/simt_kernel.cuh`: two HIP edits. (1) `KVPair{static_cast<
  KVPair::Key>(tmpkey), acc}` -- clang rejects the unsigned->int narrowing in the
  braced init that nvcc only warns on (the same narrowing class raft fixed in its
  simt_kernel). (2) the body guard is `#if (__CUDA_ARCH__ < 800) ||
  defined(USE_HIP)`: the dispatcher forces this SIMT kernel on HIP, but raft's
  compat header pins `__CUDA_ARCH__` to 800 on the device pass, so the bare
  `< 800` guard compiled the body OUT and shipped an EMPTY kernel (uninitialized
  fusedDistanceNN output -- the cupoch/RXMesh `__CUDA_ARCH__` trap). With the
  body live, the SIMT per-row mutex is acquired one lane at a time
  (`lid == j*AccThCols`) and the shuffles use raft::WarpSize, so it is
  wave-agnostic (correct on wave32 and wave64). Validated on gfx90a by the new
  FUSED_NN_TEST (argmin + L2 relu-ALWAYS epilogue vs a CPU reference).

## wave64 audit (distance slice)

The distance slice rides raft's wave-agnostic primitives: the SIMT pairwise
GEMM (PairwiseDistances in pairwise_distance_base.cuh) and the SIMT fused-NN
kernel use `raft::WarpSize` / `raft::shfl(width=AccThCols)`, and the per-row L2
norms feed through raft's coalescedReduction (the active_mask() fix already
landed in raft). cuVS has NO warp-size constant of its own (no CMake-injected
warp def, no hardcoded wave width in the compiled slice -- the only literal-32
shuffles/ballots live in the CUTLASS persistent-GEMM and sparse paths, both
`#if !defined(USE_HIP)` guarded out), so building against the fixed raft
(ce0fa68c, whose HOST raft::WarpSize is a runtime hipDeviceGetAttribute query)
auto-inherits the multi-arch fix at every host launch site (kmeans_balanced
block_dim, ivf_pq_build, vpq_dataset, scann_quantize, haversine). This was
verified by a combined gfx90a;gfx1100 build (both code objects in libcuvs.so).
No `__ballot(0xffffffff)`, no hand-rolled bitonic in the compiled slice. The
wave64 headline risks (CAGRA team-size, NN-Descent bitonic, FAISS select /
ball_cover) live in the DEFERRED neighbors stages below, not in distance.

## Deferred (documented, off the validated slice) -- the staged remainder

cuVS is a LARGE port; a fully-validated cuvs (all algorithms on wave64) re-treads
every raft wall plus adds CAGRA/NN-Descent wave64 work raft never had. This pass
delivers the distance foundation (Stage 1) that the ANN algorithms consume, and
documents the remainder behind `CUVS_HIP_SLICE` so the next pass extends it
incrementally. Deferred stages (each its own wave64/port effort):

- **CK MFMA fast path** (Layer B): port raft's validated dispatch_ck.cuh +
  dispatch_fused_nn_ck.cuh into cuvs's namespace to back L2/cosine on the aligned
  fp32 regime (raft's DistanceCDEOp/DistanceFusedCDEOp; gate
  `is_xdl_wmma_supported<float,float,32,32>` for the gfx1100 follower). The SIMT
  slice shipped here is the correctness fallback the CK path falls through to.
- **kernels/gram** (gram_matrix/kernel_matrices/kernel_factory): hipBLASLt GEMM +
  the RBF kernel. dispatch_rbf.cu IS in the slice (its instantiation is referenced
  by pairwise_distance.cu); gram_matrix.cu (the public GramMatrix API + masked_nn
  + sparse_distance + cross_component_nn tests) is the next distance sub-stage.
- **brute-force / IVF-Flat / IVF-PQ** (Stage 2/3): consume raft neighbors
  (validated) + cuVS's own FAISS select copies. IVF-PQ rides raft's wave-correct
  sub-warp primitives (SOFTER); brute-force needs cuVS's faiss select wave64
  (reuse raft's faissWarpQ/WarpMask/maskless-shuffle FAISS-ROCm port).
- **CAGRA** (Stage 4, the headline new wave64 risk): device_common.hpp
  `constexpr warp_size=32`, the dim128_t8/dim256_t16/dim512_t32 team-size
  templates, and __ballot/__shfl in the graph-search kernels assume a 32-lane
  warp. CUDA_SEPARABLE_COMPILATION (RDC = -fgpu-rdc) cagra-search static lib.
- **NN-Descent** (Stage 4): static_assert(warp_size==32) + hand-unrolled 32-lane
  bitonic + bfe.u32 PTX.
- **ball_cover / sparse / SCANN / MG**: ball_cover is raft's open wave64 blocker
  (FAISS KeyValueBlockSelect); sparse pulls cuco + CUTLASS-sparse; MG needs RCCL.
- **spectral / lanczos**: gated on raft RAFT_COMPILE_LANCZOS (the B6 rocSOLVER
  stedc convergence wall, staged in agent_space/raft_lanczos, not on the raft
  fork). Build cuVS WITHOUT spectral. Deferred spectral surface in cuvs:
  src/preprocessing/spectral/spectral_embedding.cu, src/embed/spectral.cu,
  src/cluster/single_linkage* (the spectral clustering / spectral_embedding path).

## Build (lead, gfx90a) -- repeatable

Prereqs: ported rmm at `_deps/raft-rmm/install`, ported raft at
`_deps/raft/install` (both already present on this host); conda env py_3.12
(gtest/gmock/spdlog/fmt). Build scratch is kept in agent_space/ (gitignored), not
the MOAT repo root.

cuvs MUST be built against the FIXED raft (jeffdaily/raft @ moat-port ce0fa68c,
the multi-arch keystone): reinstall raft multi-arch first (see raft notes
"## Install as a dependency"), then point cuvs at `_deps/raft/install`. The
installed raft header must show `host_warp_size()` (the runtime warp query); a
stale pre-ce0fa68c install ships a compile-time host WarpSize and is wrong for a
multi-arch build.

```bash
export CONDA_PREFIX=/opt/conda/envs/py_3.12
export HIP_VISIBLE_DEVICES=2   # pin one GCD; others may run other agents
cmake -S projects/cuvs/src/cpp -B agent_space/cuvs_build -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/raft/install;/var/lib/jenkins/moat/_deps/raft-rmm/install;/opt/rocm;$CONDA_PREFIX" \
  -DBUILD_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/cuvs/install
bash utils/timeit.sh cuvs compile -- \
  ninja -C agent_space/cuvs_build -j32 cuvs DISTANCE_TEST FUSED_NN_TEST
# both code objects in the lib:
llvm-objdump --offloading agent_space/cuvs_build/libcuvs.so | grep -io 'gfx[0-9a-z]*' | sort -u
```

Arch is read from CMAKE_HIP_ARCHITECTURES (defaulted to gfx90a only when unset),
so a follower validates with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` and no source
edit. Followers (gfx1100/gfx1151) will also need the raft rocPRIM 4.2.0 DPP
workaround (ROCPRIM_DISABLE_DPP=1 on RDNA) inherited via raft, plus -- once the CK
Layer-B lands -- the `is_xdl_wmma_supported` gfx11 gate.

## Validation (gfx90a, GCD 2)

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/var/lib/jenkins/moat/_deps/raft/install/lib:/var/lib/jenkins/moat/_deps/raft-rmm/install/lib:/opt/rocm/lib:$LD_LIBRARY_PATH"
HIP_VISIBLE_DEVICES=2 bash utils/timeit.sh cuvs test -- agent_space/cuvs_build/gtests/FUSED_NN_TEST
HIP_VISIBLE_DEVICES=2 bash utils/timeit.sh cuvs test -- agent_space/cuvs_build/gtests/DISTANCE_TEST --gtest_filter='-BigMatrix*'
```

DISTANCE_TEST exercises the PUBLIC `cuvs::distance::pairwise_distance` for every
metric x dtype against a CPU reference. FUSED_NN_TEST exercises the PUBLIC
`cuvs::distance::fusedDistanceNNMinReduce<float, KeyValuePair<int,float>, int>`
(the forced-SIMT kernel) and checks both the argmin index and the L2 relu-ALWAYS
epilogue distance against an independent CPU reference, using splitmix64-hashed
embeddings (no periodic structure -> no exact-tie false negatives). A body-less
kernel leaves the output at the init maxVal and fails immediately.

RESULTS (gfx90a, MI250X, GCD 2, ROCm 7.2.53211, built against fixed raft
ce0fa68c, multi-arch gfx90a;gfx1100):
- **FUSED_NN_TEST: 7/7 PASS, 0 FAILED.** Proves the fused-NN kernel now has a
  real body on HIP (the `__CUDA_ARCH__<800` guard fix) and that its argmin +
  L2 epilogue numerics are correct. AMD_LOG_LEVEL=3 shows the live
  `cuvs::distance::detail::fusedDistanceNNkernel<float, KeyValuePair<int,float>,
  int, ...>` launching on the native gfx90a code object
  (amdgcn-amd-amdhsa--gfx90a:sramecc+, Direct Dispatch).
- **DISTANCE_TEST `--gtest_filter='-BigMatrix*'`: 410/410 PASS, 0 FAILED**
  across 48 suites -- canberra, correlation, cosine, hamming, hellinger,
  inner-product, jensen-shannon, kl-divergence, l1, l2-expanded (+ self-distance
  X==Y), l2-sqrt-expanded, l-inf, lp-unexpanded, russell-rao, each in float +
  half + double. The slow `BigMatrix*` stress cases were excluded from the timed
  gate (each large L2 case is ~48s on the forced-SIMT path); a prior full-suite
  run confirmed the BigMatrix EucExp cases also pass.
- libcuvs.so carries BOTH gfx90a and gfx1100 code objects (llvm-objdump
  --offloading), so the multi-arch build is verified.

The l_inf static_cast fix, the forced-SIMT pairwise-distance path, and the now-
live forced-SIMT fused-NN path are all exercised on real gfx90a and correct.

## Install as a dependency

This is the contract cuml consumes. A dependent gets the cuVS headers + libcuvs.so
and (transitively) raft + rmm. The dependent's CMAKE_PREFIX_PATH must include the
cuvs, raft, and rmm install prefixes.

```bash
# after rmm + raft are installed (see their notes), install cuvs:
cmake --install agent_space/cuvs_build   # -> _deps/cuvs/install
```

Consume it:
```
find_package(cuvs REQUIRED)             # find_dependency(raft rmm hip hipblas hipsparse)
target_link_libraries(<tgt> PRIVATE cuvs::cuvs)
# CMAKE_PREFIX_PATH must list cuvs + raft + rmm + /opt/rocm + $CONDA_PREFIX
```
The dependent must enable_language(HIP) and compile cuVS-including TUs as HIP
(cuVS headers reach rocThrust/hipCUB/libhipcxx device headers). At runtime put the
cuvs, raft, rmm, and conda lib dirs on LD_LIBRARY_PATH. NOTE: the installed cuVS
is the DISTANCE slice; cuml code paths that need IVF/CAGRA/brute-force must wait
for those stages (fence them as cugraph/cuml fence cuvs-dependent code), exactly
as cuvs itself is built without spectral while raft lanczos is gated.

## Review 2026-06-02 (reviewer, linux-gfx90a) -- CHANGES REQUESTED

Reviewed moat-port HEAD 728265cc against the v25.08.00 base (9ce11a0f) with the
/pr-review skill, ROCm-fault-class aware. Scope: the delivered distance slice
(CUVS_HIP_SLICE=distance). Verified empirically on real gfx90a (GCD 1): rebuilt
nothing, re-ran the existing DISTANCE_TEST -- 410/410 PASS across 48 suites,
exit 0; device dispatch confirmed via AMD_LOG_LEVEL=3 (native
amdgcn-amd-amdhsa--gfx90a:sramecc+ code object, live
cuvs::distance::detail::pairwise_matrix_kernel on device, checked against the
naive CPU reference kernel). The tested public pairwise-distance API is correct.

### Fault Classes -- BLOCKER (silently-wrong shipped symbol)

cpp/src/distance/detail/fused_distance_nn/simt_kernel.cuh:86 -- the SIMT
fused-NN kernel `fusedDistanceNNkernel` (declared line 70) wraps its ENTIRE body
(lines 86-184) in `#if __CUDA_ARCH__ < 800`. raft's force-included compat header
defines `__CUDA_ARCH__ 800` on the HIP device pass
(_deps/raft/install/include/raft/util/hip/cuda_to_hip.h:53-54), so `800 < 800`
is false and the kernel body is compiled OUT on HIP -- the device kernel is
empty. I confirmed this directly: compiling a probe with `__CUDA_ARCH__=800`
drops the guarded store (no `v_mov_b32 v2, 0x6f`), while `__CUDA_ARCH__=700`
emits it. This is the cupoch/RXMesh `__CUDA_ARCH__` trap in PORTING_GUIDE.

The porter's forced-SIMT strategy walked into this: fused_l2_nn.cuh /
fused_cosine_nn.cuh / fused_distance_nn.cuh now guard out CUTLASS and force the
SIMT branch, but the SIMT branch it falls back to has no body on HIP. The host
launcher `fusedDistanceNNImpl` still launches the (empty) kernel, so
`cuvs::distance::fusedDistanceNNMinReduce` / `fusedDistanceNN` (public API,
instantiated by src/distance/detail/fused_distance_nn.cu which IS in the shipped
slice, cmake/hip/cuvs_hip.cmake:142) leave their output uninitialized -- silently
wrong, no compile error, no test failure. Consumers: kmeans
(cluster/detail/kmeans_common.cuh, kmeans_balanced.cuh; deferred stage) and the
public fused_distance_nn-inl.cuh API. The TESTED public pairwise path is NOT
affected -- pairwise_matrix_kernel in kernel_sm60.cuh has no `<800` body guard,
which is why DISTANCE_TEST is genuinely 410/410.

Fix, pick one and re-validate on gfx90a:
- (a) Make the forced SIMT fused-NN path actually live: extend the guard to
  `#if __CUDA_ARCH__ < 800 || defined(USE_HIP)` (or equivalent) so the body
  compiles on HIP, then add masked_nn.cu / a FusedL2NN test to the slice and
  show it passing (recall/argmin against a CPU reference; per PORTING_GUIDE use
  a splitmix-hashed embedding, never a periodic generator, to avoid exact-tie
  false negatives). This is the same `l2_exp_distance_op::epilog` (relu ALWAYS,
  optional sqrt) vs pairwise's `l2_exp_cutlass_op` distinction raft's review
  flagged -- validate the argmin numerics, not just that it runs.
- (b) Honestly defer fused-NN: drop src/distance/detail/fused_distance_nn.cu
  from CUVS_DISTANCE_SOURCES so the shipped libcuvs.so does not export a
  silently-broken public symbol, and move fused-NN to the deferred list.

### Notes accuracy -- must fix (inaccurate fault-class analysis)

notes.md:96 ("The SIMT fused-NN ... is wave64-correct by construction") and
notes.md:190-191 ("The l_inf static_cast fix and the forced-SIMT
fused-NN/distance path are exercised and correct") are both false for fused-NN:
it is NOT exercised by DISTANCE_TEST (cuvs_hip_tests.cmake:59-61 explicitly
excludes masked_nn) and its kernel is empty on HIP, so it is not correct. The
wave64 reasoning about the per-row mutex (updateReducedVal walking lanes via
raft::WarpSize) is sound IN PRINCIPLE, but it is moot while the body is gated
out. Correct these claims to match whichever fix above is chosen.

### Verified correct (no action; recorded so the porter need not re-investigate)

- l_inf.cuh:54 cast: raft::max(float,__half) has no host-pass overload
  (math.hpp:421 static_asserts is_same_v); the device pass converts via
  __half2float (math.hpp:401-403). static_cast<AccT>(diff) is value-identical on
  CUDA. Correct.
- dispatch_sm80.cuh HIP stub: signature matches the dispatch-inl.cuh forward
  decl (line 58); the `else` branch calling it is discarded by
  `if constexpr(cutlass_op_unavailable==true)` on HIP, so the stub is never
  instantiated. Correct.
- kernel_sm60.cuh:45 early-exit skip: SM_compute_arch() static_asserts in the
  host pass; single HIP target arch so the sm60 kernel is always launched.
  Correct, arch-unified.
- simt_kernel.cuh:115 KVPair narrowing static_cast: `tmpkey < n` still uses the
  uncast value; the cast only feeds the KVPair. Value-identical. (Lives inside
  the gated-out body, so currently dead on HIP -- becomes live under fix (a).)
- CUTLASS handling: zero cutlass symbols in the built libcuvs.so; all
  cutlass/cute includes USE_HIP-guarded; CUDA path byte-identical (all edits
  `#if defined(USE_HIP)` / `#if !defined(USE_HIP)`).
- Build wiring: USE_HIP option default OFF, early include+return() before the
  rapids bootstrap; enable_language(HIP); arch from CMAKE_HIP_ARCHITECTURES
  (gfx90a default only when unset); find_package(rmm/raft) not CPM; offload-
  compress + gc-sections present.
- Commit hygiene: title "[ROCm] Port cuVS pairwise-distance subsystem to HIP
  (Strategy A)" (54 chars); body names Claude, has a Test Plan, no noreply
  trailer, no ghstack; ASCII clean, no em-dash. Fork main (ee7fada6) == upstream
  rapidsai/cuvs main HEAD (clean mirror, no MOAT commits); moat-port == local
  HEAD; Actions disabled on the fork. The v25.08.00 base pin is intentional and
  preserved.

Verdict: Request Changes. The blocker is a shipped public symbol that is
silently wrong on HIP, plus notes claiming that path is validated when it is
not. Resolve fused-NN (fix or honest deferral), correct the notes, re-validate,
then bounce back for re-review.

## Review 2026-06-02 (second reviewer, linux-gfx90a) -- CONFIRMS CHANGES REQUESTED

Independent second review of moat-port HEAD 728265cc vs v25.08.00 base
(9ce11a0f) on real gfx90a (GCD 0). State was already changes-requested from the
concurrent first review; this run independently re-verifies that finding rather
than duplicating it.

Re-ran DISTANCE_TEST (`--gtest_filter='-BigMatrix*'`) on GCD 0:
410/410 PASS across 48 suites, exit 0; libcuvs.so confirmed to carry native
gfx90a code objects (llvm-objdump --offloading shows
hipv4-amdgcn-amd-amdhsa--gfx90a bundles). The tested pairwise-distance public
API is genuinely correct.

CONFIRMED BLOCKER (independently reproduced): the first review is right.
cpp/src/distance/detail/fused_distance_nn/simt_kernel.cuh:86 wraps the entire
`fusedDistanceNNkernel` body in `#if __CUDA_ARCH__ < 800` (closing #endif at
line ~184). raft's force-included compat header defines `__CUDA_ARCH__ 800` on
the HIP device pass (_deps/raft/install/include/raft/util/hip/cuda_to_hip.h:53-54),
so `800 < 800` is false and the kernel body is compiled OUT -- the device kernel
is empty on HIP. The forced-SIMT edits in fused_l2_nn.cuh / fused_cosine_nn.cuh /
fused_distance_nn.cuh guard out CUTLASS and fall into this body-less SIMT branch.
src/distance/detail/fused_distance_nn.cu (in the shipped slice,
cmake/hip/cuvs_hip.cmake:142) instantiates the PUBLIC
cuvs::distance::fusedDistanceNNMinReduce (float/int, float/int64,
KeyValuePair-out) -- so libcuvs.so exports a silently-wrong symbol (uninitialized
output, no compile error, no test failure). DISTANCE_TEST does not exercise it
(cuvs_hip_tests.cmake excludes masked_nn), which is why 410/410 still passes.
This is the cupoch/RXMesh `__CUDA_ARCH__` trap in PORTING_GUIDE.

The two fixes the first review proposed both stand:
- (a) extend the guard to `#if __CUDA_ARCH__ < 800 || defined(USE_HIP)` so the
  SIMT body compiles on HIP, then add a fused-NN/masked_nn test to the slice and
  validate the argmin numerics on gfx90a (splitmix-hashed embedding, not a
  periodic generator; the l2_exp_distance_op::epilog relu-ALWAYS vs pairwise's
  l2_exp_cutlass_op distinction must be checked, not just that it runs); or
- (b) drop src/distance/detail/fused_distance_nn.cu from CUVS_DISTANCE_SOURCES
  so libcuvs.so does not export the broken public symbol, and move fused-NN to
  the deferred list.

Also fix the inaccurate claims at notes.md:96 and notes.md:190-191 (the
forced-SIMT fused-NN path is NOT exercised and is NOT correct on HIP) per
whichever fix is chosen.

Everything else in the port verified clean: Strategy A applied correctly (top-of
-file USE_HIP guard before any rapids include -> standalone cuvs_hip.cmake;
find_package(rmm/raft) against the installed forks, no CPM/CUTLASS fetch); arch
read from CMAKE_HIP_ARCHITECTURES (gfx90a default only when unset, follower-safe);
the l_inf static_cast<AccT> fix is value-identical on CUDA (verified against
raft/core/math.hpp:385-428: host pass lacks a (float,__half) overload and
static_asserts; the cast matches the device-pass __half2float); the simt_kernel
KVPair::Key static_cast is the exact implicit conversion nvcc performs; the
pairwise dispatch-inl forces sm60 with the full SM range and never reaches the
sm80 stub; commit hygiene clean ([ROCm] title 56 chars, mentions Claude, no
noreply trailer, no ghstack, no em-dash); fork main is a clean upstream mirror;
Actions disabled on the fork. The fused-NN __CUDA_ARCH__ trap is the only
blocker.

## Review 2026-06-02 (third reviewer, linux-gfx90a) -- REVIEW PASSED

Re-review of moat-port HEAD 0c2b709a (delta from the prior reviewed 728265cc) against base v25.08.00 (9ce11a0f) with the /pr-review skill on real gfx90a (GCD 2). The delta since the two prior CHANGES-REQUESTED reviews is exactly the fix and nothing else (git diff 728265cc..0c2b709a: simt_kernel.cuh guard, new tests/distance/fused_l2_nn_simt.cu, cuvs_hip_tests.cmake wiring).

The prior blocker is resolved. simt_kernel.cuh:90 guard is now `#if (__CUDA_ARCH__ < 800) || defined(USE_HIP)`, so the SIMT fusedDistanceNNkernel body is live on HIP. Verified empirically on a FRESH rebuild (touched the changed sources, relinked libcuvs.so + FUSED_NN_TEST -- compiled clean, only pre-existing nodiscard warnings):
- FUSED_NN_TEST: 7/7 PASS, deterministic (3 runs). The test exercises the public cuvs::distance::fusedDistanceNNMinReduce<float,KVP,int> and asserts BOTH the argmin index and the L2 relu-ALWAYS epilogue distance vs an independent CPU reference (splitmix64-hashed inputs). It genuinely catches a body-less kernel: initOutBuffer=true -> initKernel sets out->value = FLT_MAX (helper_structs.cuh:87-90), so an empty kernel leaves FLT_MAX and EXPECT_NEAR fails catastrophically vs the small reference distance.
- AMD_LOG_LEVEL=3: live cuvs::distance::detail::fusedDistanceNNkernel<float, KeyValuePair<int,float>, int, KernelPolicy<...>, ...> dispatched on the native amdgcn-amd-amdhsa--gfx90a:sramecc+ code object (Direct Dispatch). A real body, not the body-less empty kernel that previously shipped.
- DISTANCE_TEST --gtest_filter='-BigMatrix*': 410/410 PASS across 48 suites (the CUDA path / pairwise slice unaffected).
- Multi-arch: built -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"; llvm-objdump --offloading on the freshly-linked libcuvs.so shows BOTH gfx90a and gfx1100 code objects.

Wave64 correctness of the now-live SIMT kernel re-verified: the kernel's reduction shuffle (raft::shfl width=P::AccThCols, a sub-warp width) and the per-row mutex walk (`raft::WarpSize / P::AccThCols` lanes, one lane at a time) both derive from raft's device-pass raft::WarpSize, which the installed multi-arch raft (ce0fa68c) resolves to 64 on the __GFX9__ device pass and 32 on RDNA (_deps/raft/install/include/raft/util/cuda_dev_essentials.cuh:87-123). The installed raft is the keystone multi-arch build (host_warp_size() runtime hipDeviceGetAttribute query present, cuda_dev_essentials.cuh:98), not a stale pre-ce0fa68c install. cuVS has no warp-size constant of its own in the compiled slice. The m=513 wave64 case (param /6) passes -- empirical wave64 confirmation.

CUDA-path inertness confirmed: `|| defined(USE_HIP)` adds nothing on CUDA (USE_HIP undefined there), so sm80+ codegen is unchanged; all other edits remain `#if defined(USE_HIP)` / `#if !defined(USE_HIP)` guarded (the previously-verified l_inf static_cast, dispatch_sm80 stub, dispatch-inl cutlass_op_unavailable, kernel_sm60 early-exit skip are byte-unchanged in this delta).

Notes accuracy: the prior false claims are corrected. notes.md:90-103 now describes the guard fix honestly ("Keep the body live on HIP", "Validated on gfx90a by the new FUSED_NN_TEST"); notes.md:228-229 now says the fused-NN path is now-live and exercised. Both match what I observed on hardware.

Commit hygiene: title "[ROCm] Port cuVS pairwise-distance subsystem to HIP (Strategy A)" (64 chars); body names Claude (Anthropic), has a Test Plan with literal commands, no Co-Authored-By noreply trailer, no ghstack, no em-dash, ASCII clean. No AMD-internal account references in the diff. Branch moat-port; fork main ee7fada6 is the clean upstream mirror (unchanged); Actions disabled on the fork (enabled:false).

No problems found. Proceeds to validation.
