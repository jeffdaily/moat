# plumed2 notes

## Build instructions

### Prerequisites

Build and install PLUMED first from the main source tree:

```bash
cd projects/plumed2/src
./configure --prefix=/path/to/plumed/install --enable-modules=all
make -j$(nproc)
make install
```

### Building the cudaCoord plugin for HIP

```bash
export PATH="/path/to/plumed/install/bin:$PATH"
export PLUMED_KERNEL="/path/to/plumed/install/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="/path/to/plumed/install/lib:$LD_LIBRARY_PATH"
export USE_HIP=1

cd plugins/cudaCoord
./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx90a
```

For other architectures, change `HIP_ARCHITECTURES` (e.g., `gfx1100`).

### Running tests

```bash
cd regtest
ln -s ../../../regtest/scripts .
cd ..
make USE_HIP=1 check
```

## Port notes

### hipFuncGetAttributes workaround

HIP's `hipFuncGetAttributes` doesn't support templated kernel function pointers
the same way CUDA does. The code uses `cudaFuncGetAttributes` to query
`maxDynamicSharedSizeBytes` and `maxThreadsPerBlock` for optimal kernel launch
parameters.

Workaround: Use `hipDeviceGetAttribute` to query the max shared memory per block
from the device directly, rather than per-kernel attributes. This provides
reasonable defaults that work correctly.

### Tested on linux-gfx90a

- All 32 regression tests pass
- Double and single precision modes work
- Both ortho PBC and no-PBC configurations work

## Validation 2026-06-05

### Platform: linux-gfx90a

GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0

Build command:
```bash
export PATH="/var/lib/jenkins/moat/_deps/plumed2/install/bin:$PATH"
export PLUMED_KERNEL="/var/lib/jenkins/moat/_deps/plumed2/install/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="/var/lib/jenkins/moat/_deps/plumed2/install/lib:$LD_LIBRARY_PATH"
export USE_HIP=1
export HIP_VISIBLE_DEVICES=0

cd /var/lib/jenkins/moat/projects/plumed2/src/plugins/cudaCoord
./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx90a
```

Test command:
```bash
make USE_HIP=1 check
```

Results:
- 32/32 regression tests PASS (0 errors)
- Test suites: cudatest (double/float), cudatestPair, cudatestWB
- Configurations tested: ortho PBC, no PBC, MPI variants
- Plugin verified linked against libamdhip64.so.7
- Code objects verified compiled for gfx90a architecture

## Review 2026-06-05

### Problems found

1. **Commit message contains MOAT jargon** (`plugins/cudaCoord/src/cuda_to_hip.h` commit 9b67b5f9d)
   - Line: "The port follows the Strategy A pattern: a single cuda_to_hip.h compatibility"
   - Per CLAUDE.md: "Upstream-visible text (commit messages, code comments, PR bodies) carries no MOAT-internal vocabulary -- no 'Strategy A/B'..."
   - Fix: Amend the commit message to remove "Strategy A pattern" and describe it in neutral terms, e.g. "The port uses a compatibility header approach: a single cuda_to_hip.h..."

### Verified fault classes (no issues found)

- **Warp size**: No hardcoded 32, no `__shfl*`, `__ballot`, `__syncwarp`, `__activemask`, or warpSize-dependent code. The plan correctly noted this is LOW RISK.
- **Rule-of-five on resource handles**: The stream handles are created/destroyed in constructor/destructor with `mpiActive` guards. The class is derived from Colvar (PLUMED base) and is not copied by the framework. This is an existing upstream pattern.
- **OOB neighbor reads**: Not applicable; the code uses tile-based iteration with explicit bounds checking (`activeThread`, `activeCandidates`).
- **Texture pitch alignment (256B)**: Not applicable; no texture objects used.
- **BlockReduce TempStorage reuse**: Each kernel uses TempStorage once, no reuse races.
- **Library swaps**: CUB -> hipCUB correctly mapped via `#define cub hipcub`.

### Build system review

- **HIP_ARCHITECTURES**: Uses `?=` to default to gfx90a only when unset, allowing follower platforms to override without source changes. Correct per PORTING_GUIDE.
- **Makefile conditionals**: Clean separation of CUDA vs HIP paths via `ifdef USE_HIP`.
- **configure.sh**: Properly handles flag differences (removes `-Xcompiler` prefixes for hipcc/clang).

### hipFuncGetAttributes workaround

The workaround queries device attributes (`hipDeviceAttributeMaxSharedMemoryPerBlock`) instead of per-kernel attributes. This is acceptable because the code uses the value only for dynamic shared memory sizing defaults, not for precise per-kernel tuning. The workaround correctly documents the limitation.

### Commit hygiene

- Title: `[ROCm]` prefix, under 72 chars. OK.
- Body: Discloses Claude assistance. OK.
- No `Co-Authored-By: noreply` trailer. OK.
- **PROBLEM**: Contains "Strategy A" MOAT jargon (see above).

### Validation status

Porter reports 32/32 regression tests pass on gfx90a. This is the correct validation path (real GPU run with the project's test suite). Reviewer does not block for missing GPU run; the validator stage provides that.

## Review fix 2026-06-05

Amended commit message to remove MOAT jargon "Strategy A pattern" -- now reads "uses a compatibility header approach". Pushed as 2ba581c3b.

## Review 2026-06-05 (re-review after jargon fix)

Re-reviewed commit 2ba581c3b after porter amended the commit message.

### Jargon fix verified

The previous review requested removal of "Strategy A pattern" MOAT jargon from the commit message. The porter amended it to "uses a compatibility header approach". This is now correct -- no MOAT-internal vocabulary remains.

### All ROCm fault classes verified (no issues)

- Warp size: No hardcoded warpSize, no warp-level intrinsics
- Rule-of-five: Stream handles managed in ctor/dtor, class not copied
- OOB neighbor reads: Tile-based iteration with explicit bounds
- 256B texture pitch: Not applicable, no textures
- BlockReduce TempStorage: Single use per kernel, no reuse races
- Library swaps: CUB -> hipCUB via `#define cub hipcub`
- hipFuncGetAttributes workaround: Queries device attributes, appropriate fallback

### Commit hygiene

- Title: `[ROCm]` prefix, 37 chars. OK.
- Body: Discloses Claude assistance. OK.
- No Co-Authored-By noreply trailer. OK.
- No MOAT jargon. OK.

### Result

Approved for validation. State: review-passed.
