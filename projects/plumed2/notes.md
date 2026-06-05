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
