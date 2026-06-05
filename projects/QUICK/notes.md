# QUICK notes

## Port Summary

QUICK is a quantum chemistry package with existing authoritative HIP support from the original developers (Merz/Goetz labs at UCSD/MSU, published in J. Chem. Inf. Model. 2023). The HIP code was disabled due to ROCm 5.4.3-6.2.0 compiler bugs; ROCm 6.2.1+ fixes those bugs.

The port was a validate-and-improve effort, not from-scratch:
1. Removed the configure script HIP exit block
2. Fixed hipcc path detection for ROCm 7.x ($ROCM_PATH/bin vs $ROCM_PATH/hip/bin)
3. Added missing C++ standard library includes for HIP compilation

## Build Instructions

```bash
export ROCM_PATH=/opt/rocm
mkdir build && cd build
cmake .. -DHIP=ON -DCOMPILER=GNU -DQUICK_USER_ARCH=gfx90a -DCMAKE_BUILD_TYPE=Release
make -j16
make install DESTDIR=$PWD/../install
```

## Test Instructions

```bash
export QUICK_HOME=/path/to/install/usr/local
export QUICK_BASIS=$QUICK_HOME/basis
export LD_LIBRARY_PATH=$QUICK_HOME/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
cd $QUICK_HOME
HIP_VISIBLE_DEVICES=2 ./runtest --hip --ene
```

## Validation Results (gfx90a)

- ene_acetone_rhf_321g: PASSED (TOTAL ENERGY = -190.882196695 vs ref -190.882196697, 2 microhartree agreement)
- ene_psb5_rhf_631g: PASSED (from runtest)

Note: PSB5 with 631gss basis takes >30 minutes on gfx90a, which is expected for this computationally intensive calculation.

## Known Issues

- Issue #433 reports 3x performance regression vs AmberTools23 QUICK on some systems; root cause investigation pending
