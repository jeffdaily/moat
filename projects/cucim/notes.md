# cucim notes

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated RAPIDS).

## Fork / branch
- Fork: https://github.com/jeffdaily/cucim (jeffdaily user account; `--org jeffdaily` is rejected, jeffdaily is a user not an org -- use `gh repo fork rapidsai/cucim --clone=false`).
- Actions disabled on the fork (`gh api -X PUT repos/jeffdaily/cucim/actions/permissions -F enabled=false`).
- Port branch: `moat-port` off origin/main @ abd0ff0; fork default branch stays a clean upstream mirror.

## Environment (gfx90a host, ROCm 7.2.1)
- 4x MI250X GCDs (gfx90a, wave64). GPUs 0 and 1 free (~11 MB used); 2 and 3 in use. Use GPU 0 via HIP_VISIBLE_DEVICES=0.
- conda env `py_3.12` (/opt/conda/envs/py_3.12), python 3.12.13. 128 cores; cap builds -j16.

## Phase 1 setup -- the load-bearing item: ROCm CuPy (RESOLVED)
- A ROCm CuPy is ALREADY installed: `cupy-rocm-7-0` 14.1.0 (NOT a CUDA wheel as the plan feared). Its `runtime.is_hip` is True once it imports.
- The actual blocker was a NumPy ABI mismatch, not a CUDA-vs-ROCm wheel: `cupy-rocm-7-0` 14.1.0 requires `numpy>=2.0,<2.6` but the env shipped numpy 1.26.4 -> `ImportError: numpy.core.multiarray failed to import` (the classic numpy-2-built-extension-vs-numpy-1 break). Fix is a numpy>=2.0 upgrade (done in a dedicated venv to avoid mutating the shared conda env; see below).
- scikit-image in env is 0.22.0; cuCIM run-deps want >=0.23.2,<0.27.0 (numeric reference parity). Aligned in the venv.
