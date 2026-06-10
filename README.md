# MOAT

MOAT (Moat Obliteration via Automated Translation) ports popular CUDA GitHub projects to ROCm/HIP, one repo at a time, across AMD targets: Linux gfx90a, Linux gfx1100, and Windows on gfx1101 and gfx1201. It is driven by Claude: a planner analyzes each project, a porter applies the change on a fork in the jeffdaily org, a reviewer checks it, and a validator builds and runs the real tests on AMD hardware. This repo is the control plane; it tracks progress and accumulates porting best practices in PORTING_GUIDE.md.

## How it works

Each project gets a folder under `projects/` holding its plan, notes, and a per-platform status file. A fresh Claude CLI run in this repo detects its AMD architecture, finds the next actionable project, and continues the pipeline. Linux gfx90a is the lead platform; the follower archs (Linux gfx1100, Windows gfx1101 and gfx1201) reuse the resulting fork branch and re-validate, since the AMD targets share one unified ROCm port. gfx1151 is a retired Windows host: its earlier validations are kept as records, and it is scheduled no new work.

## Scope and honesty

The project list below is a best-effort ranked union of targeted GitHub searches, not a census of every CUDA repo (GitHub search caps results per query and misses repos whose dominant language is not Cuda). Ports aim to be minimally invasive: for pure CMake projects we prefer `enable_language(HIP)` plus a single cuda-to-hip compat header (the colmap model); for pytorch extensions we rely on torch's build-time hipify. A CPU-only build smoketest proves compilation only; correctness is gated on real-GPU test runs. See PORTING_GUIDE.md.

Projects we will not port (already ported, already supported, can't be ported, or not a real target) are recorded with reasons in `data/dispositions.json` and kept out of the actionable list; `utils/triage.py` manages those decisions.

## Projects

<!-- MOAT:TABLE:START -->
Status: ✅ done · 🔧 in progress · 🟡 queued (gfx1201 follower; lead done) · 🔄 re-check (HEAD moved) · ⬜ todo/gated · 🚫 blocked/failed · — n/a or pending. Outcome: 🟣 PR merged · 🟢 PR open · 🔴 PR closed · 🔵 validated (existing ROCm confirmed on N archs) · 🍴 fork-only · ⚪ superseded · — pending. † The Windows archs (gfx1101 / gfx1201 / gfx1151) are a redundant tier -- any ONE completed (✅) satisfies the Windows requirement for PR-readiness; the two Linux archs (gfx90a then gfx1100) are each required. The project name links to upstream, (fork) to our `moat-port` branch.

| Project | gfx90a<br>Linux | gfx1100<br>Linux | gfx1101 †<br>Windows | gfx1201 †<br>Windows | gfx1151 †<br>Windows | Outcome |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| [3](https://github.com/mumax/3) ([fork](https://github.com/jeffdaily/3/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [3dgrut](https://github.com/nv-tlabs/3dgrut) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [3DGS-LM](https://github.com/lukasHoel/3DGS-LM) ([fork](https://github.com/jeffdaily/3DGS-LM/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#15](https://github.com/lukasHoel/3DGS-LM/pull/15) |
| [3DUNDERWORLD-SLS-GPU_CPU](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU) ([fork](https://github.com/jeffdaily/3DUNDERWORLD-SLS-GPU_CPU/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | 🟢 [#33](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU/pull/33) |
| [3P-ADMM-PC2](https://github.com/Samarvivian/3P-ADMM-PC2) ([fork](https://github.com/jeffdaily/3P-ADMM-PC2/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 [#10](https://github.com/Samarvivian/3P-ADMM-PC2/pull/10) |
| [accelerated-scan](https://github.com/proger/accelerated-scan) ([fork](https://github.com/jeffdaily/accelerated-scan/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#17](https://github.com/proger/accelerated-scan/pull/17) |
| [aihwkit](https://github.com/IBM/aihwkit) ([fork](https://github.com/jeffdaily/aihwkit/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#770](https://github.com/IBM/aihwkit/pull/770) |
| [alien](https://github.com/chrxh/alien) ([fork](https://github.com/jeffdaily/alien/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | 🚫 | 🟢 [#710](https://github.com/chrxh/alien/pull/710) |
| [amgcl](https://github.com/ddemidov/amgcl) ([fork](https://github.com/jeffdaily/amgcl/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | 🟢 [#315](https://github.com/ddemidov/amgcl/pull/315) |
| [arbor](https://github.com/arbor-sim/arbor) ([fork](https://github.com/jeffdaily/arbor/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#2512](https://github.com/arbor-sim/arbor/pull/2512) |
| [arrayfire](https://github.com/arrayfire/arrayfire) ([fork](https://github.com/jeffdaily/arrayfire/tree/moat-port)) | ✅ | ✅ | 🔧 | ✅ | — | 🟢 [#3708](https://github.com/arrayfire/arrayfire/pull/3708) |
| [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) ([fork](https://github.com/jeffdaily/AutoDock-GPU/tree/moat-port)) | ✅ | ✅ | 🔄 | ✅ | — | 🟢 [#320](https://github.com/ccsb-scripps/AutoDock-GPU/pull/320) |
| [bam](https://github.com/ZaidQureshi/bam) ([fork](https://github.com/jeffdaily/bam/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [baspacho](https://github.com/facebookresearch/baspacho) ([fork](https://github.com/jeffdaily/baspacho/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#10](https://github.com/facebookresearch/baspacho/pull/10) |
| [bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda) ([fork](https://github.com/jeffdaily/bellhopcuda/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#65](https://github.com/A-New-BellHope/bellhopcuda/pull/65) |
| [brian2cuda](https://github.com/brian-team/brian2cuda) ([fork](https://github.com/jeffdaily/brian2cuda/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#327](https://github.com/brian-team/brian2cuda/pull/327) |
| [catboost](https://github.com/catboost/catboost) ([fork](https://github.com/jeffdaily/catboost/tree/moat-port)) | ✅ | ✅ | 🔄 | ✅ | 🔄 | 🟢 [#3111](https://github.com/catboost/catboost/pull/3111) |
| [colmap](https://github.com/colmap/colmap) ([fork](https://github.com/jeffdaily/colmap/tree/moat-port)) | ✅ | 🚫 | 🚫 | 🚫 | — | 🟢 [#4420](https://github.com/colmap/colmap/pull/4420) |
| [colvars](https://github.com/Colvars/colvars) | 🚫 | 🚫 | 🚫 | 🚫 | — | — |
| [CPM.cu](https://github.com/OpenBMB/CPM.cu) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) ([fork](https://github.com/jeffdaily/CTranslate2/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | 🔵 validated (5 arch) |
| [CubbyFlow](https://github.com/utilForever/CubbyFlow) ([fork](https://github.com/jeffdaily/CubbyFlow/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [cucim](https://github.com/rapidsai/cucim) ([fork](https://github.com/jeffdaily/cucim/tree/moat-port)) | ✅ | ✅ | — | 🟡 | — | — |
| [cuCollections](https://github.com/NVIDIA/cuCollections) ([fork](https://github.com/jeffdaily/cuCollections/tree/moat-port)) | ✅ | ✅ | — | 🟡 | — | — |
| [cuda-efficient-features](https://github.com/fixstars/cuda-efficient-features) ([fork](https://github.com/jeffdaily/cuda-efficient-features/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#3](https://github.com/fixstars/cuda-efficient-features/pull/3) |
| [CUDA-L2](https://github.com/deepreinforce-ai/CUDA-L2) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [CUDA-ScanMatcher-ICP](https://github.com/botforge/CUDA-ScanMatcher-ICP) ([fork](https://github.com/jeffdaily/CUDA-ScanMatcher-ICP/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [cudaKDTree](https://github.com/ingowald/cudaKDTree) ([fork](https://github.com/jeffdaily/cudaKDTree/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| [CudaSift](https://github.com/Celebrandil/CudaSift) ([fork](https://github.com/jeffdaily/CudaSift/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| [cudf](https://github.com/rapidsai/cudf) ([fork](https://github.com/jeffdaily/cudf/tree/moat-port)) | ✅ | ✅ | — | 🟡 | — | — |
| [cuEquivariance](https://github.com/NVIDIA/cuEquivariance) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [cugraph](https://github.com/rapidsai/cugraph) ([fork](https://github.com/jeffdaily/cugraph/tree/moat-port)) | ✅ | ✅ | — | 🟡 | — | — |
| [CuMesh](https://github.com/JeffreyXiang/CuMesh) ([fork](https://github.com/jeffdaily/CuMesh/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [cuml](https://github.com/rapidsai/cuml) ([fork](https://github.com/jeffdaily/cuml/tree/moat-port)) | ✅ | ✅ | — | 🟡 | — | — |
| [cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C) ([fork](https://github.com/jeffdaily/cuPDLP-C/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx) ([fork](https://github.com/jeffdaily/cuPDLPx/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [cupoch](https://github.com/neka-nat/cupoch) ([fork](https://github.com/jeffdaily/cupoch/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | — |
| [CuRast](https://github.com/m-schuetz/CuRast) ([fork](https://github.com/jeffdaily/CuRast/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [cuSZ](https://github.com/szcompressor/cuSZ) ([fork](https://github.com/jeffdaily/cuSZ/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [cuvs](https://github.com/rapidsai/cuvs) ([fork](https://github.com/jeffdaily/cuvs/tree/moat-port)) | ✅ | ✅ | — | 🟡 | — | — |
| [CV-CUDA](https://github.com/CVCUDA/CV-CUDA) ([fork](https://github.com/jeffdaily/CV-CUDA/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [DDN-SLAM](https://github.com/DrLi-Ming/DDN-SLAM) ([fork](https://github.com/jeffdaily/DDN-SLAM/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [DEM-Engine](https://github.com/projectchrono/DEM-Engine) ([fork](https://github.com/jeffdaily/DEM-Engine/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [dgSPARSE-Lib](https://github.com/dgSPARSE/dgSPARSE-Lib) ([fork](https://github.com/jeffdaily/dgSPARSE-Lib/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [dietgpu](https://github.com/facebookresearch/dietgpu) ([fork](https://github.com/jeffdaily/dietgpu/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| [DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone) ([fork](https://github.com/jeffdaily/DiffPhysDrone/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [DynOSAM](https://github.com/ACFR-RPG/DynOSAM) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [egg.c](https://github.com/d0rc/egg.c) ([fork](https://github.com/jeffdaily/egg.c/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [ElasticFusion](https://github.com/mp3guy/ElasticFusion) ([fork](https://github.com/jeffdaily/ElasticFusion/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | — |
| [EnvGS](https://github.com/zju3dv/EnvGS) ([fork](https://github.com/jeffdaily/EnvGS/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [espresso](https://github.com/espressomd/espresso) ([fork](https://github.com/jeffdaily/espresso/tree/moat-port)) | ✅ | ✅ | — | 🚫 | — | — |
| [evogp](https://github.com/EMI-Group/evogp) ([fork](https://github.com/jeffdaily/evogp/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | — |
| [faiss](https://github.com/facebookresearch/faiss) ([fork](https://github.com/jeffdaily/faiss/tree/moat-port)) | ✅ | ✅ | — | ✅ | ✅ | — |
| [FaithC](https://github.com/Luo-Yihao/FaithC) ([fork](https://github.com/jeffdaily/FaithC/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [Fast-Poisson-Image-Editing](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing) ([fork](https://github.com/jeffdaily/Fast-Poisson-Image-Editing/tree/moat-port)) | ❓ | ✅ | ✅ | ✅ | ✅ | 🟢 [#25](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/pull/25) |
| [faster-gaussian-splatting](https://github.com/nerficg-project/faster-gaussian-splatting) ([fork](https://github.com/jeffdaily/faster-gaussian-splatting/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [FastGeodis](https://github.com/masadcv/FastGeodis) ([fork](https://github.com/jeffdaily/FastGeodis/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [fdtd3d](https://github.com/zer011b/fdtd3d) ([fork](https://github.com/jeffdaily/fdtd3d/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [ffpa-attn](https://github.com/xlite-dev/ffpa-attn) ([fork](https://github.com/jeffdaily/ffpa-attn/tree/moat-port)) | ✅ | 🚫 | ✅ | ✅ | — | — |
| [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2) ([fork](https://github.com/jeffdaily/FLAMEGPU2/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [FlashKDA](https://github.com/MoonshotAI/FlashKDA) | 🚫 | 🚫 | 🚫 | 🚫 | — | — |
| [FlashMoE](https://github.com/osayamenja/FlashMoE) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [FlashRT](https://github.com/LiangSu8899/FlashRT) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [foldmason](https://github.com/steineggerlab/foldmason) ([fork](https://github.com/jeffdaily/foldmason/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [foldseek](https://github.com/steineggerlab/foldseek) ([fork](https://github.com/jeffdaily/foldseek/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [fp6_llm](https://github.com/usyd-fsalab/fp6_llm) | 🚫 | 🚫 | 🚫 | 🚫 | — | — |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) ([fork](https://github.com/jeffdaily/fused-ssim/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🔵 validated (3 arch) |
| [futhark](https://github.com/diku-dk/futhark) | 🚫 | 🚫 | 🚫 | 🚫 | — | — |
| [gaussian_splatting](https://github.com/joeyan/gaussian_splatting) ([fork](https://github.com/jeffdaily/gaussian_splatting/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [gdtk](https://github.com/gdtk-uq/gdtk) ([fork](https://github.com/jeffdaily/gdtk/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [GOMC](https://github.com/GOMC-WSU/GOMC) ([fork](https://github.com/jeffdaily/GOMC/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [GooFit](https://github.com/GooFit/GooFit) ([fork](https://github.com/jeffdaily/GooFit/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) ([fork](https://github.com/jeffdaily/gpu4pyscf/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [GPU_IPC](https://github.com/KemengHuang/GPU_IPC) ([fork](https://github.com/jeffdaily/GPU_IPC/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [Gpufit](https://github.com/gpufit/Gpufit) ([fork](https://github.com/jeffdaily/Gpufit/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | 🚫 | — |
| [GPUMD](https://github.com/brucefan1983/GPUMD) ([fork](https://github.com/jeffdaily/GPUMD/tree/moat-port)) | ❓ | ✅ | ✅ | ✅ | ✅ | 🟢 [#1538](https://github.com/brucefan1983/GPUMD/pull/1538) |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) ([fork](https://github.com/jeffdaily/gpuRIR/tree/moat-port)) | ✅ | ✅ | — | ✅ | ✅ | — |
| [gRASPA](https://github.com/snurr-group/gRASPA) ([fork](https://github.com/jeffdaily/gRASPA/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [gsplat](https://github.com/nerfstudio-project/gsplat) ([fork](https://github.com/jeffdaily/gsplat/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 [#970](https://github.com/nerfstudio-project/gsplat/pull/970) |
| [gtsam_points](https://github.com/koide3/gtsam_points) ([fork](https://github.com/jeffdaily/gtsam_points/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | — |
| [heavydb](https://github.com/heavyai/heavydb) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU) ([fork](https://github.com/jeffdaily/HEonGPU/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [icicle](https://github.com/ingonyama-zk/icicle) ([fork](https://github.com/jeffdaily/icicle/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [k2](https://github.com/k2-fsa/k2) ([fork](https://github.com/jeffdaily/k2/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | — |
| [kaldi](https://github.com/kaldi-asr/kaldi) ([fork](https://github.com/jeffdaily/kaldi/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [lc0](https://github.com/LeelaChessZero/lc0) ([fork](https://github.com/jeffdaily/lc0/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | 🚫 | — |
| [LEAP](https://github.com/llnl/LEAP) ([fork](https://github.com/jeffdaily/LEAP/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [libCEED](https://github.com/CEED/libCEED) | 🚫 | 🚫 | 🚫 | 🚫 | — | — |
| [libhipcxx](https://github.com/ROCm/libhipcxx) ([fork](https://github.com/jeffdaily/libhipcxx/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | 🟢 [#23](https://github.com/ROCm/libhipcxx/pull/23) |
| [libSGM](https://github.com/fixstars/libSGM) ([fork](https://github.com/jeffdaily/libSGM/tree/moat-port)) | ✅ | ✅ | — | ✅ | ✅ | 🟢 [#89](https://github.com/fixstars/libSGM/pull/89) |
| [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) ([fork](https://github.com/jeffdaily/LichtFeld-Studio/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [LiteGS](https://github.com/MooreThreads/LiteGS) ([fork](https://github.com/jeffdaily/LiteGS/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [llm-awq](https://github.com/mit-han-lab/llm-awq) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [llm.c](https://github.com/karpathy/llm.c) ([fork](https://github.com/jeffdaily/llm.c/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 [#854](https://github.com/karpathy/llm.c/pull/854) |
| [llmq](https://github.com/IST-DASLab/llmq) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [LMCache](https://github.com/LMCache/LMCache) ([fork](https://github.com/jeffdaily/LMCache/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | 🔵 validated (2 arch) |
| [mahout](https://github.com/apache/mahout) ([fork](https://github.com/jeffdaily/mahout/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| [marian-dev](https://github.com/marian-nmt/marian-dev) ([fork](https://github.com/jeffdaily/marian-dev/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) ([fork](https://github.com/jeffdaily/MASt3R-SLAM/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [mcx](https://github.com/fangq/mcx) ([fork](https://github.com/jeffdaily/mcx/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [metaeuk](https://github.com/soedinglab/metaeuk) ([fork](https://github.com/jeffdaily/metaeuk/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [mHC.cu](https://github.com/AndreSlavescu/mHC.cu) ([fork](https://github.com/jeffdaily/mHC.cu/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [mirage](https://github.com/mirage-project/mirage) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [mmcv](https://github.com/open-mmlab/mmcv) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) ([fork](https://github.com/jeffdaily/MMseqs2/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) ([fork](https://github.com/jeffdaily/MPPI-Generic/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | — |
| [mwa_hyperdrive](https://github.com/MWATelescope/mwa_hyperdrive) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [NATTEN](https://github.com/SHI-Labs/NATTEN) | 🚫 | 🚫 | 🚫 | 🚫 | — | — |
| [nerfacc](https://github.com/nerfstudio-project/nerfacc) ([fork](https://github.com/jeffdaily/nerfacc/tree/moat-port)) | 🔄 | 🔄 | — | ✅ | — | — |
| [ntransformer](https://github.com/xaskasdf/ntransformer) ([fork](https://github.com/jeffdaily/ntransformer/tree/moat-port)) | ✅ | ✅ | — | ✅ | ✅ | — |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) ([fork](https://github.com/jeffdaily/nvdiffrast/tree/moat-port)) | 🚫 | ✅ | — | 🟡 | — | — |
| [OCTproZ](https://github.com/spectralcode/OCTproZ) ([fork](https://github.com/jeffdaily/OCTproZ/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [ohm](https://github.com/csiro-robotics/ohm) ([fork](https://github.com/jeffdaily/ohm/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [oneflow](https://github.com/Oneflow-Inc/oneflow) ([fork](https://github.com/jeffdaily/oneflow/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [op43dgs](https://github.com/LetianHuang/op43dgs) ([fork](https://github.com/jeffdaily/op43dgs/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [Open3D](https://github.com/isl-org/Open3D) ([fork](https://github.com/jeffdaily/Open3D/tree/moat-port)) | ✅ | ✅ | 🚫 | ✅ | — | 🟢 [#7509](https://github.com/isl-org/Open3D/pull/7509) |
| [opencv_contrib](https://github.com/opencv/opencv_contrib) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [PhoenixOS](https://github.com/SJTU-IPADS/PhoenixOS) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [plumed2](https://github.com/plumed/plumed2) ([fork](https://github.com/jeffdaily/plumed2/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [plvs](https://github.com/luigifreda/plvs) ([fork](https://github.com/jeffdaily/plvs/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [Pointcept](https://github.com/Pointcept/Pointcept) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [popsift](https://github.com/alicevision/popsift) ([fork](https://github.com/jeffdaily/popsift/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 [#186](https://github.com/alicevision/popsift/pull/186) |
| [prismatic](https://github.com/prism-em/prismatic) ([fork](https://github.com/jeffdaily/prismatic/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [pyslam](https://github.com/luigifreda/pyslam) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [pytorch3d](https://github.com/facebookresearch/pytorch3d) ([fork](https://github.com/facebookresearch/pytorch3d/tree/moat-port)) | ❓ | ✅ | ✅ | ✅ | ✅ | 🟢 [#2039](https://github.com/facebookresearch/pytorch3d/pull/2039) |
| [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [qrack](https://github.com/unitaryfoundation/qrack) ([fork](https://github.com/jeffdaily/qrack/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [Quest](https://github.com/mit-han-lab/Quest) ([fork](https://github.com/jeffdaily/Quest/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [QUICK](https://github.com/merzlab/QUICK) ([fork](https://github.com/jeffdaily/QUICK/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [raft](https://github.com/rapidsai/raft) ([fork](https://github.com/jeffdaily/raft/tree/moat-port)) | ✅ | ✅ | — | 🟡 | — | — |
| [rmcl](https://github.com/uos/rmcl) ([fork](https://github.com/jeffdaily/rmagine/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [rmm](https://github.com/rapidsai/rmm) ([fork](https://github.com/jeffdaily/rmm/tree/moat-port)) | ✅ | ✅ | — | 🟡 | ✅ | — |
| [RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA) ([fork](https://github.com/jeffdaily/RWKV-CUDA/tree/moat-port)) | ✅ | ✅ | — | ✅ | ✅ | — |
| [RXMesh](https://github.com/owensgroup/RXMesh) ([fork](https://github.com/jeffdaily/RXMesh/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | — | 🟢 [#73](https://github.com/owensgroup/RXMesh/pull/73) |
| [SCAMP](https://github.com/zpzim/SCAMP) ([fork](https://github.com/jeffdaily/SCAMP/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [SpargeAttn](https://github.com/thu-ml/SpargeAttn) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [sparser-faster-llms](https://github.com/SakanaAI/sparser-faster-llms) | 🚫 | 🚫 | 🚫 | 🚫 | — | — |
| [spconv](https://github.com/traveller59/spconv) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [splatad](https://github.com/carlinds/splatad) ([fork](https://github.com/jeffdaily/splatad/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [sppark](https://github.com/supranational/sppark) ([fork](https://github.com/jeffdaily/sppark/tree/moat-port)) | 🚫 | ⬜ | ⬜ | ⬜ | — | — |
| [stdgpu](https://github.com/stotko/stdgpu) ([fork](https://github.com/jeffdaily/stdgpu/tree/moat-port)) | ✅ | ✅ | — | ✅ | 🚫 | — |
| [STRUMPACK](https://github.com/pghysels/STRUMPACK) ([fork](https://github.com/jeffdaily/STRUMPACK/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
| [symforce](https://github.com/symforce-org/symforce) ([fork](https://github.com/jeffdaily/symforce/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [TIGRE](https://github.com/CERN/TIGRE) ([fork](https://github.com/jeffdaily/TIGRE/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [tiny-vllm](https://github.com/jmaczan/tiny-vllm) ([fork](https://github.com/jeffdaily/tiny-vllm/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [torch-linear-assignment](https://github.com/ivan-chai/torch-linear-assignment) ([fork](https://github.com/jeffdaily/torch-linear-assignment/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [TTT3R](https://github.com/Inception3D/TTT3R) ([fork](https://github.com/jeffdaily/TTT3R/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [TurboFNO](https://github.com/shixun404/TurboFNO) ([fork](https://github.com/jeffdaily/TurboFNO/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [unified-cache-management](https://github.com/ModelEngine-Group/unified-cache-management) ([fork](https://github.com/jeffdaily/unified-cache-management/tree/moat-port)) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| [Velvet](https://github.com/vitalight/Velvet) ([fork](https://github.com/jeffdaily/Velvet/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [visionaray](https://github.com/szellmann/visionaray) ([fork](https://github.com/jeffdaily/visionaray/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [yalm](https://github.com/andrewkchan/yalm) ([fork](https://github.com/jeffdaily/yalm/tree/moat-port)) | ✅ | ✅ | — | ✅ | — | — |
| [YarnBall](https://github.com/jerry060599/YarnBall) | 🚫 | ⬜ | ⬜ | ⬜ | ⬜ | — |
| [ZhiLight](https://github.com/zhihu/ZhiLight) ([fork](https://github.com/jeffdaily/ZhiLight/tree/moat-port)) | ✅ | ✅ | 🚫 | 🚫 | — | — |
<!-- MOAT:TABLE:END -->

## Layout

See `projects/README.md` for the per-project files, `PORTING_GUIDE.md` for porting strategy and fault classes, and `CLAUDE.md` for how a Claude CLI drives the pipeline.
