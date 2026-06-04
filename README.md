# MOAT

MOAT (Moat Obliteration via Automated Translation) ports popular CUDA GitHub projects to ROCm/HIP, one repo at a time, across AMD targets: Linux gfx90a, Linux gfx1100, and Windows gfx1151. It is driven by Claude: a planner analyzes each project, a porter applies the change on a fork in the jeffdaily org, a reviewer checks it, and a validator builds and runs the real tests on AMD hardware. This repo is the control plane; it tracks progress and accumulates porting best practices in PORTING_GUIDE.md.

## How it works

Each project gets a folder under `projects/` holding its plan, notes, and a per-platform status file. A fresh Claude CLI run in this repo detects its AMD architecture, finds the next actionable project, and continues the pipeline. Linux gfx90a is the lead platform; gfx1100 and gfx1151 reuse the resulting fork branch and re-validate, since the three AMD targets share one unified ROCm port.

## Scope and honesty

The project list below is a best-effort ranked union of targeted GitHub searches, not a census of every CUDA repo (GitHub search caps results per query and misses repos whose dominant language is not Cuda). Ports aim to be minimally invasive: for pure CMake projects we prefer `enable_language(HIP)` plus a single cuda-to-hip compat header (the colmap model); for pytorch extensions we rely on torch's build-time hipify. A CPU-only build smoketest proves compilation only; correctness is gated on real-GPU test runs. See PORTING_GUIDE.md.

Projects we will not port (already ported, already supported, can't be ported, or not a real target) are recorded with reasons in `data/dispositions.json` and kept out of the actionable list; `utils/triage.py` manages those decisions.

## Projects

<!-- MOAT:TABLE:START -->
Status: ✅ done · 🔧 in progress · 🟡 queued (follower; lead done) · 🔄 re-check (HEAD moved) · ⬜ todo/gated · 🚫 blocked · — n/a. Outcome: 🟣 PR merged · 🟢 PR open · 🔴 PR closed · 🔵 validated (existing ROCm confirmed on N archs) · 🍴 fork-only · ⚪ superseded · — pending. The project name links to upstream, (fork) to our `moat-port` branch.

| Project | gfx90a<br>Linux | gfx1100<br>Linux | gfx1151<br>Windows | Outcome |
| --- | :---: | :---: | :---: | --- |
| [3](https://github.com/mumax/3) ([fork](https://github.com/jeffdaily/3/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [3DGS-LM](https://github.com/lukasHoel/3DGS-LM) ([fork](https://github.com/jeffdaily/3DGS-LM/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [3DUNDERWORLD-SLS-GPU_CPU](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU) ([fork](https://github.com/jeffdaily/3DUNDERWORLD-SLS-GPU_CPU/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [3P-ADMM-PC2](https://github.com/Samarvivian/3P-ADMM-PC2) ([fork](https://github.com/jeffdaily/3P-ADMM-PC2/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [aihwkit](https://github.com/IBM/aihwkit) ([fork](https://github.com/jeffdaily/aihwkit/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [alien](https://github.com/chrxh/alien) ([fork](https://github.com/jeffdaily/alien/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [amgcl](https://github.com/ddemidov/amgcl) ([fork](https://github.com/jeffdaily/amgcl/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [arrayfire](https://github.com/arrayfire/arrayfire) ([fork](https://github.com/jeffdaily/arrayfire/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) ([fork](https://github.com/jeffdaily/AutoDock-GPU/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [bam](https://github.com/ZaidQureshi/bam) | ⬜ | ⬜ | ⬜ | — |
| [catboost](https://github.com/catboost/catboost) ([fork](https://github.com/jeffdaily/catboost/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [CPM.cu](https://github.com/OpenBMB/CPM.cu) | 🚫 | ⬜ | ⬜ | — |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) ([fork](https://github.com/jeffdaily/CTranslate2/tree/moat-port)) | ✅ | ✅ | ✅ | 🔵 validated (3 arch) |
| [cucim](https://github.com/rapidsai/cucim) ([fork](https://github.com/jeffdaily/cucim/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [cuCollections](https://github.com/NVIDIA/cuCollections) ([fork](https://github.com/jeffdaily/cuCollections/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [CUDA-L2](https://github.com/deepreinforce-ai/CUDA-L2) | 🚫 | ⬜ | ⬜ | — |
| [cudaKDTree](https://github.com/ingowald/cudaKDTree) ([fork](https://github.com/jeffdaily/cudaKDTree/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [CudaSift](https://github.com/Celebrandil/CudaSift) ([fork](https://github.com/jeffdaily/CudaSift/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [cudf](https://github.com/rapidsai/cudf) ([fork](https://github.com/jeffdaily/cudf/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [cuEquivariance](https://github.com/NVIDIA/cuEquivariance) | 🚫 | ⬜ | ⬜ | — |
| [cugraph](https://github.com/rapidsai/cugraph) ([fork](https://github.com/jeffdaily/cugraph/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [cuml](https://github.com/rapidsai/cuml) ([fork](https://github.com/jeffdaily/cuml/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [cupoch](https://github.com/neka-nat/cupoch) ([fork](https://github.com/jeffdaily/cupoch/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [cuvs](https://github.com/rapidsai/cuvs) ([fork](https://github.com/jeffdaily/cuvs/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [CV-CUDA](https://github.com/CVCUDA/CV-CUDA) ([fork](https://github.com/jeffdaily/CV-CUDA/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [DDN-SLAM](https://github.com/DrLi-Ming/DDN-SLAM) ([fork](https://github.com/jeffdaily/DDN-SLAM/tree/moat-port)) | 🚫 | ⬜ | ⬜ | — |
| [dietgpu](https://github.com/facebookresearch/dietgpu) ([fork](https://github.com/jeffdaily/dietgpu/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone) ([fork](https://github.com/jeffdaily/DiffPhysDrone/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [egg.c](https://github.com/d0rc/egg.c) ([fork](https://github.com/jeffdaily/egg.c/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [ElasticFusion](https://github.com/mp3guy/ElasticFusion) ([fork](https://github.com/jeffdaily/ElasticFusion/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [EnvGS](https://github.com/zju3dv/EnvGS) ([fork](https://github.com/jeffdaily/EnvGS/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [espresso](https://github.com/espressomd/espresso) ([fork](https://github.com/jeffdaily/espresso/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [evogp](https://github.com/EMI-Group/evogp) ([fork](https://github.com/jeffdaily/evogp/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [faiss](https://github.com/facebookresearch/faiss) ([fork](https://github.com/jeffdaily/faiss/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [FaithC](https://github.com/Luo-Yihao/FaithC) ([fork](https://github.com/jeffdaily/FaithC/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [Fast-Poisson-Image-Editing](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing) ([fork](https://github.com/jeffdaily/Fast-Poisson-Image-Editing/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#25](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/pull/25) |
| [ffpa-attn](https://github.com/xlite-dev/ffpa-attn) | ⬜ | ⬜ | ⬜ | — |
| [FlashKDA](https://github.com/MoonshotAI/FlashKDA) | 🔧 | ⬜ | ⬜ | — |
| [FlashMoE](https://github.com/osayamenja/FlashMoE) | ⬜ | ⬜ | ⬜ | — |
| [FlashRT](https://github.com/LiangSu8899/FlashRT) | ⬜ | ⬜ | ⬜ | — |
| [foldseek](https://github.com/steineggerlab/foldseek) ([fork](https://github.com/jeffdaily/foldseek/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [fp6_llm](https://github.com/usyd-fsalab/fp6_llm) | ⬜ | ⬜ | ⬜ | — |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) ([fork](https://github.com/jeffdaily/fused-ssim/tree/moat-port)) | ✅ | ✅ | 🚫 | 🔵 validated (2 arch) |
| [gaussian_splatting](https://github.com/joeyan/gaussian_splatting) ([fork](https://github.com/jeffdaily/gaussian_splatting/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) ([fork](https://github.com/jeffdaily/gpu4pyscf/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [Gpufit](https://github.com/gpufit/Gpufit) ([fork](https://github.com/jeffdaily/Gpufit/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [GPUMD](https://github.com/brucefan1983/GPUMD) ([fork](https://github.com/jeffdaily/GPUMD/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#1538](https://github.com/brucefan1983/GPUMD/pull/1538) |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) ([fork](https://github.com/jeffdaily/gpuRIR/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [gsplat](https://github.com/nerfstudio-project/gsplat) ([fork](https://github.com/jeffdaily/gsplat/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#970](https://github.com/nerfstudio-project/gsplat/pull/970) |
| [gtsam_points](https://github.com/koide3/gtsam_points) ([fork](https://github.com/jeffdaily/gtsam_points/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [heavydb](https://github.com/heavyai/heavydb) | 🚫 | ⬜ | ⬜ | — |
| [icicle](https://github.com/ingonyama-zk/icicle) ([fork](https://github.com/jeffdaily/icicle/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [k2](https://github.com/k2-fsa/k2) ([fork](https://github.com/jeffdaily/k2/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [kaldi](https://github.com/kaldi-asr/kaldi) ([fork](https://github.com/jeffdaily/kaldi/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [lc0](https://github.com/LeelaChessZero/lc0) ([fork](https://github.com/jeffdaily/lc0/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [LEAP](https://github.com/llnl/LEAP) ([fork](https://github.com/jeffdaily/LEAP/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [libSGM](https://github.com/fixstars/libSGM) ([fork](https://github.com/jeffdaily/libSGM/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#89](https://github.com/fixstars/libSGM/pull/89) |
| [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) ([fork](https://github.com/jeffdaily/LichtFeld-Studio/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [LiteGS](https://github.com/MooreThreads/LiteGS) ([fork](https://github.com/jeffdaily/LiteGS/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [llm.c](https://github.com/karpathy/llm.c) ([fork](https://github.com/jeffdaily/llm.c/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#854](https://github.com/karpathy/llm.c/pull/854) |
| [llmq](https://github.com/IST-DASLab/llmq) | 🚫 | ⬜ | ⬜ | — |
| [LMCache](https://github.com/LMCache/LMCache) ([fork](https://github.com/jeffdaily/LMCache/tree/moat-port)) | ✅ | ✅ | 🚫 | 🔵 validated (2 arch) |
| [mahout](https://github.com/apache/mahout) ([fork](https://github.com/jeffdaily/mahout/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [marian-dev](https://github.com/marian-nmt/marian-dev) ([fork](https://github.com/jeffdaily/marian-dev/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) ([fork](https://github.com/jeffdaily/MASt3R-SLAM/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [mHC.cu](https://github.com/AndreSlavescu/mHC.cu) ([fork](https://github.com/jeffdaily/mHC.cu/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [mirage](https://github.com/mirage-project/mirage) | 🚫 | ⬜ | ⬜ | — |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) ([fork](https://github.com/jeffdaily/MMseqs2/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) ([fork](https://github.com/jeffdaily/MPPI-Generic/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [NATTEN](https://github.com/SHI-Labs/NATTEN) | 🚫 | 🚫 | 🚫 | — |
| [ntransformer](https://github.com/xaskasdf/ntransformer) ([fork](https://github.com/jeffdaily/ntransformer/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [oneflow](https://github.com/Oneflow-Inc/oneflow) ([fork](https://github.com/jeffdaily/oneflow/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [op43dgs](https://github.com/LetianHuang/op43dgs) ([fork](https://github.com/jeffdaily/op43dgs/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [Open3D](https://github.com/isl-org/Open3D) ([fork](https://github.com/jeffdaily/Open3D/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [PhoenixOS](https://github.com/SJTU-IPADS/PhoenixOS) | ⬜ | ⬜ | ⬜ | — |
| [popsift](https://github.com/alicevision/popsift) ([fork](https://github.com/jeffdaily/popsift/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#186](https://github.com/alicevision/popsift/pull/186) |
| [pytorch3d](https://github.com/facebookresearch/pytorch3d) ([fork](https://github.com/facebookresearch/pytorch3d/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#2039](https://github.com/facebookresearch/pytorch3d/pull/2039) |
| [qrack](https://github.com/unitaryfoundation/qrack) ([fork](https://github.com/jeffdaily/qrack/tree/moat-port)) | 🔧 | ⬜ | ⬜ | — |
| [Quest](https://github.com/mit-han-lab/Quest) | 🔧 | ⬜ | ⬜ | — |
| [raft](https://github.com/rapidsai/raft) ([fork](https://github.com/jeffdaily/raft/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [rmcl](https://github.com/uos/rmcl) ([fork](https://github.com/jeffdaily/rmagine/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [rmm](https://github.com/rapidsai/rmm) ([fork](https://github.com/jeffdaily/rmm/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA) ([fork](https://github.com/jeffdaily/RWKV-CUDA/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [RXMesh](https://github.com/owensgroup/RXMesh) ([fork](https://github.com/jeffdaily/RXMesh/tree/moat-port)) | 🔄 | ✅ | ✅ | — |
| [SpargeAttn](https://github.com/thu-ml/SpargeAttn) | 🚫 | ⬜ | ⬜ | — |
| [sparser-faster-llms](https://github.com/SakanaAI/sparser-faster-llms) | ⬜ | ⬜ | ⬜ | — |
| [splatad](https://github.com/carlinds/splatad) ([fork](https://github.com/jeffdaily/splatad/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [sppark](https://github.com/supranational/sppark) | ⬜ | ⬜ | ⬜ | — |
| [STRUMPACK](https://github.com/pghysels/STRUMPACK) ([fork](https://github.com/jeffdaily/STRUMPACK/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [TIGRE](https://github.com/CERN/TIGRE) ([fork](https://github.com/jeffdaily/TIGRE/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [TurboFNO](https://github.com/shixun404/TurboFNO) ([fork](https://github.com/jeffdaily/TurboFNO/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [unified-cache-management](https://github.com/ModelEngine-Group/unified-cache-management) ([fork](https://github.com/jeffdaily/unified-cache-management/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [yalm](https://github.com/andrewkchan/yalm) ([fork](https://github.com/jeffdaily/yalm/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
| [ZhiLight](https://github.com/zhihu/ZhiLight) ([fork](https://github.com/jeffdaily/ZhiLight/tree/moat-port)) | ✅ | ✅ | 🟡 | — |
<!-- MOAT:TABLE:END -->

## Layout

See `projects/README.md` for the per-project files, `PORTING_GUIDE.md` for porting strategy and fault classes, and `CLAUDE.md` for how a Claude CLI drives the pipeline.
