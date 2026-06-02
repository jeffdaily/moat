# MOAT

MOAT (Moat Obliteration via Automated Translation) ports popular CUDA GitHub projects to ROCm/HIP, one repo at a time, across AMD targets: Linux gfx90a, Linux gfx1100, and Windows gfx1151. It is driven by Claude: a planner analyzes each project, a porter applies the change on a fork in the jeffdaily org, a reviewer checks it, and a validator builds and runs the real tests on AMD hardware. This repo is the control plane; it tracks progress and accumulates porting best practices in PORTING_GUIDE.md.

## How it works

Each project gets a folder under `projects/` holding its plan, notes, and a per-platform status file. A fresh Claude CLI run in this repo detects its AMD architecture, finds the next actionable project, and continues the pipeline. Linux gfx90a is the lead platform; gfx1100 and gfx1151 reuse the resulting fork branch and re-validate, since the three AMD targets share one unified ROCm port.

## Scope and honesty

The project list below is a best-effort ranked union of targeted GitHub searches, not a census of every CUDA repo (GitHub search caps results per query and misses repos whose dominant language is not Cuda). Ports aim to be minimally invasive: for pure CMake projects we prefer `enable_language(HIP)` plus a single cuda-to-hip compat header (the colmap model); for pytorch extensions we rely on torch's build-time hipify. A CPU-only build smoketest proves compilation only; correctness is gated on real-GPU test runs. See PORTING_GUIDE.md.

Projects we will not port (already ported, already supported, can't be ported, or not a real target) are recorded with reasons in `data/dispositions.json` and kept out of the actionable list; `utils/triage.py` manages those decisions.

Status legend: `todo` not started, `planning` / `porting` / `review` / `validating` in progress, `done` completed, `gated` waiting on the gfx90a port, `revalidate` the shared branch changed since this platform last passed, `blocked` needs input.

## Projects

<!-- MOAT:TABLE:START -->
| Project | Upstream | Fork | gfx90a | gfx1100 | gfx1151 | Upstream PR |
| --- | --- | --- | --- | --- | --- | --- |
| Open3D | [isl-org/Open3D](https://github.com/isl-org/Open3D) | [Open3D](https://github.com/jeffdaily/Open3D/tree/moat-port) | done | done | validating (blocked) | - |
| pytorch3d | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | [pytorch3d](https://github.com/facebookresearch/pytorch3d/tree/moat-port) | done | done | done | - |
| catboost | [catboost/catboost](https://github.com/catboost/catboost) | [catboost](https://github.com/jeffdaily/catboost/tree/moat-port) | revalidate | revalidate | validating (blocked) | - |
| cudf | [rapidsai/cudf](https://github.com/rapidsai/cudf) | [cudf](https://github.com/jeffdaily/cudf/tree/moat-port) | done | done | validating | - |
| LMCache | [LMCache/LMCache](https://github.com/LMCache/LMCache) | [LMCache](https://github.com/jeffdaily/LMCache/tree/moat-port) | done | done | validating (blocked) | - |
| gsplat | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | [gsplat](https://github.com/jeffdaily/gsplat/tree/moat-port) | done | done | done | - |
| llm.c | [karpathy/llm.c](https://github.com/karpathy/llm.c) | [llm.c](https://github.com/jeffdaily/llm.c/tree/moat-port) | done | done | done | - |
| kaldi | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi) | [kaldi](https://github.com/jeffdaily/kaldi/tree/moat-port) | done | done | validating | - |
| cuml | [rapidsai/cuml](https://github.com/rapidsai/cuml) | [cuml](https://github.com/jeffdaily/cuml/tree/moat-port) | done | validating | gated | - |
| CTranslate2 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | [CTranslate2](https://github.com/jeffdaily/CTranslate2/tree/moat-port) | done | done | validating | - |
| alien | [chrxh/alien](https://github.com/chrxh/alien) | [alien](https://github.com/jeffdaily/alien/tree/moat-port) | done | done | validating | - |
| mahout | [apache/mahout](https://github.com/apache/mahout) | [mahout](https://github.com/jeffdaily/mahout/tree/moat-port) | done | done | validating | - |
| LichtFeld-Studio | [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) | [LichtFeld-Studio](https://github.com/jeffdaily/LichtFeld-Studio/tree/moat-port) | done | done | validating | - |
| lc0 | [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0) | [lc0](https://github.com/jeffdaily/lc0/tree/moat-port) | done | done | validating | - |
| oneflow | [Oneflow-Inc/oneflow](https://github.com/Oneflow-Inc/oneflow) | [oneflow](https://github.com/jeffdaily/oneflow/tree/moat-port) | done | done | validating | - |
| arrayfire | [arrayfire/arrayfire](https://github.com/arrayfire/arrayfire) | [arrayfire](https://github.com/jeffdaily/arrayfire/tree/moat-port) | done | done | validating | - |
| CV-CUDA | [CVCUDA/CV-CUDA](https://github.com/CVCUDA/CV-CUDA) | [CV-CUDA](https://github.com/jeffdaily/CV-CUDA/tree/moat-port) | done | done | validating | - |
| cugraph | [rapidsai/cugraph](https://github.com/rapidsai/cugraph) | [cugraph](https://github.com/jeffdaily/cugraph/tree/moat-port) | planned (blocked) | gated | gated | - |
| mirage | [mirage-project/mirage](https://github.com/mirage-project/mirage) | - | todo (blocked) | gated | gated | - |
| k2 | [k2-fsa/k2](https://github.com/k2-fsa/k2) | [k2](https://github.com/jeffdaily/k2/tree/moat-port) | done | done | validating | - |
| raft | [rapidsai/raft](https://github.com/rapidsai/raft) | [raft](https://github.com/jeffdaily/raft/tree/moat-port) | revalidate | review | validating | - |
| heavydb | [heavyai/heavydb](https://github.com/heavyai/heavydb) | - | todo (blocked) | gated | gated | - |
| rmm | [rapidsai/rmm](https://github.com/rapidsai/rmm) | [rmm](https://github.com/jeffdaily/rmm/tree/moat-port) | done | done | done | - |
| cuvs | [rapidsai/cuvs](https://github.com/rapidsai/cuvs) | [cuvs](https://github.com/jeffdaily/cuvs/tree/moat-port) | done | validating | validating | - |
| GPUMD | [brucefan1983/GPUMD](https://github.com/brucefan1983/GPUMD) | [GPUMD](https://github.com/jeffdaily/GPUMD/tree/moat-port) | done | done | done | - |
| cuCollections | [NVIDIA/cuCollections](https://github.com/NVIDIA/cuCollections) | [cuCollections](https://github.com/jeffdaily/cuCollections/tree/moat-port) | done | done | validating | - |
| amgcl | [ddemidov/amgcl](https://github.com/ddemidov/amgcl) | [amgcl](https://github.com/jeffdaily/amgcl/tree/moat-port) | done | done | validating (blocked) | - |
| CudaSift | [Celebrandil/CudaSift](https://github.com/Celebrandil/CudaSift) | [CudaSift](https://github.com/jeffdaily/CudaSift/tree/moat-port) | done | done | done | - |
| cupoch | [neka-nat/cupoch](https://github.com/neka-nat/cupoch) | [cupoch](https://github.com/jeffdaily/cupoch/tree/moat-port) | done | done | validating (blocked) | - |
| popsift | [alicevision/popsift](https://github.com/alicevision/popsift) | [popsift](https://github.com/jeffdaily/popsift/tree/moat-port) | done | done | done | - |
| NATTEN | [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN) | - | planned | gated | gated | - |
| ElasticFusion | [mp3guy/ElasticFusion](https://github.com/mp3guy/ElasticFusion) | [ElasticFusion](https://github.com/jeffdaily/ElasticFusion/tree/moat-port) | done | done | validating | - |
| cucim | [rapidsai/cucim](https://github.com/rapidsai/cucim) | [cucim](https://github.com/jeffdaily/cucim/tree/moat-port) | done | done | validating | - |
| ZhiLight | [zhihu/ZhiLight](https://github.com/zhihu/ZhiLight) | [ZhiLight](https://github.com/jeffdaily/ZhiLight/tree/moat-port) | done | done | validating | - |
| SpargeAttn | [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn) | - | todo (blocked) | gated | gated | - |
| 3DUNDERWORLD-SLS-GPU_CPU | [theICTlab/3DUNDERWORLD-SLS-GPU_CPU](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU) | [3DUNDERWORLD-SLS-GPU_CPU](https://github.com/jeffdaily/3DUNDERWORLD-SLS-GPU_CPU/tree/moat-port) | done | done | validating | - |
| STRUMPACK | [pghysels/STRUMPACK](https://github.com/pghysels/STRUMPACK) | [STRUMPACK](https://github.com/jeffdaily/STRUMPACK/tree/moat-port) | done | done | validating | - |
| AutoDock-GPU | [ccsb-scripps/AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) | [AutoDock-GPU](https://github.com/jeffdaily/AutoDock-GPU/tree/moat-port) | done | done | validating | - |
| FlashKDA | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) | - | planned | gated | gated | - |
| 3P-ADMM-PC2 | [Samarvivian/3P-ADMM-PC2](https://github.com/Samarvivian/3P-ADMM-PC2) | [3P-ADMM-PC2](https://github.com/jeffdaily/3P-ADMM-PC2/tree/moat-port) | done | done | validating | - |
| unified-cache-management | [ModelEngine-Group/unified-cache-management](https://github.com/ModelEngine-Group/unified-cache-management) | - | todo | gated | gated | - |
| gtsam_points | [koide3/gtsam_points](https://github.com/koide3/gtsam_points) | [gtsam_points](https://github.com/jeffdaily/gtsam_points/tree/moat-port) | done | done | validating | - |
| gpu4pyscf | [pyscf/gpu4pyscf](https://github.com/pyscf/gpu4pyscf) | [gpu4pyscf](https://github.com/jeffdaily/gpu4pyscf/tree/moat-port) | done | done | validating | - |
| RXMesh | [owensgroup/RXMesh](https://github.com/owensgroup/RXMesh) | [RXMesh](https://github.com/jeffdaily/RXMesh/tree/moat-port) | done | done | validating | - |
| MPPI-Generic | [ACDSLab/MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) | [MPPI-Generic](https://github.com/jeffdaily/MPPI-Generic/tree/moat-port) | done | done | validating | - |
| sppark | [supranational/sppark](https://github.com/supranational/sppark) | - | todo | gated | gated | - |
| Gpufit | [gpufit/Gpufit](https://github.com/gpufit/Gpufit) | [Gpufit](https://github.com/jeffdaily/Gpufit/tree/moat-port) | done | done | validating | - |
| FlashRT | [LiangSu8899/FlashRT](https://github.com/LiangSu8899/FlashRT) | - | todo | gated | gated | - |
| evogp | [EMI-Group/evogp](https://github.com/EMI-Group/evogp) | [evogp](https://github.com/jeffdaily/evogp/tree/moat-port) | done | done | validating | - |
| ffpa-attn | [xlite-dev/ffpa-attn](https://github.com/xlite-dev/ffpa-attn) | - | todo | gated | gated | - |
| FlashMoE | [osayamenja/FlashMoE](https://github.com/osayamenja/FlashMoE) | - | todo | gated | gated | - |
| LiteGS | [MooreThreads/LiteGS](https://github.com/MooreThreads/LiteGS) | [LiteGS](https://github.com/jeffdaily/LiteGS/tree/moat-port) | done | done | validating | - |
| CUDA-L2 | [deepreinforce-ai/CUDA-L2](https://github.com/deepreinforce-ai/CUDA-L2) | - | todo (blocked) | gated | gated | - |
| Fast-Poisson-Image-Editing | [Trinkle23897/Fast-Poisson-Image-Editing](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing) | [Fast-Poisson-Image-Editing](https://github.com/jeffdaily/Fast-Poisson-Image-Editing/tree/moat-port) | done | done | revalidate | [#25](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/pull/25) |
| icicle | [ingonyama-zk/icicle](https://github.com/ingonyama-zk/icicle) | - | revalidate | validating | validating | - |
| dietgpu | [facebookresearch/dietgpu](https://github.com/facebookresearch/dietgpu) | [dietgpu](https://github.com/jeffdaily/dietgpu/tree/moat-port) | done | done | validating | - |
| bam | [ZaidQureshi/bam](https://github.com/ZaidQureshi/bam) | - | todo | gated | gated | - |
| sparser-faster-llms | [SakanaAI/sparser-faster-llms](https://github.com/SakanaAI/sparser-faster-llms) | - | todo | gated | gated | - |
| llmq | [IST-DASLab/llmq](https://github.com/IST-DASLab/llmq) | - | planned (blocked) | gated | gated | - |
| LEAP | [llnl/LEAP](https://github.com/llnl/LEAP) | [LEAP](https://github.com/jeffdaily/LEAP/tree/moat-port) | done | done | validating | - |
| ntransformer | [xaskasdf/ntransformer](https://github.com/xaskasdf/ntransformer) | [ntransformer](https://github.com/jeffdaily/ntransformer/tree/moat-port) | done | done | validating | - |
| FaithC | [Luo-Yihao/FaithC](https://github.com/Luo-Yihao/FaithC) | [FaithC](https://github.com/jeffdaily/FaithC/tree/moat-port) | done | done | validating | - |
| rmcl | [uos/rmcl](https://github.com/uos/rmcl) | [rmcl](https://github.com/jeffdaily/rmagine/tree/moat-port) | done | done | validating | - |
| fused-ssim | [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) | [fused-ssim](https://github.com/jeffdaily/fused-ssim/tree/moat-port) | done | done | validating (blocked) | - |
| gpuRIR | [DavidDiazGuerra/gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) | [gpuRIR](https://github.com/jeffdaily/gpuRIR/tree/moat-port) | done | done | validating | - |
| yalm | [andrewkchan/yalm](https://github.com/andrewkchan/yalm) | [yalm](https://github.com/jeffdaily/yalm/tree/moat-port) | done | done | validating | - |
| mHC.cu | [AndreSlavescu/mHC.cu](https://github.com/AndreSlavescu/mHC.cu) | [mHC.cu](https://github.com/jeffdaily/mHC.cu/tree/moat-port) | done | done | validating | - |
| EnvGS | [zju3dv/EnvGS](https://github.com/zju3dv/EnvGS) | [EnvGS](https://github.com/jeffdaily/EnvGS/tree/moat-port) | done | done | validating | - |
| egg.c | [d0rc/egg.c](https://github.com/d0rc/egg.c) | [egg.c](https://github.com/jeffdaily/egg.c/tree/moat-port) | done | done | validating | - |
| DiffPhysDrone | [HenryHuYu/DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone) | [DiffPhysDrone](https://github.com/jeffdaily/DiffPhysDrone/tree/moat-port) | done | done | validating | - |
| libSGM | [fixstars/libSGM](https://github.com/fixstars/libSGM) | [libSGM](https://github.com/jeffdaily/libSGM/tree/moat-port) | done | done | validating | - |
| CPM.cu | [OpenBMB/CPM.cu](https://github.com/OpenBMB/CPM.cu) | - | planned (blocked) | gated | gated | - |
| marian-dev | [marian-nmt/marian-dev](https://github.com/marian-nmt/marian-dev) | [marian-dev](https://github.com/jeffdaily/marian-dev/tree/moat-port) | done | done | validating | - |
| RWKV-CUDA | [BlinkDL/RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA) | [RWKV-CUDA](https://github.com/jeffdaily/RWKV-CUDA/tree/moat-port) | done | done | validating | - |
| splatad | [carlinds/splatad](https://github.com/carlinds/splatad) | [splatad](https://github.com/jeffdaily/splatad/tree/moat-port) | done | done | validating | - |
| cudaKDTree | [ingowald/cudaKDTree](https://github.com/ingowald/cudaKDTree) | [cudaKDTree](https://github.com/jeffdaily/cudaKDTree/tree/moat-port) | done | done | done | - |
| Quest | [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest) | - | planned | gated | gated | - |
| PhoenixOS | [SJTU-IPADS/PhoenixOS](https://github.com/SJTU-IPADS/PhoenixOS) | - | todo | gated | gated | - |
| 3DGS-LM | [lukasHoel/3DGS-LM](https://github.com/lukasHoel/3DGS-LM) | [3DGS-LM](https://github.com/jeffdaily/3DGS-LM/tree/moat-port) | done | done | validating | - |
| fp6_llm | [usyd-fsalab/fp6_llm](https://github.com/usyd-fsalab/fp6_llm) | - | todo | gated | gated | - |
| gaussian_splatting | [joeyan/gaussian_splatting](https://github.com/joeyan/gaussian_splatting) | [gaussian_splatting](https://github.com/jeffdaily/gaussian_splatting/tree/moat-port) | done | done | validating | - |
| DDN-SLAM | [DrLi-Ming/DDN-SLAM](https://github.com/DrLi-Ming/DDN-SLAM) | [DDN-SLAM](https://github.com/jeffdaily/DDN-SLAM/tree/moat-port) | porting (blocked) | gated | gated | - |
| op43dgs | [LetianHuang/op43dgs](https://github.com/LetianHuang/op43dgs) | [op43dgs](https://github.com/jeffdaily/op43dgs/tree/moat-port) | done | done | validating | - |
| faiss | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | [faiss](https://github.com/jeffdaily/faiss/tree/moat-port) | done | done | validating | - |
<!-- MOAT:TABLE:END -->

## Layout

See `projects/README.md` for the per-project files, `PORTING_GUIDE.md` for porting strategy and fault classes, and `CLAUDE.md` for how a Claude CLI drives the pipeline.
