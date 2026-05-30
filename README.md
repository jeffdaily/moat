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
| Project | Upstream | Fork | gfx90a | gfx1100 | gfx1151 |
| --- | --- | --- | --- | --- | --- |
| Open3D | [isl-org/Open3D](https://github.com/isl-org/Open3D) | - | todo | gated | gated |
| catboost | [catboost/catboost](https://github.com/catboost/catboost) | - | todo | gated | gated |
| cudf | [rapidsai/cudf](https://github.com/rapidsai/cudf) | - | todo | gated | gated |
| LMCache | [LMCache/LMCache](https://github.com/LMCache/LMCache) | - | todo | gated | gated |
| gsplat | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | [gsplat](https://github.com/jeffdaily/gsplat/tree/moat-port) | done | revalidate | validating |
| llm.c | [karpathy/llm.c](https://github.com/karpathy/llm.c) | - | todo | gated | gated |
| kaldi | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi) | - | todo | gated | gated |
| cuml | [rapidsai/cuml](https://github.com/rapidsai/cuml) | - | todo | gated | gated |
| CTranslate2 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | - | todo | gated | gated |
| alien | [chrxh/alien](https://github.com/chrxh/alien) | - | todo | gated | gated |
| mahout | [apache/mahout](https://github.com/apache/mahout) | - | todo | gated | gated |
| LichtFeld-Studio | [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) | - | todo | gated | gated |
| lc0 | [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0) | - | todo | gated | gated |
| oneflow | [Oneflow-Inc/oneflow](https://github.com/Oneflow-Inc/oneflow) | - | todo | gated | gated |
| arrayfire | [arrayfire/arrayfire](https://github.com/arrayfire/arrayfire) | - | todo | gated | gated |
| CV-CUDA | [CVCUDA/CV-CUDA](https://github.com/CVCUDA/CV-CUDA) | - | todo | gated | gated |
| cugraph | [rapidsai/cugraph](https://github.com/rapidsai/cugraph) | - | todo | gated | gated |
| mirage | [mirage-project/mirage](https://github.com/mirage-project/mirage) | - | todo | gated | gated |
| k2 | [k2-fsa/k2](https://github.com/k2-fsa/k2) | - | todo | gated | gated |
| raft | [rapidsai/raft](https://github.com/rapidsai/raft) | - | todo | gated | gated |
| heavydb | [heavyai/heavydb](https://github.com/heavyai/heavydb) | - | todo | gated | gated |
| rmm | [rapidsai/rmm](https://github.com/rapidsai/rmm) | - | todo | gated | gated |
| cuvs | [rapidsai/cuvs](https://github.com/rapidsai/cuvs) | - | todo | gated | gated |
| GPUMD | [brucefan1983/GPUMD](https://github.com/brucefan1983/GPUMD) | [GPUMD](https://github.com/jeffdaily/GPUMD/tree/moat-port) | done | done | validating |
| amgcl | [ddemidov/amgcl](https://github.com/ddemidov/amgcl) | [amgcl](https://github.com/jeffdaily/amgcl/tree/moat-port) | done | done | validating |
| CudaSift | [Celebrandil/CudaSift](https://github.com/Celebrandil/CudaSift) | [CudaSift](https://github.com/jeffdaily/CudaSift/tree/moat-port) | done | done | validating |
| cupoch | [neka-nat/cupoch](https://github.com/neka-nat/cupoch) | [cupoch](https://github.com/jeffdaily/cupoch/tree/moat-port) | done | done | validating |
| popsift | [alicevision/popsift](https://github.com/alicevision/popsift) | [popsift](https://github.com/jeffdaily/popsift/tree/moat-port) | done | done | validating |
| NATTEN | [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN) | - | todo | gated | gated |
| ElasticFusion | [mp3guy/ElasticFusion](https://github.com/mp3guy/ElasticFusion) | - | todo | gated | gated |
| cucim | [rapidsai/cucim](https://github.com/rapidsai/cucim) | - | todo | gated | gated |
| ZhiLight | [zhihu/ZhiLight](https://github.com/zhihu/ZhiLight) | - | todo | gated | gated |
| SpargeAttn | [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn) | - | todo | gated | gated |
| 3DUNDERWORLD-SLS-GPU_CPU | [theICTlab/3DUNDERWORLD-SLS-GPU_CPU](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU) | [3DUNDERWORLD-SLS-GPU_CPU](https://github.com/jeffdaily/3DUNDERWORLD-SLS-GPU_CPU/tree/moat-port) | done | done | validating |
| STRUMPACK | [pghysels/STRUMPACK](https://github.com/pghysels/STRUMPACK) | - | planned | gated | gated |
| AutoDock-GPU | [ccsb-scripps/AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) | [AutoDock-GPU](https://github.com/jeffdaily/AutoDock-GPU/tree/moat-port) | done | done | validating |
| FlashKDA | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) | - | todo | gated | gated |
| 3P-ADMM-PC2 | [Samarvivian/3P-ADMM-PC2](https://github.com/Samarvivian/3P-ADMM-PC2) | - | todo | gated | gated |
| unified-cache-management | [ModelEngine-Group/unified-cache-management](https://github.com/ModelEngine-Group/unified-cache-management) | - | todo | gated | gated |
| gtsam_points | [koide3/gtsam_points](https://github.com/koide3/gtsam_points) | - | todo | gated | gated |
| gpu4pyscf | [pyscf/gpu4pyscf](https://github.com/pyscf/gpu4pyscf) | - | todo | gated | gated |
| RXMesh | [owensgroup/RXMesh](https://github.com/owensgroup/RXMesh) | - | porting (blocked) | gated | gated |
| MPPI-Generic | [ACDSLab/MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) | [MPPI-Generic](https://github.com/jeffdaily/MPPI-Generic/tree/moat-port) | done | done | validating |
| sppark | [supranational/sppark](https://github.com/supranational/sppark) | - | todo | gated | gated |
| Gpufit | [gpufit/Gpufit](https://github.com/gpufit/Gpufit) | [Gpufit](https://github.com/jeffdaily/Gpufit/tree/moat-port) | done | done | validating |
| FlashRT | [LiangSu8899/FlashRT](https://github.com/LiangSu8899/FlashRT) | - | todo | gated | gated |
| evogp | [EMI-Group/evogp](https://github.com/EMI-Group/evogp) | - | todo | gated | gated |
| ffpa-attn | [xlite-dev/ffpa-attn](https://github.com/xlite-dev/ffpa-attn) | - | todo | gated | gated |
| FlashMoE | [osayamenja/FlashMoE](https://github.com/osayamenja/FlashMoE) | - | todo | gated | gated |
| LiteGS | [MooreThreads/LiteGS](https://github.com/MooreThreads/LiteGS) | - | todo | gated | gated |
| CUDA-L2 | [deepreinforce-ai/CUDA-L2](https://github.com/deepreinforce-ai/CUDA-L2) | - | todo | gated | gated |
| Fast-Poisson-Image-Editing | [Trinkle23897/Fast-Poisson-Image-Editing](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing) | [Fast-Poisson-Image-Editing](https://github.com/jeffdaily/Fast-Poisson-Image-Editing/tree/moat-port) | done | done | validating |
| icicle | [ingonyama-zk/icicle](https://github.com/ingonyama-zk/icicle) | - | todo | gated | gated |
| dietgpu | [facebookresearch/dietgpu](https://github.com/facebookresearch/dietgpu) | - | todo | gated | gated |
| bam | [ZaidQureshi/bam](https://github.com/ZaidQureshi/bam) | - | todo | gated | gated |
| sparser-faster-llms | [SakanaAI/sparser-faster-llms](https://github.com/SakanaAI/sparser-faster-llms) | - | todo | gated | gated |
| llmq | [IST-DASLab/llmq](https://github.com/IST-DASLab/llmq) | - | todo | gated | gated |
| LEAP | [llnl/LEAP](https://github.com/llnl/LEAP) | - | todo | gated | gated |
| ntransformer | [xaskasdf/ntransformer](https://github.com/xaskasdf/ntransformer) | - | todo | gated | gated |
| FaithC | [Luo-Yihao/FaithC](https://github.com/Luo-Yihao/FaithC) | - | todo | gated | gated |
| rmcl | [uos/rmcl](https://github.com/uos/rmcl) | - | todo | gated | gated |
| fused-ssim | [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) | [fused-ssim](https://github.com/jeffdaily/fused-ssim/tree/moat-port) | done | done | validating |
| gpuRIR | [DavidDiazGuerra/gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) | [gpuRIR](https://github.com/jeffdaily/gpuRIR/tree/moat-port) | done | done | validating |
| yalm | [andrewkchan/yalm](https://github.com/andrewkchan/yalm) | - | todo | gated | gated |
| mHC.cu | [AndreSlavescu/mHC.cu](https://github.com/AndreSlavescu/mHC.cu) | - | todo | gated | gated |
| EnvGS | [zju3dv/EnvGS](https://github.com/zju3dv/EnvGS) | - | todo | gated | gated |
| egg.c | [d0rc/egg.c](https://github.com/d0rc/egg.c) | - | todo | gated | gated |
| DiffPhysDrone | [HenryHuYu/DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone) | - | todo | gated | gated |
| libSGM | [fixstars/libSGM](https://github.com/fixstars/libSGM) | - | todo | gated | gated |
| CPM.cu | [OpenBMB/CPM.cu](https://github.com/OpenBMB/CPM.cu) | - | todo | gated | gated |
| marian-dev | [marian-nmt/marian-dev](https://github.com/marian-nmt/marian-dev) | - | todo | gated | gated |
| RWKV-CUDA | [BlinkDL/RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA) | - | todo | gated | gated |
| splatad | [carlinds/splatad](https://github.com/carlinds/splatad) | - | todo | gated | gated |
| cudaKDTree | [ingowald/cudaKDTree](https://github.com/ingowald/cudaKDTree) | [cudaKDTree](https://github.com/jeffdaily/cudaKDTree/tree/moat-port) | done | done | validating |
| Quest | [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest) | - | todo | gated | gated |
| PhoenixOS | [SJTU-IPADS/PhoenixOS](https://github.com/SJTU-IPADS/PhoenixOS) | - | todo | gated | gated |
| 3DGS-LM | [lukasHoel/3DGS-LM](https://github.com/lukasHoel/3DGS-LM) | - | todo | gated | gated |
| fp6_llm | [usyd-fsalab/fp6_llm](https://github.com/usyd-fsalab/fp6_llm) | - | todo | gated | gated |
| gaussian_splatting | [joeyan/gaussian_splatting](https://github.com/joeyan/gaussian_splatting) | - | todo | gated | gated |
| DDN-SLAM | [DrLi-Ming/DDN-SLAM](https://github.com/DrLi-Ming/DDN-SLAM) | - | todo | gated | gated |
| op43dgs | [LetianHuang/op43dgs](https://github.com/LetianHuang/op43dgs) | - | todo | gated | gated |
<!-- MOAT:TABLE:END -->

## Layout

See `projects/README.md` for the per-project files, `PORTING_GUIDE.md` for porting strategy and fault classes, and `CLAUDE.md` for how a Claude CLI drives the pipeline.
