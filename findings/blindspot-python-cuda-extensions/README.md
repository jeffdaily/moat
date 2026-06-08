# Discovery blind spot: Python repos with a hidden CUDA extension

This is a one-off systematic sweep to close a known gap in MOAT discovery: GitHub
repositories whose primary language is Python (or otherwise not Cuda) that ship a
native GPU extension built from one or more `.cu`/`.cuh` device-code files, and
whose name, description, and topics do NOT contain "cuda" or "gpu". MOAT's
automated discovery (`utils/discover.py`) keys on either CUDA-language-dominance
(`language:Cuda`) or "cuda"/"gpu" in repo metadata, so this class was invisible.

Result: 275 verified candidates in `data/blindspot_candidates.json`
(`source: "blindspot-python-cuda-ext"`). 93% are Python-primary; the rest are
C++/Jupyter/Assembly/Cuda-primary repos that still lack "cuda"/"gpu" in metadata.
None of them are already adopted, already in `data/candidates.json`, already
skipped in `data/dispositions.json`, or already first-class ROCm-supported.

This sweep is read-only on every external repo. It did not scaffold or adopt
anything; that remains a manual `python3 utils/moatlib.py scaffold <owner/repo>`
step after triage.

## 1. The gap, precisely

`utils/discover.py` has two passes:

- The metadata pass (`gh search repos`) indexes only name + description +
  topics. A CUDA library is invisible to it if its dominant language is not Cuda
  AND none of those three fields contains "cuda"/"gpu" -- even when the repo
  ships hundreds of KB of `.cu`. The two recall queries `cuda language:C++` and
  `cuda language:Python` do not help here, because they still require the literal
  token "cuda" to appear in the indexed metadata.
- The `[code_search]` pass (`discover.py --code-search`) was written to close
  exactly this gap by matching CUDA SOURCE via the REST code-search API. But in
  practice it has never durably populated `candidates.json`: there are zero
  `code:`-tagged records in the current file, no `data/.code_search_cache.json`
  on disk, and the git history shows a single short run (commit c674850, "persist
  6 code-search candidates") that a later metadata-only "relaxed sweep" (commit
  1a07e59) overwrote -- the metadata pass rewrites `candidates.json` wholesale,
  so it clobbers code-search findings unless the two are merged.

Net effect: the exemplars slipped through entirely. `facebookresearch/pytorch3d`
(9891 stars, 328 KB of `.cu`, empty topics, no "cuda" in its description) and
`nerfstudio-project/nerfacc` (1457 stars) are NOT in `data/candidates.json` --
they were adopted by hand. This sweep confirms there are many more like them.

## 2. Method (repeatable)

Five complementary tactics. GitHub code search caps at 1000 results/query
(10 pages x 100) and is relevance-ordered (no star sort), with a ~10 req/min
primary limit plus a stricter undocumented secondary/abuse limit; every pass
below throttled 12-16 s between code-search calls and used core-API (5000/hr)
for the per-repo verification.

### Tactic 1 -- build-system signature code search (most precise)

A torch CUDA extension almost always declares itself in `setup.py` or via the
JIT loader. These queries find the declaration directly, in any repo regardless
of metadata or dominant language. Run via `gh api -X GET search/code -f q=...`:

```
CUDAExtension extension:py filename:setup
from torch.utils.cpp_extension import CUDAExtension
BuildExtension CUDAExtension extension:py
cpp_extension.load extension:py
load_inline cuda_sources extension:py
nvcc extra_compile_args extension:py
CUDA_HOME extension:py filename:setup
name=cuda path:csrc extension:cu
AT_DISPATCH_FLOATING_TYPES extension:cu
torch extension cuda path:csrc extension:cu
pybind11 cuda extension:cu
TORCH_EXTENSION_NAME extension:cu
```

This surfaced 5325 unique repos. The narrow setup.py/csrc-scoped queries are the
high-precision core; the broad `extension:cu` queries add recall. The biggest
hidden-extension finds came from here, including the non-rendering ones the topic
sweeps below would never reach (`HazyResearch/flash-fft-conv`,
`HazyResearch/H3`, `Felix-Petersen/difflogic`, `HanGuo97/flute`,
`mit-han-lab/llm-awq`, the PaddlePaddle stack).

### Tactic 2 -- `.cu`-in-Python-repo, verified against the git tree

For every repo surfaced, confirm real device code by listing the default-branch
tree and filtering for `.cu`/`.cuh`:

```
gh api repos/{o}/{r}/git/trees/{default_branch}?recursive=1 \
  --jq '[.tree[]?.path | select(test("\\.cuh?$"))]'
```

Keep only repos with at least one `.cu`/`.cuh` and the blind-spot signature
(Python-primary OR no "cuda"/"gpu" in name/description/topics).

### Tactic 3 -- ecosystem topic sweeps + language filter

```
gh search repos --language=Python --topic=<T> --sort=stars --limit=60
```

for T in: pytorch, gaussian-splatting, 3d-gaussian-splatting, nerf,
neural-rendering, volume-rendering, radiance-field, differentiable-rendering,
inverse-rendering, 3d-reconstruction, point-cloud, slam, ray-tracing,
molecular-dynamics, computational-chemistry, bioinformatics, deep-learning,
attention-mechanism. Each repo then went through the Tactic-2 tree check. (The
generic `deep-learning`/`pytorch` topics are noisy -- most hits are pure-Python
libraries that call torch's built-in CUDA -- so the tree check is what makes them
useful; the narrow rendering/geometry topics yield ~31% real-`.cu` hits.)

### Tactic 4 -- sibling/org expansion from known hits

```
gh search repos --owner=<ORG> --language=Python --sort=stars --limit=40
```

for orgs/authors that ship CUDA extensions: facebookresearch, open-mmlab,
mit-han-lab, traveller59, nerfstudio-project, rusty1s (PyG), ashawkey,
lucidrains, NVIDIA-AI-IOT (NVIDIA orgs are then dropped). This caught the PyG
extension family (`pytorch_scatter`/`_sparse`/`_cluster`/`_spline_conv`) and the
OpenMMLab toolboxes.

### Tactic 5 -- curated lists (web), then verify on GitHub

`MrNeRF/awesome-3D-gaussian-splatting` and web searches for "custom CUDA kernel
PyTorch extension library" seeded concrete names (`pointrix-project/msplat`,
`GAP-LAB-CUHK-SZ/gaustudio`, `nyu-systems/Grendel-GS`, `satyajitghana/torch-point-ops`,
`yifita/pytorch_points`), each then run through the Tactic-2 tree check.

### Filtering applied to every candidate

- Has >= 1 real `.cu`/`.cuh` in its own tree (not vendored).
- Blind-spot signature: Python-primary OR no "cuda"/"gpu" in name/desc/topics.
- Not archived, not a fork.
- Not already adopted (`projects/`), not in `data/candidates.json`, not skipped
  in `data/dispositions.json`, not in an excluded org (NVIDIA*, ROCm*, NVlabs,
  nv-tlabs).
- Not already first-class ROCm-supported: a tree scan for `hipcc`/`rocm`/`/hip/`/
  `/amd/` plus an `org:ROCm` fork search. 5 repos were dropped on this gate
  (`PaddlePaddle/Paddle`, `facebookresearch/xformers`, `HawkAaron/warp-transducer`
  [has a ROCm fork], `IsaacRe/vllm-kvcompress`, `MetaX-MACA/vLLM-metax`). NOTE:
  the naive `_hip|hip_` substring match also flagged false positives -- "ship_"
  in shell-script names and a vendored Eigen `arch/HIP` tree -- which were
  verified by hand and restored; a real fix needs the tighter regex plus a
  vendored-tree exclusion (see section 4).

## 3. Candidates

Full ranked list: `data/blindspot_candidates.json` (275 entries, by stars then
confidence). Classification is a heuristic hint, not a filter:
reusable-library (79), library-from-paper (32), application/other (105),
paper-demo (59). The library tiers are the high-MOAT-value subset; paper-demo
repos are one-off research code and lower priority.

### Recommended first 5 to scaffold

Chosen for reach (a base library unlocks a family), reuse, and port tractability:

1. **open-mmlab/mmcv** (6444 stars, 111 `.cu`) -- the CUDA foundation under the
   whole OpenMMLab family (mmdetection3d, mmpose, mmaction, mmdeploy, ...).
   Porting it has the most downstream leverage of anything in the list.
2. **rusty1s/pytorch_scatter** (1738) + the sibling PyG extensions
   **pytorch_sparse** (1100), **pytorch_cluster** (926) -- foundational for the
   entire PyTorch Geometric ecosystem; small, well-scoped `.cu`, ideal tractable
   first ports.
3. **traveller59/spconv** (2276) -- Spatial Sparse Convolution, a hard dependency
   of OpenPCDet, Pointcept, and most LiDAR 3D-detection repos.
4. **Pointcept/Pointcept** (3066, 21 `.cu`) -- widely used point-cloud perception
   codebase, self-contained CUDA ops.
5. **mit-han-lab/llm-awq** (3556, 19 `.cu`) -- high-impact LLM weight quantization
   (AWQ); directly AMD-relevant and frequently requested on ROCm.

(`facebookresearch/detectron2`, at 34531 stars, tops the list, but its `.cu`
footprint is small/optional and much already runs on CPU/torch built-ins; it is
high-visibility but a thinner CUDA port than the five above.)

### Tier 1 -- reusable libraries / toolkits (highest MOAT value)

79 repos. Top 40:

| Stars | Repo | Lang | .cu | Class | Description |
|---:|---|---|---:|---|---|
| 34531 | facebookresearch/detectron2 | Python | 7 | reusable-library | Detectron2 is a platform for object detection, segmentation and other |
| 12951 | PaddlePaddle/PaddleNLP | Python | 170 | reusable-library | Easy-to-use and powerful LLM and SLM library with awesome model zoo. |
| 9338 | PaddlePaddle/PaddleSeg | Python | 2 | reusable-library | Easy-to-use image segmentation library with awesome pre-trained models |
| 6444 | open-mmlab/mmcv | Python | 111 | reusable-library | OpenMMLab Computer Vision Foundation |
| 6441 | open-mmlab/mmdetection3d | Python | 5 | reusable-library | OpenMMLab's next-generation platform for general 3D object detection. |
| 5610 | open-mmlab/OpenPCDet | Python | 16 | reusable-library | OpenPCDet Toolbox for LiDAR-based 3D Object Detection. |
| 5381 | facebookresearch/sapiens | Python | 111 | reusable-library | High-resolution models for human tasks. |
| 3843 | open-mmlab/mmpretrain | Python | 2 | reusable-library | OpenMMLab Pre-training Toolbox and Benchmark |
| 3125 | open-mmlab/mmdeploy | Python | 32 | reusable-library | OpenMMLab Model Deployment Framework |
| 3117 | open-mmlab/mmskeleton | Python | 2 | reusable-library | A OpenMMLAB toolbox for human pose estimation, skeleton-based action |
| 3066 | Pointcept/Pointcept | Python | 21 | reusable-library | Pointcept: Perceive the world with sparse points, a codebase for point cloud |
| 2276 | traveller59/spconv | Python | 1 | reusable-library | Spatial Sparse Convolution Library |
| 2259 | CoinCheung/pytorch-loss | Python | 14 | reusable-library | label-smooth, amsoftmax, partial-fc, focal-loss, triplet-loss, lovasz |
| 2212 | ashawkey/torch-ngp | Python | 5 | reusable-library | A pytorch CUDA extension implementation of instant-ngp (sdf and nerf) |
| 2017 | open-mmlab/mmgeneration | Python | 6 | reusable-library | MMGeneration is a powerful toolkit for generative models, based on PyTorch |
| 1876 | open-mmlab/mmaction | Python | 6 | reusable-library | An open-source toolbox for action understanding based on PyTorch |
| 1811 | yerfor/GeneFacePlusPlus | Python | 10 | reusable-library | GeneFace++: Generalized and Stable Real-Time 3D Talking Face Generation |
| 1742 | GAP-LAB-CUHK-SZ/gaustudio | Jupyter Notebook | 4 | reusable-library | A Modular Framework for 3D Gaussian Splatting and Beyond |
| 1738 | rusty1s/pytorch_scatter | Python | 7 | reusable-library | PyTorch Extension Library of Optimized Scatter Operations |
| 1555 | V2AI/Det3D | Python | 13 | reusable-library | World's first general purpose 3D object detection codebase. |
| 1431 | yangyanli/PointCNN | Python | 7 | reusable-library | PointCNN: Convolution On X-Transformed Points (NeurIPS 2018) |
| 1422 | open-mmlab/mmhuman3d | Python | 1 | reusable-library | OpenMMLab 3D Human Parametric Model Toolbox and Benchmark |
| 1293 | ShichenLiu/SoftRas | Python | 4 | reusable-library | Project page of paper "Soft Rasterizer: A Differentiable Renderer" |
| 1100 | rusty1s/pytorch_sparse | Python | 7 | reusable-library | PyTorch Extension Library of Optimized Autograd Sparse Matrix Operations |
| 1063 | OFA-Sys/ONE-PEACE | Python | 10 | reusable-library | A general representation model across vision, audio, language modalities |
| 926 | rusty1s/pytorch_cluster | C++ | 8 | reusable-library | PyTorch Extension Library of Optimized Graph Cluster Algorithms |
| 850 | DengKaiCQ/VGGT-Long | Python | 14 | reusable-library | Official implement of VGGT-Long |
| 805 | OpenDCAI/OpenWorldLib | Python | 8 | reusable-library | Unified Codebase for Advanced World Models. |
| 797 | caiyuanhao1998/SAX-NeRF | Python | 1 | reusable-library | Structure-Aware Sparse-View X-ray 3D Reconstruction (CVPR 2024) |
| 783 | Felix-Petersen/difflogic | Python | 1 | reusable-library | A Library for Differentiable Logic Gate Networks |
| 680 | HKUST-SAIL/RaDe-GS | C++ | 14 | reusable-library | RaDe-GS: Rasterizing Depth in Gaussian Splatting |
| 642 | MIC-DKFZ/nnDetection | Python | 1 | reusable-library | nnDetection is a self-configuring framework for 3D medical detection |
| 638 | PaddlePaddle/Paddle3D | Python | 45 | reusable-library | A 3D computer vision development toolkit based on PaddlePaddle |
| 637 | Tessellate-Imaging/Monk_Object_Detection | Jupyter Notebook | 69 | reusable-library | A one-stop repository for low-code easily-installable object detection |
| 580 | Kai-46/ARF-svox2 | Python | 12 | reusable-library | Artistic Radiance Fields |
| 552 | DeepSceneSeg/EfficientPS | Python | 14 | reusable-library | PyTorch code for training EfficientPS for Panoptic Segmentation |
| 523 | HazyResearch/H3 | Assembly | 39 | reusable-library | Language Modeling with the H3 State Space Model |
| 447 | RaduAlexandru/permuto_sdf | Python | 18 | reusable-library | PermutoSDF: Fast Multi-View Reconstruction (CVPR'23) |
| 444 | Sense-X/X-Temporal | Python | 1 | reusable-library | A general video understanding codebase from SenseTime X-Lab |
| 432 | JIA-Lab-research/Stratified-Transformer | Python | 19 | reusable-library | Stratified Transformer for 3D Point Cloud Segmentation (CVPR 2022) |

### Tier 2 -- libraries shipped from a paper (reusable kernels, paper-scoped)

32 repos. Top 25:

| Stars | Repo | Lang | .cu | Class | Description |
|---:|---|---|---:|---|---|
| 3556 | mit-han-lab/llm-awq | Python | 19 | library-from-paper | [MLSys 2024 Best Paper] AWQ: Activation-aware Weight Quantization |
| 2832 | OpenGVLab/InternImage | Python | 11 | library-from-paper | [CVPR 2023 Highlight] InternImage: Large-Scale Vision Foundation Models |
| 1134 | Tencent-Hunyuan/HunyuanWorld-Mirror | Python | 23 | library-from-paper | [ICML 2026] WorldMirror: Fast and Universal 3D reconstruction |
| 1088 | lizhe00/AnimatableGaussians | Python | 13 | library-from-paper | [CVPR 2024] Animatable Gaussians: Pose-dependent Gaussian maps |
| 805 | nianticlabs/acezero | Python | 13 | library-from-paper | [ECCV 2024 Oral] ACE0: learning-based structure-from-motion |
| 735 | OpenGVLab/DCNv4 | Python | 12 | library-from-paper | [CVPR 2024] Deformable Convolution v4 |
| 671 | NJU-3DV/Relightable3DGaussian | Python | 13 | library-from-paper | [ECCV2024] Relightable 3D Gaussian: Real-time Point Cloud Relighting |
| 501 | MrTornado24/Next3D | Python | 7 | library-from-paper | [CVPR 2023 Highlight] Next3D: Generative Neural Texture Rasterization |
| 460 | Anttwo/MILo | Python | 17 | library-from-paper | [SIGGRAPH Asia 2025 TOG] MILo: Mesh-In-the-Loop |
| 450 | Haiyang-W/DSVT | Python | 16 | library-from-paper | [CVPR2023] DSVT: Dynamic Sparse Voxel Transformer |
| 434 | Pointcept/PointTransformerV2 | Python | 19 | library-from-paper | [NeurIPS'22] PyTorch implementation of PTv2 |
| 425 | SqueezeAILab/KVQuant | Python | 32 | library-from-paper | [NeurIPS 2024] KVQuant: 10M Context Length LLM Inference |
| 298 | RongLiu-Leo/beta-splatting | Python | 23 | library-from-paper | [SIGGRAPH'25] Deformable Beta Splatting |
| 283 | PJLab-ADG/LoGoNet | Python | 18 | library-from-paper | [CVPR2023] LoGoNet: Accurate 3D Object Detection Local-to-Global |
| 255 | LMD0311/HERMES | Python | 28 | library-from-paper | [ICCV 2025] HERMES: Unified Self-Driving World Model |
| 223 | zju3dv/PVO | Python | 44 | library-from-paper | [CVPR 2023] PVO: Panoptic Visual Odometry |
| 211 | Kunhao-Liu/StyleGaussian | Python | 10 | library-from-paper | [SIGGRAPH Asia 2024] StyleGaussian: Instant 3D Style Transfer |
| 191 | Eaphan/GLENet | Python | 14 | library-from-paper | GLENet: 3D Object Detectors with Generative Label Uncertainty |
| 176 | hustvl/Dynamic-2DGS | Python | 10 | library-from-paper | [ACM MM 2025] Dynamic 2D Gaussians |
| 172 | mit-han-lab/fastrl | Python | 102 | library-from-paper | [ASPLOS'26] Efficient Reasoning RL Training (note: vendored sglang AMD) |
| 160 | PKU-YuanGroup/HoloTime | Python | 22 | library-from-paper | [ACM MM 2025] HoloTime: Panoramic 4D from Video Diffusion |
| 142 | mit-han-lab/flatformer | Python | 16 | library-from-paper | [CVPR'23] FlatFormer: Flattened Window Attention for Point Clouds |
| 138 | HpWang-whu/RoReg | Python | 41 | library-from-paper | [TPAMI 2023] RoReg: Pairwise Point Cloud Registration |
| 126 | JIA-Lab-research/Ref-NPR | Python | 12 | library-from-paper | [CVPR 2023] Ref-NPR: Reference-Based Non-PhotoRealistic Radiance Fields |
| 124 | MarkHershey/PSIVG | Python | 17 | library-from-paper | [CVPR 2026] Physical Simulator In-the-Loop Video Generation |

### Cross-domain (non-rendering) finds -- breadth beyond 3D/vision

These are the repos the rendering/geometry topic sweeps would never reach; they
came from the build-system signature search (Tactic 1) and matter because they
prove the gap is not just a 3D-vision phenomenon. Highlights: `HazyResearch/H3`
and `HazyResearch/flash-fft-conv` (state-space / FFT-conv kernels),
`Felix-Petersen/difflogic` (differentiable logic gates), `HanGuo97/flute`
(LUT-quantized matmul), `NetEase-FuXi/EETQ` and `SqueezeAILab/KVQuant` (LLM
quantization), `MIC-DKFZ/nnDetection` (medical imaging),
`ClementPinard/Pytorch-Correlation-extension` (optical-flow correlation),
`HawkAaron/warp-transducer` (RNN-T -- already has a ROCm fork, excluded).

### Edge cases -- Cuda-dominant but no "cuda"/"gpu" in metadata

These 5 are Cuda-language-dominant, so `discover.py`'s existing `language:Cuda`
query WOULD reach them; they are included because their name/description/topics
still lack "cuda"/"gpu", which is the other half of the blind-spot signature, and
they are genuine un-ported CUDA-extension libraries. Treat them as lower-priority
within this list since the existing pass already covers the mechanism.

| Stars | Repo | Lang | .cu | Description |
|---:|---|---|---:|---|
| 487 | ashawkey/diff-gaussian-rasterization | Cuda | 4 | (diff gaussian rasterizer; many splatting repos vendor a fork of this) |
| 374 | ThibaultGROUEIX/ChamferDistancePytorch | Cuda | 4 | Chamfer Distance in PyTorch with f-score |
| 327 | SarahWeiii/diso | Cuda | 2 | Differentiable Iso-Surface Extraction Package (DISO) |
| 213 | pointrix-project/msplat | Cuda | 6 | A modular differential gaussian rasterization library |
| 150 | ByteDance-Seed/decoupleQ | Cuda | 3 | A quantization algorithm for LLM |

### Lower-confidence / unsure

- A handful of "application/other" repos (e.g. `Kedreamix/Linly-Talker`) bundle
  `.cu` from a vendored sub-project rather than building their own extension;
  they are kept but flagged `application/other`, not library-class.
- `mit-han-lab/fastrl` and `InnovatorLM/Innovator-VL` carry AMD/ROCm paths ONLY
  inside a vendored `3rdparty/sglang` tree, not in their own code; they passed
  the ROCm gate (their own port is absent) but are LLM-serving derivatives of
  uncertain standalone value.
- `Tessellate-Imaging/Monk_Object_Detection` (69 `.cu`) aggregates many upstream
  detectors' kernels; high `.cu` count overstates how much is original.

## 4. Recommended fix to utils/discover.py

The blind-spot closer already exists conceptually (the `[code_search]` pass); it
needs to be made effective and durable. Concretely:

1. **Make the code-search pass idempotent and non-clobbering.** The metadata pass
   currently rewrites `candidates.json` from scratch, erasing code-search
   findings. Persist code-search candidates so a later metadata run merges rather
   than overwrites -- either always run the code-search pass as part of the
   normal `discover.py` run (not only behind `--code-search`), or have the
   metadata pass read-modify-write `candidates.json` (union by `full_name`)
   instead of truncating it. The durable side-channel
   (`agent_space/new_records_postrun.json`) the code already writes should be the
   merge source.

2. **Add the build-system-signature queries to `[code_search]`.** The current
   `[code_search].queries` lean on `__global__`/`cudaMalloc`/AT_DISPATCH, which
   are broad. The highest-precision signal for THIS class is the extension
   declaration in `setup.py`: add `CUDAExtension extension:py filename:setup`,
   `BuildExtension CUDAExtension extension:py`, `CUDA_HOME extension:py
   filename:setup`, and `cpp_extension.load extension:py` near the FRONT of the
   query list (they page first, so the budget lands on them). These are what
   surfaced the PaddlePaddle / OpenMMLab / PyG / HazyResearch finds here.

3. **Verify `.cu` presence via the git tree, not only the languages API.** The
   languages-API byte count (`min_cuda_bytes`) misses repos where `.cu` is a tiny
   fraction of a large Python repo, and it cannot tell own-code from a vendored
   dependency. Add a tree check
   (`git/trees/{branch}?recursive=1` filtered to `\.cuh?$`) and keep a repo only
   if it has own-tree `.cu` outside `thirdparty/|third_party/|external/|extern/|
   /eigen|site-packages/`.

4. **Fix the ROCm-already-supported detector before reusing it.** A future
   "skip already-ported" gate must NOT use a loose `_hip|hip_` substring: it
   matches "ship_" in script names and vendored `arch/HIP` trees. Use
   `hipcc|\brocm\b|/hip/|\.hip$|hip\.cmake|/amd/` AND exclude vendored trees, plus
   an `org:ROCm` fork search. (This sweep found 5 genuine already-ROCm repos and
   4 false positives with the loose pattern.)

5. **Optionally seed a small set of high-yield Python orgs** (facebookresearch,
   open-mmlab, mit-han-lab, rusty1s, nerfstudio-project, ashawkey) into a
   per-org sweep -- cheap recall for the prolific extension authors, though the
   code-search signature pass already covers most of them topic-agnostically.

Reproducing this exact sweep: the per-repo verification helpers used here
(code-search harness, tree-based `.cu` check, ROCm-support check) were throwaway
scripts in `agent_space/` (gitignored); the queries in section 2 are sufficient
to regenerate the candidate set, and the per-repo filters in section 2 plus the
fixes above are what a permanent `discover.py` pass should encode.
