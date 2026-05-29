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
| cudf | [rapidsai/cudf](https://github.com/rapidsai/cudf) | - | todo | gated | gated |
| LMCache | [LMCache/LMCache](https://github.com/LMCache/LMCache) | - | todo | gated | gated |
| gsplat | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | - | todo | gated | gated |
| llm.c | [karpathy/llm.c](https://github.com/karpathy/llm.c) | - | todo | gated | gated |
| kaldi | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi) | - | todo | gated | gated |
| cuml | [rapidsai/cuml](https://github.com/rapidsai/cuml) | - | todo | gated | gated |
| CTranslate2 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | - | todo | gated | gated |
| alien | [chrxh/alien](https://github.com/chrxh/alien) | - | todo | gated | gated |
| lc0 | [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0) | - | todo | gated | gated |
| oneflow | [Oneflow-Inc/oneflow](https://github.com/Oneflow-Inc/oneflow) | - | todo | gated | gated |
| CV-CUDA | [CVCUDA/CV-CUDA](https://github.com/CVCUDA/CV-CUDA) | - | todo | gated | gated |
| cugraph | [rapidsai/cugraph](https://github.com/rapidsai/cugraph) | - | todo | gated | gated |
<!-- MOAT:TABLE:END -->

## Layout

See `projects/README.md` for the per-project files, `PORTING_GUIDE.md` for porting strategy and fault classes, and `CLAUDE.md` for how a Claude CLI drives the pipeline.
