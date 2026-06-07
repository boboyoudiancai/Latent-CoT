# Third-Party Notices

This repository builds on the open-source StarVLA codebase. The current public
project identity is **LaRA-VLA**, while many third-party notices preserved here
are already present in that upstream base. This file is intended to preserve
and summarize those visible attributions in the current repository state.

This repository is released under the MIT License at the repository root.
However, individual files may carry their own third-party copyright or license
headers, and those file-level notices should be preserved.

This file is a best-effort inventory based on the file headers and upstream
references visible in the current repository. It is not intended to claim a
full independent provenance reconstruction for every historical StarVLA file.

This file is informational only and is not legal advice.

## How To Read This File

- The repository as a whole builds on StarVLA.
- Some files in StarVLA already preserve third-party provenance and SPDX
  notices.
- This document records the most visible third-party notices currently present
  in this repository and flags a few items that deserve additional review.

## Third-Party Notices Preserved In This Repository

### 1. NVIDIA Isaac-GR00T / NVIDIA-authored Apache-2.0 components

Upstream:

- https://github.com/NVIDIA/Isaac-GR00T

License:

- Apache License 2.0

Local files that retain NVIDIA SPDX / Apache-2.0 headers:

- `laravla/dataloader/gr00t_lerobot/data_config.py`
- `laravla/dataloader/gr00t_lerobot/datasets.py`
- `laravla/dataloader/gr00t_lerobot/embodiment_tags.py`
- `laravla/dataloader/gr00t_lerobot/schema.py`
- `laravla/dataloader/gr00t_lerobot/video.py`
- `laravla/dataloader/gr00t_lerobot/transform/__init__.py`
- `laravla/dataloader/gr00t_lerobot/transform/base.py`
- `laravla/dataloader/gr00t_lerobot/transform/concat.py`
- `laravla/dataloader/gr00t_lerobot/transform/state_action.py`
- `laravla/dataloader/gr00t_lerobot/transform/video.py`
- `laravla/model/modules/action_model/flow_matching_head/__init__.py`
- `laravla/model/modules/action_model/flow_matching_head/action_encoder.py`
- `laravla/model/modules/action_model/flow_matching_head/cross_attention_dit.py`

Notes:

- These files already preserve Apache-2.0 SPDX headers in the repository.
- The current repository keeps those file-level notices as inherited from the
  StarVLA base and subsequent local modifications.

### 2. OpenAI diffusion repositories

Upstreams:

- https://github.com/openai/guided-diffusion
- https://github.com/openai/improved-diffusion
- https://github.com/openai/glide-text2im

License:

- MIT License

Local files that explicitly reference these upstreams:

- `laravla/model/modules/action_model/__init__.py`
- `laravla/model/modules/action_model/DiT_modules/diffusion_utils.py`
- `laravla/model/modules/action_model/DiT_modules/gaussian_diffusion.py`
- `laravla/model/modules/action_model/DiT_modules/respace.py`
- `laravla/model/modules/action_model/DiT_modules/timestep_sampler.py`

Notes:

- These files include in-file comments indicating they were modified from
  OpenAI diffusion repositories.
- The current repository preserves those provenance comments.

### 3. Meta DiT

Upstream:

- https://github.com/facebookresearch/DiT

License:

- The upstream DiT repository is released under the license distributed with
  that project; the vendored file in this repository retains Meta copyright
  notices and refers to the upstream license file.

Local file:

- `laravla/model/modules/action_model/DiT_modules/models.py`

Important note:

- This file is the most license-sensitive vendored component currently visible
  in the repository.
- Review the exact upstream DiT license terms carefully before any stable or
  commercially-positioned release.

### 4. Meta DINOv2

Upstream:

- https://github.com/facebookresearch/dinov2

License:

- Apache License 2.0 for the copied transform file, per the local header

Local file:

- `laravla/model/modules/dino_model/dino_transforms.py`

Notes:

- `laravla/model/modules/dino_model/dino.py` is a local wrapper that loads
  DINOv2 via `torch.hub`; it is not itself marked as a copied upstream file.

### 5. OpenVLA / Prismatic logging utility

Upstream:

- https://github.com/openvla/openvla
- https://github.com/TRI-ML/prismatic-vlms

License:

- MIT License

Local file:

- `laravla/training/trainer_utils/overwatch.py`

Notes:

- The file header explicitly says it was originally from the OpenVLA / Prismatic
  project and modified in this repository.

## Files Requiring Additional Provenance Review

The following files carry third-party copyright or "modified by" notices, but
their exact upstream file mapping is not fully documented inside this current
repository snapshot. They should be reviewed before a stable public release.

### A. NVIDIA-labeled action headers

Files:

- `laravla/model/modules/action_model/GR00T_ActionHeader.py`
- `laravla/model/modules/action_model/LayerwiseFM_ActionHeader.py`

Observed in local headers:

- NVIDIA copyright notice
- local modification notice

Recommended follow-up:

- confirm the exact upstream source file(s) in NVIDIA Isaac-GR00T or related
  NVIDIA releases
- confirm whether an SPDX identifier or a clearer upstream reference should be
  added to these files

### B. CogACT-labeled action header

File:

- `laravla/model/modules/action_model/DiTActionHeader.py`

Observed in local header:

- `Copyright 2025 CogACT`
- local modification notice

Recommended follow-up:

- confirm the exact upstream CogACT source file
- confirm the intended redistribution notice to keep alongside the repository
  MIT license

## Repository Policy For Preserved Third-Party Code

When keeping or modifying third-party-derived code in this repository:

1. Preserve upstream copyright and license headers.
2. Add an explicit "Modified by ..." note when changes are made locally.
3. Prefer linking the exact upstream project or file when known.
4. Do not remove SPDX identifiers from vendored files.
5. Record new vendored or adapted code in this notice file.

## Practical Release Checklist

Before a stable public release, verify:

- files that preserve third-party headers still retain their upstream notices
- the repository root `LICENSE` does not obscure file-level third-party licenses
- the Meta DiT vendored file has been reviewed for compatibility with the
  release you intend to make
- NVIDIA- and CogACT-labeled action-header files have explicit upstream mapping
