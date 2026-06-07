# LaRA-VLA Rename Plan

> Goal: remove `starVLA` as the active project identity from this repository and
> migrate the codebase to **LaRA-VLA**.
>
> Scope: this is not a single-file rename. It is a full migration covering:
> package metadata, Python namespace, source directory, shell scripts, config
> paths, docs, file names, generated artifacts, and user-facing project naming.
>
> Important rule:
> user-facing branding should become **LaRA-VLA**, while the Python package
> namespace should become `laravla`.

---

## Current Status

As of the current repository state:

- `R1` completed
- `R2` completed
- `R3` completed
- `R4` completed
- `R5` completed
- `R6` partially completed
- `R7` intentionally conservative; factual provenance is still preserved
- `R8` partially completed

Notes:

- The active source tree has already moved from `starVLA/` to `laravla/`.
- User-facing script names now use `run_laravla_*.sh`.
- User-facing asset names now use `laravla_*`.
- Remaining `StarVLA` occurrences are primarily:
  - provenance / legal text
  - this migration plan itself
  - retained Hugging Face model IDs

---

## 1. Naming Policy

This migration should use the following naming rules consistently.

### 1.1 Human-facing display name

Use:

- `LaRA-VLA`

Applies to:

- repository title
- top-level README
- docs
- script descriptions
- package description text
- result/report labels when human-facing

### 1.2 Python package namespace

Use:

- `laravla`

Applies to:

- source directory rename: `starVLA/` -> `laravla/`
- Python imports
- file-system code paths in scripts/docs
- argparse default config paths

### 1.3 Python distribution / package manager name

Recommended:

- `[project].name = "LaRA-VLA"`

Reason:

- it matches the public project branding
- Python package indexes normalize case and hyphens anyway

### 1.4 File and script naming

Recommended:

- shell script names use `laravla` in lowercase
- generated paths and result folders use `laravla` in lowercase

Examples:

- `scripts/run_starvla_bridge.sh` -> `scripts/run_laravla_bridge.sh`
- `scripts/run_starvla_libero.sh` -> `scripts/run_laravla_libero.sh`

---

## 2. Current Blast Radius

Current scan results in this repository show that the rename will touch a large
surface area.

### 2.1 Python import / namespace references

- approximately `53` occurrences of:
  - `from starVLA ...`
  - `import starVLA ...`

### 2.2 Path references

- approximately `85` occurrences of:
  - `starVLA/...`
  - docs and shell commands pointing into `starVLA/`

### 2.3 File and directory names currently containing `starVLA` / `starvla`

At minimum:

- `starVLA/`
- `starVLA.egg-info/`
- `scripts/run_starvla_bridge.sh`
- `scripts/run_starvla_libero.sh`
- `assets/starvla_LIBERO.png`
- `assets/starvla_simpleEnv.png`

### 2.4 Generated artifacts and caches

Generated directories also currently contain `StarVLA`:

- `qwen_cache/models--StarVLA--Qwen3-VL-4B-Instruct-Action`
- `starVLA.egg-info/`
- `__pycache__/`

These should not be hand-edited. They should be deleted and regenerated after
the rename.

---

## 3. High-Risk Rename Areas

The following areas need special care.

### 3.1 Source directory rename

Renaming:

- `starVLA/` -> `laravla/`

This is the core change and affects nearly every import path.

### 3.2 Config paths embedded in code

Several files still hardcode config paths such as:

- `starVLA/config/training/...`
- parser defaults pointing to old `starVLA/...` paths

These must be updated together with the directory rename.

### 3.3 Script file names

Current script names still expose `starvla`:

- `scripts/run_starvla_bridge.sh`
- `scripts/run_starvla_libero.sh`

These should be renamed, and every reference to them in docs and scripts should
be updated.

### 3.4 Legal / provenance text

Some `StarVLA` mentions are factual provenance, not just branding.

Examples:

- `THIRD_PARTY_NOTICES.md`
- `LICENSE`
- upstream-related wording in docs

These should **not** be mass-replaced blindly. They need case-by-case review.

### 3.5 Hugging Face model IDs

Some configs and helper files still reference model IDs under the `StarVLA/...`
namespace, for example:

- `StarVLA/Qwen3-VL-4B-Instruct-Action`

Decision for this migration:

- keep existing `StarVLA/...` Hugging Face model IDs unchanged
- do not rename or migrate model repository identifiers as part of the
  `starVLA` -> `laravla` codebase migration

Reason:

- these model IDs are external published assets
- keeping them avoids breaking compatibility with existing checkpoints,
  downloads, cache directories, and reproduction commands

This affects:

- training yaml configs
- upload scripts
- local cache folder names
- README examples

---

## 4. Recommended Execution Strategy

This should be done as a staged migration, not one giant blind search-replace.

### Phase R0: Freeze Naming Rules

Before editing files, explicitly freeze the naming rules:

- project display name: `LaRA-VLA`
- Python package namespace: `laravla`
- distribution name: `LaRA-VLA`
- shell script prefix: `laravla`

Deliverable:

- this plan document approved

### Phase R1: Update Repository Metadata

Files:

- `pyproject.toml`
- `README.md`
- `pyrightconfig.json`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

Tasks:

1. Change project name in `pyproject.toml`
2. Change project description from `StarVLA` to `LaRA-VLA`
3. Update any package-find exclusions from `starVLA/...` to `laravla/...`
4. Decide whether the root license copyright line should become:
   - `LaRA-VLA Team`
   - or a more specific author list
5. Keep provenance text in `THIRD_PARTY_NOTICES.md` accurate

Deliverable:

- repository metadata reflects LaRA-VLA as the active public identity

### Phase R2: Rename the Source Directory

Rename:

- `starVLA/` -> `laravla/`

Also remove generated stale artifacts later:

- `starVLA.egg-info/`
- stale `__pycache__/`

Deliverable:

- source code lives under `laravla/`

### Phase R3: Rewrite Python Imports

Update all imports:

- `from starVLA...` -> `from laravla...`
- `import starVLA...` -> `import laravla...`

Main impact areas:

- `laravla/training/*`
- `laravla/model/*`
- `laravla/dataloader/*`
- `deployment/*`
- `examples/*`

Deliverable:

- no runtime import path still depends on `starVLA`

### Phase R4: Rewrite File-System Paths

Update all path-like strings and CLI examples:

- `starVLA/config/...` -> `laravla/config/...`
- `starVLA/training/train.py` -> `laravla/training/train.py`

Main impact areas:

- shell scripts in `scripts/`
- README and docs
- parser defaults in Python files
- helper scripts and evaluation wrappers

Deliverable:

- no user-facing command points at `starVLA/...`

### Phase R5: Rename Scripts and Other File Names

Recommended renames:

- `scripts/run_starvla_bridge.sh` -> `scripts/run_laravla_bridge.sh`
- `scripts/run_starvla_libero.sh` -> `scripts/run_laravla_libero.sh`

Also review whether to rename:

- `assets/starvla_LIBERO.png`
- `assets/starvla_simpleEnv.png`

Deliverable:

- user-facing file names no longer expose `starvla`

### Phase R6: Update Documentation and Examples

Files:

- `README.md`
- `examples/LIBERO/README.md`
- `examples/SimplerEnv/README.md`
- `docs/*`

Tasks:

1. Replace the active project identity with `LaRA-VLA`
2. Replace package/path examples with `laravla/...`
3. Keep docs concise and aligned with the current maintained workflow
4. Remove any leftover wording that implies this repo is still published as
   `StarVLA`

Deliverable:

- docs present LaRA-VLA consistently

### Phase R7: Review Legal and Provenance Text

This phase should be manual and careful.

Do **not** blindly replace every `StarVLA` string in:

- `THIRD_PARTY_NOTICES.md`
- file-level copyright headers
- third-party provenance text
- historical attribution text

Instead:

1. Keep third-party notices truthful
2. Keep upstream attributions where legally or historically required
3. Replace only branding / package-identity mentions that are not part of
   preserved provenance

Deliverable:

- no misleading provenance edits

### Phase R8: Clean Generated Artifacts

After the rename:

- delete `starVLA.egg-info/`
- remove stale `__pycache__/`
- optionally clear stale `qwen_cache` entries if you do not want visible
  `StarVLA` cache folders locally

Note:

- Hugging Face cache folder names are derived from model IDs; if you continue
  to use `StarVLA/...` model IDs, those cache names will continue to exist
  locally

Deliverable:

- no stale generated artifacts still refer to `starVLA`

---

## 5. Concrete File Groups To Touch

### 5.1 Metadata

- `pyproject.toml`
- `pyrightconfig.json`
- `LICENSE`
- `README.md`
- `THIRD_PARTY_NOTICES.md`

### 5.2 Source tree

- everything currently under `starVLA/`

### 5.3 Launchers and scripts

- `scripts/run_bridge_multistage.sh`
- `scripts/run_libero_multistage.sh`
- `scripts/run_starvla_bridge.sh`
- `scripts/run_starvla_libero.sh`
- `examples/LIBERO/eval_libero_all.sh`
- `examples/LIBERO/run_all_ckpts_libero_all.sh`
- `examples/SimplerEnv/bridge_eval.sh`
- `examples/SimplerEnv/run_all_ckpts_bridge.sh`

### 5.4 Docs and plans

- `README.md`
- `examples/LIBERO/README.md`
- `examples/SimplerEnv/README.md`
- `docs/open_source_release_schedule.md`
- `docs/config_and_training_cleanup_plan.md`
- `assets/intro_v1.md`

### 5.5 Tool defaults and parser paths

Examples already known:

- `laravla/training/train.py` default config path
- `laravla/model/framework/QwenGR00T.py` default config path
- `laravla/model/modules/vlm/QWen2_5.py` default config path
- `laravla/dataloader/lerobot_datasets.py` default config path

### 5.6 Model / org naming references

Examples already known:

- `StarVLA/Qwen3-VL-4B-Instruct-Action`
- `StarVLA/Qwen3VL-GR00T-Bridge-RT-1`

Policy for this rename:

- keep these model IDs for compatibility
- do not rewrite these identifiers during the namespace / path migration

---

## 6. Things That Should Not Be Blindly Replaced

The following should be reviewed manually instead of mass-replaced:

### 6.1 Third-party file headers

Files with preserved third-party notices should keep those notices intact.

### 6.2 Copyright ownership

If a file says:

- `Copyright (c) StarVLA Team`
- `Copyright 2025 starVLA community`

decide carefully whether that line should be changed.

This is not the same as changing a product name in the README.

### 6.3 Upstream provenance statements

If a statement means:

- "this repo builds on StarVLA"

it may still be factually correct, even after public branding becomes
LaRA-VLA.

---

## 7. Validation Checklist After The Rename

After the migration, run at least the following checks.

### 7.1 Import checks

```bash
python -c "from laravla.training.train import main; print('OK')"
python -c "from laravla.model.framework import build_framework; print('OK')"
```

### 7.2 YAML load checks

```bash
python - <<'PY'
from omegaconf import OmegaConf
for path in [
    "laravla/config/training/bridge_lerobot_stage2.yaml",
    "laravla/config/training/libero_all_ecot_stage4.yaml",
]:
    OmegaConf.load(path)
    print("OK:", path)
PY
```

### 7.3 Shell syntax checks

```bash
bash -n scripts/run_bridge_multistage.sh
bash -n scripts/run_libero_multistage.sh
bash -n examples/LIBERO/eval_libero_all.sh
bash -n examples/SimplerEnv/bridge_eval.sh
```

### 7.4 Residual-reference scans

```bash
rg -n "\\bstarVLA\\b|from starVLA|import starVLA|starVLA/" .
rg -n "run_starvla|starvla_" .
```

Expected result:

- only deliberate historical/provenance mentions remain, if any

### 7.5 Packaging checks

```bash
pip install -e .
python -c "import laravla; print('OK')"
```

---

## 8. Recommended Commit Strategy

Do not do the entire rename in one opaque commit.

Recommended sequence:

1. `rename: define LaRA-VLA naming policy and metadata`
2. `rename: move source package from starVLA to laravla`
3. `rename: rewrite Python imports to laravla`
4. `rename: update scripts, config paths, and docs`
5. `rename: rename user-facing scripts and assets`
6. `cleanup: remove stale starVLA build artifacts and residual refs`

---

## 9. My Recommendation

If the goal is truly "no active StarVLA traces in the product identity", then
the correct move is:

- perform a full namespace rename
- not just edit `pyproject.toml`

Changing only the package metadata would leave the repository in an awkward
half-renamed state:

- public name says `LaRA-VLA`
- code still imports `starVLA`
- scripts still point to `starVLA/...`
- file names still expose `starvla`

That state is worse than either staying on `starVLA` temporarily or doing the
rename properly.

---

## 10. Immediate Next Step

Before applying code changes, the next concrete task should be:

1. approve the naming policy in Section 1
2. approve the execution order in Section 4
3. then start with Phase R1 and Phase R2 together

That will avoid a half-renamed repository and keep the migration coherent.
