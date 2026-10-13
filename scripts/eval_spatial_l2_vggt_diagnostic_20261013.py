#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from bisect import bisect_right
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = Path("/share/home/u25505/third_party_vggt")
sys.path.insert(0, str(ROOT))
if VGGT_ROOT.is_dir():
    sys.path.insert(0, str(VGGT_ROOT))

from laravla.dataloader.lerobot_datasets import get_vla_dataset
from laravla.model.framework.base_framework import baseframework
from laravla.model.modules.vlm.vggt_teacher import build_vggt_transform
from vggt.models.vggt import VGGT


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-indices", default="22788,136732,250676")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vggt-weights", required=True)
    p.add_argument("--max-ply-points", type=int, default=100000)
    p.add_argument("--max-plot-points", type=int, default=18000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-latents", action="store_true")
    return p.parse_args()


def move(v, device):
    if torch.is_tensor(v):
        return v.to(device=device)
    if isinstance(v, dict):
        return {k: move(x, device) for k, x in v.items()}
    if isinstance(v, list):
        return [move(x, device) for x in v]
    if isinstance(v, tuple):
        return tuple(move(x, device) for x in v)
    return v


def dataset_index(dataset):
    if not hasattr(dataset, "datasets"):
        return None, None
    parts = list(dataset.datasets)
    offsets = [0]
    for part in parts:
        if hasattr(part, "transforms") and hasattr(part.transforms, "eval"):
            part.transforms.eval()
        offsets.append(offsets[-1] + len(part))
    return parts, offsets


def sample_at(dataset, parts, offsets, index):
    if parts is None:
        return dataset[index], "dataset"
    pi = bisect_right(offsets, index) - 1
    part = parts[pi]
    trajectory, frame = part.all_steps[index - offsets[pi]]
    data = part.transforms(part.get_step_data(trajectory, frame))
    return part._build_sample_from_data(data, trajectory, frame), str(
        getattr(part, "dataset_name", f"dataset_{pi}")
    )


@torch.inference_mode()
def extract(model, samples, device):
    q = model.qwen_vl_interface
    images = [s["image"] for s in samples]
    texts = [s.get("lang", s.get("language", "")) for s in samples]
    keys = [s.get("spatial_cache_key") for s in samples]
    inputs = move(q.build_qwenvl_inputs(images=images, instructions=texts), device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = q(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1]
    n = int(q.spatial_target_tokens)
    base = hidden[inputs["input_ids"] == int(q.spatial_base_token_id)].view(len(samples), n, -1)
    if str(q.spatial_view_mode) == "base_only":
        tokens = base[:, None]
    else:
        wrist = hidden[inputs["input_ids"] == int(q.spatial_wrist_token_id)].view(len(samples), n, -1)
        tokens = torch.stack((base, wrist), dim=1)
    projector = getattr(q, "spatial_align_projector", None)
    if projector is not None:
        parameter = next(projector.parameters())
        tokens = tokens.to(dtype=parameter.dtype)
    projected = q.project_spatial_prediction(tokens)
    target = q.get_external_spatial_target(images=images, device=device, cache_keys=keys)
    return images, tokens.float().cpu(), projected.float().cpu(), target.float().cpu(), str(q.spatial_view_mode)


def lift(tokens, side):
    b, v, n, c = tokens.shape
    small = int(math.sqrt(n))
    if small * small != n:
        raise ValueError(f"non-square latent token count {n}")
    grid = tokens.float().reshape(b * v, small, small, c).permute(0, 3, 1, 2)
    dense = F.interpolate(grid, (side, side), mode="bilinear", align_corners=True)
    return dense.permute(0, 2, 3, 1).reshape(b, v, side * side, c)


def replace(layers, dense, patch_start, layer_indices):
    selected = {int(index) % len(layers) for index in layer_indices}
    out = []
    for index, layer in enumerate(layers):
        if layer is None:
            out.append(None)
            continue
        copy = layer.clone()
        if index in selected:
            if copy.shape[2] - patch_start != dense.shape[2]:
                raise ValueError("VGGT patch count mismatch")
            copy[:, :, patch_start:, :] = dense.to(device=copy.device, dtype=copy.dtype)
        out.append(copy)
    return out


def ply(path, points, colors):
    points = np.asarray(points, dtype="<f4")
    colors = np.asarray(colors, dtype=np.uint8)
    data = np.empty(len(points), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(data)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def cloud(points, conf, image, limit, rng):
    points = np.asarray(points, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    if conf.ndim == 3 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if points.ndim == 4:
        points = points[0]
    colors = np.clip(np.asarray(image).transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    mask = np.isfinite(points).all(-1) & np.isfinite(conf)
    if not mask.any():
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    mask &= conf >= np.quantile(conf[mask], 0.35)
    points, colors = points[mask], colors[mask]
    if len(points) > limit:
        ids = rng.choice(len(points), limit, replace=False)
        points, colors = points[ids], colors[ids]
    return points, colors


def pmetrics(t, s):
    t, s = np.asarray(t, np.float32), np.asarray(s, np.float32)
    mask = np.isfinite(t).all(-1) & np.isfinite(s).all(-1)
    if not mask.any():
        return {"valid_fraction": 0.0}
    d = t[mask] - s[mask]
    l2 = np.linalg.norm(d, axis=-1)
    return {
        "valid_fraction": float(mask.mean()),
        "point_rmse": float(np.sqrt(np.mean(d ** 2))),
        "point_l2_mean": float(l2.mean()),
        "point_l2_median": float(np.median(l2)),
    }


def figure(path, image, clouds, title, limit, rng):
    fig = plt.figure(figsize=(19, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(np.clip(np.asarray(image).transpose(1, 2, 0), 0, 1))
    ax.axis("off")
    ax.set_title("Input RGB")
    for col, (name, points, colors) in enumerate(clouds, 1):
        if len(points) > limit:
            ids = rng.choice(len(points), limit, replace=False)
            points, colors = points[ids], colors[ids]
        ax = fig.add_subplot(grid[0, col], projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, s=0.25, linewidths=0)
        ax.set_axis_off()
        ax.set_title(name)
        ax = fig.add_subplot(grid[1, col], projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, s=0.25, linewidths=0)
        ax.view_init(elev=85, azim=-90)
        ax.set_axis_off()
        ax.set_title(name + " top")
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.inference_mode()
def run_one(name, checkpoint, indices, output, device, vggt, transform, cfg):
    start = time.time()
    model = baseframework.from_pretrained(str(checkpoint))
    model.to(device).eval()
    dataset = get_vla_dataset(data_cfg=model.config.datasets.vla_data, mode="train")
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    parts, offsets = dataset_index(dataset)
    samples, names = [], []
    for index in indices:
        sample, dataset_name = sample_at(dataset, parts, offsets, index)
        samples.append(sample)
        names.append(dataset_name)
    images, hidden, projected, target, view_mode = extract(model, samples, device)
    view_count = 1 if view_mode == "base_only" else 2
    rows = [torch.stack([transform(x.convert("RGB")) for x in list(im)[:view_count]]) for im in images]
    image_tensor = torch.stack(rows).to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        layers, patch_start = vggt.aggregator(image_tensor)
    patch_count = layers[-1].shape[2] - int(patch_start)
    side = int(math.sqrt(patch_count))
    predicted_dense = lift(projected, side).to(device)
    target_dense = lift(target, side).to(device)
    dpt_layers = tuple(int(index) for index in vggt.point_head.intermediate_layer_idx)
    teacher_layer = int(model.config.framework.spatial.get("teacher_layer", -1)) % len(layers)
    mode_layers = {
        "target_all_layers": replace(layers, target_dense, int(patch_start), dpt_layers),
        "latent_all_layers": replace(layers, predicted_dense, int(patch_start), dpt_layers),
        "target_last_layer": replace(layers, target_dense, int(patch_start), (teacher_layer,)),
        "latent_last_layer": replace(layers, predicted_dense, int(patch_start), (teacher_layer,)),
    }
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        teacher_points, teacher_conf = vggt.point_head(layers, image_tensor, int(patch_start))
        mode_predictions = {
            mode: vggt.point_head(replaced, image_tensor, int(patch_start))
            for mode, replaced in mode_layers.items()
        }
    root = Path(output) / name
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "view_mode": view_mode,
        "source_patch_count": patch_count,
        "source_patch_side": side,
        "method": "4x4 spatial features bilinear-upsampled to the 37x37 VGGT grid",
        "dpt_layer_indices": list(dpt_layers),
        "distilled_teacher_layer": teacher_layer,
        "modes": {
            "teacher_native": "unmodified VGGT aggregator layers",
            "target_all_layers": "pooled teacher target repeated into every DPT layer (bottleneck oracle)",
            "latent_all_layers": "predicted latent repeated into every DPT layer (legacy-comparable probe)",
            "target_last_layer": "pooled teacher target substituted only at the distilled layer",
            "latent_last_layer": "predicted latent substituted only at the distilled layer",
        },
        "samples": [],
        "elapsed_seconds": time.time() - start,
    }
    for row, (index, dataset_name, sample) in enumerate(zip(indices, names, samples)):
        sample_root = root / f"sample_{index:06d}"
        sample_root.mkdir(exist_ok=True)
        if cfg.save_latents:
            torch.save({
                "pre_projector": hidden[row],
                "projected_4x4": projected[row],
                "vggt_target_4x4": target[row],
                "upsampled_prediction": predicted_dense[row].cpu(),
                "upsampled_target": target_dense[row].cpu(),
            }, sample_root / "spatial_latent.pt")
        cosine = F.cosine_similarity(projected[row], target[row], dim=-1)
        rec = {
            "global_index": index,
            "dataset_name": dataset_name,
            "feature_cosine_mean": float(cosine.mean()),
            "feature_cosine_per_view": [float(x) for x in cosine.mean(dim=1)],
            "feature_mse": float(F.mse_loss(projected[row], target[row])),
            "views": [],
        }
        for view in range(view_count):
            tp = teacher_points[row, view].float().cpu().numpy()
            tc = teacher_conf[row, view].float().cpu().numpy()
            image = image_tensor[row, view].float().cpu().numpy()
            teacher_cloud, teacher_colors = cloud(tp, tc, image, cfg.max_ply_points, rng)
            ply(sample_root / f"teacher_native_view{view}.ply", teacher_cloud, teacher_colors)
            mode_clouds = {}
            mode_metrics = {}
            for mode, (points_tensor, conf_tensor) in mode_predictions.items():
                points = points_tensor[row, view].float().cpu().numpy()
                confidence = conf_tensor[row, view].float().cpu().numpy()
                points_cloud, points_colors = cloud(points, confidence, image, cfg.max_ply_points, rng)
                mode_clouds[mode] = (points_cloud, points_colors)
                ply(sample_root / f"{mode}_view{view}.ply", points_cloud, points_colors)
                mode_metrics[mode + "_vs_native"] = pmetrics(tp, points)
            target_all_points = mode_predictions["target_all_layers"][0][row, view].float().cpu().numpy()
            target_last_points = mode_predictions["target_last_layer"][0][row, view].float().cpu().numpy()
            latent_all_points = mode_predictions["latent_all_layers"][0][row, view].float().cpu().numpy()
            latent_last_points = mode_predictions["latent_last_layer"][0][row, view].float().cpu().numpy()
            mode_metrics["latent_all_layers_vs_target"] = pmetrics(target_all_points, latent_all_points)
            mode_metrics["latent_last_layer_vs_target"] = pmetrics(target_last_points, latent_last_points)
            rec["views"].append(mode_metrics)
            figure(
                sample_root / f"comparison_view{view}.png",
                image,
                [
                    ("VGGT native", teacher_cloud, teacher_colors),
                    ("16-token target oracle", *mode_clouds["target_all_layers"]),
                    ("predicted latent / all layers", *mode_clouds["latent_all_layers"]),
                ],
                f"{name} sample={index} view={view}",
                cfg.max_plot_points,
                rng,
            )
        summary["samples"].append(rec)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    cfg = args()
    experiments = []
    for value in cfg.experiment:
        name, path = value.split("=", 1)
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        experiments.append((name, path))
    device = torch.device(cfg.device)
    indices = [int(x) for x in cfg.sample_indices.split(",") if x]
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    vggt = VGGT(enable_camera=False, enable_point=True, enable_depth=True, enable_track=False)
    state = torch.load(cfg.vggt_weights, map_location="cpu", weights_only=False)
    incompat = vggt.load_state_dict(state, strict=False)
    vggt.to(device).eval()
    transform = build_vggt_transform(518, "resize")
    summaries = []
    for name, checkpoint in experiments:
        print(f"[start] {name}", flush=True)
        summaries.append(run_one(name, checkpoint, indices, output, device, vggt, transform, cfg))
        print(f"[done] {name}", flush=True)
    manifest = {
        "experiments": [name for name, _ in experiments],
        "sample_indices": indices,
        "vggt_weights": str(Path(cfg.vggt_weights).resolve()),
        "vggt_missing_keys": len(incompat.missing_keys),
        "vggt_unexpected_keys": len(incompat.unexpected_keys),
        "summaries": summaries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
