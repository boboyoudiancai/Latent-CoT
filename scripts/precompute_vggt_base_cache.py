#!/usr/bin/env python
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from laravla.dataloader.lerobot_datasets import get_vla_dataset
from laravla.model.modules.vlm.vggt_teacher import VGGTTeacher, pool_vggt_tokens


PACKED_FORMAT = "vggt-packed-bfloat16-v1"


def build_underlying_index(dataset):
    datasets = list(dataset.datasets)
    offsets = [0]
    for single_dataset in datasets:
        single_dataset.transforms.eval()
        offsets.append(offsets[-1] + len(single_dataset))
    return datasets, offsets


def get_underlying_sample(datasets, offsets, global_index):
    dataset_index = bisect_right(offsets, global_index) - 1
    single_dataset = datasets[dataset_index]
    local_index = global_index - offsets[dataset_index]
    trajectory_id, base_index = single_dataset.all_steps[local_index]
    data = single_dataset.get_step_data(trajectory_id, base_index)
    transformed = single_dataset.transforms(data)
    return single_dataset._build_sample_from_data(transformed, trajectory_id, base_index)


class PackedShardWriter:
    def __init__(self, cache_dir: str | Path, start: int, end: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.start = int(start)
        self.end = int(end)
        self.count = 0
        self.sample_shape = None
        stem = f"packed_{self.start:09d}_{self.end:09d}"
        self.data_path = self.cache_dir / f"{stem}.bf16"
        self.keys_path = self.cache_dir / f"{stem}.keys"
        self.manifest_path = self.cache_dir / f"{stem}.manifest.json"
        self.partial_data_path = self.cache_dir / f"{stem}.bf16.partial"
        self.partial_keys_path = self.cache_dir / f"{stem}.keys.partial"
        if self.manifest_path.exists():
            raise FileExistsError(f"completed cache shard already exists: {self.manifest_path}")
        stale = [path for path in (self.partial_data_path, self.partial_keys_path) if path.exists()]
        if stale:
            raise FileExistsError(f"partial cache shard already exists: {stale}")

    def append(self, keys, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if values.ndim != 4 or tuple(values.shape[1:3])[:1] != (1,):
            raise ValueError(f"expected base-only targets [B,1,N,C], got {tuple(values.shape)}")
        if len(keys) != int(values.shape[0]):
            raise ValueError("packed cache key/value batch size mismatch")
        shape = tuple(int(value) for value in values.shape[1:])
        if self.sample_shape is None:
            self.sample_shape = shape
        elif shape != self.sample_shape:
            raise ValueError(f"packed cache shape changed: {shape} vs {self.sample_shape}")
        if self.count + len(keys) > self.end - self.start:
            raise ValueError("packed cache shard received too many samples")

        raw = values.view(torch.uint16).numpy().tobytes(order="C")
        with self.partial_data_path.open("ab") as handle:
            handle.write(raw)
        with self.partial_keys_path.open("a", encoding="ascii") as handle:
            for key in keys:
                if not isinstance(key, bytes) or len(key) != 32:
                    raise ValueError("VGGT packed cache keys must be 32-byte SHA-256 digests")
                handle.write(key.hex() + "\n")
        self.count += len(keys)

    def finalize(self) -> None:
        expected = self.end - self.start
        if self.count != expected:
            raise ValueError(f"incomplete cache shard: expected={expected} actual={self.count}")
        if self.sample_shape is None:
            raise ValueError("cannot finalize an empty cache shard")
        expected_bytes = self.count * math.prod(self.sample_shape) * 2
        actual_bytes = self.partial_data_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"packed data size mismatch: expected={expected_bytes} actual={actual_bytes}")
        key_count = sum(1 for line in self.partial_keys_path.read_text(encoding="ascii").splitlines() if line)
        if key_count != self.count:
            raise ValueError(f"packed key count mismatch: expected={self.count} actual={key_count}")

        os.replace(self.partial_data_path, self.data_path)
        os.replace(self.partial_keys_path, self.keys_path)
        manifest = {
            "count": self.count,
            "data_file": self.data_path.name,
            "dtype": "bfloat16",
            "end": self.end,
            "format": PACKED_FORMAT,
            "keys_file": self.keys_path.name,
            "sample_shape": list(self.sample_shape),
            "start": self.start,
            "view_mode": "base_only",
            "source_view_indices": [0],
        }
        tmp_manifest = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp_manifest.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_manifest, self.manifest_path)


@torch.inference_mode()
def encode_base_only(teacher: VGGTTeacher, samples, device: torch.device, target_tokens: int):
    rows = []
    for sample in samples:
        views = sample.get("image")
        if not isinstance(views, (list, tuple)) or not views:
            raise ValueError("base-only VGGT cache requires sample['image'][0]")
        base_image = teacher._as_rgb(views[0])
        rows.append(teacher.transform(base_image))
    batch = torch.stack(rows, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
    teacher.encoder.to(device=device)
    teacher.encoder.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        features, patch_start_idx = teacher._unwrap_features({}, batch)
    if features.ndim == 3:
        bsz, views = batch.shape[:2]
        if features.shape[0] == bsz * views:
            features = features.reshape(bsz, views, features.shape[1], features.shape[2])
        elif features.shape[0] == bsz:
            features = features.unsqueeze(1)
        else:
            raise ValueError(f"unsupported VGGT feature shape {tuple(features.shape)}")
    if features.ndim != 4 or features.shape[1] != 1:
        raise ValueError(f"expected one-view VGGT features, got {tuple(features.shape)}")
    features = features[..., int(patch_start_idx):, :]
    return pool_vggt_tokens(features, int(target_tokens), teacher.use_vggt_pe).to(torch.bfloat16).cpu()


def validate_cache(cache_dir: str | Path, expected_count: int, expected_shards: int) -> dict:
    cache_dir = Path(cache_dir)
    manifests = []
    all_keys = set()
    total = 0
    expected_start = 0
    for path in sorted(cache_dir.glob("packed_*.manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != PACKED_FORMAT:
            raise ValueError(f"bad cache format: {path}")
        if manifest.get("sample_shape") != [1, 16, 2048]:
            raise ValueError(f"bad base-only sample shape in {path}: {manifest.get('sample_shape')}")
        start = int(manifest["start"])
        end = int(manifest["end"])
        count = int(manifest["count"])
        if start != expected_start or end - start != count:
            raise ValueError(f"non-contiguous cache range in {path}: {start}:{end} count={count}")
        data_path = cache_dir / manifest["data_file"]
        keys_path = cache_dir / manifest["keys_file"]
        expected_bytes = count * math.prod(manifest["sample_shape"]) * 2
        if data_path.stat().st_size != expected_bytes:
            raise ValueError(f"bad packed data size: {data_path}")
        local_count = 0
        for line in keys_path.read_text(encoding="ascii").splitlines():
            key = bytes.fromhex(line)
            if key in all_keys:
                raise ValueError(f"duplicate cache key: {line}")
            all_keys.add(key)
            local_count += 1
        if local_count != count:
            raise ValueError(f"bad key count in {keys_path}: {local_count} vs {count}")
        expected_start = end
        total += count
        manifests.append(path.name)
    if len(manifests) != int(expected_shards):
        raise ValueError(f"expected {expected_shards} shards, found {len(manifests)}")
    if total != int(expected_count):
        raise ValueError(f"expected {expected_count} samples, found {total}")
    result = {
        "format": PACKED_FORMAT,
        "view_mode": "base_only",
        "sample_shape": [1, 16, 2048],
        "shards": len(manifests),
        "samples": total,
        "unique_keys": len(all_keys),
        "validation": "passed",
    }
    (cache_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def self_test_writer() -> None:
    with tempfile.TemporaryDirectory(prefix="vggt_base_writer_") as directory:
        values = torch.arange(3 * 1 * 4 * 8, dtype=torch.float32).reshape(3, 1, 4, 8)
        keys = [bytes([index + 1]) * 32 for index in range(3)]
        writer = PackedShardWriter(directory, 0, 3)
        writer.append(keys[:2], values[:2])
        writer.append(keys[2:], values[2:])
        writer.finalize()
        manifest = json.loads(next(Path(directory).glob("*.manifest.json")).read_text())
        assert manifest["sample_shape"] == [1, 4, 8]
        assert manifest["view_mode"] == "base_only"
    print("BASE_CACHE_WRITER_TEST_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker_id", type=int, default=-1)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--expected_count", type=int, default=273465)
    parser.add_argument("--expected_shards", type=int, default=8)
    parser.add_argument("--self_test_writer", action="store_true")
    args = parser.parse_args()

    if args.self_test_writer:
        self_test_writer()
        return
    if args.validate_only:
        print(json.dumps(validate_cache(args.cache_dir, args.expected_count, args.expected_shards), sort_keys=True))
        return
    if not args.config_yaml or not args.cache_dir:
        parser.error("--config_yaml and --cache_dir are required for generation")

    cfg = OmegaConf.load(args.config_yaml)
    spatial = cfg.framework.spatial
    mixture = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train")
    datasets, offsets = build_underlying_index(mixture)
    total_samples = offsets[-1]
    end = total_samples if args.end < 0 else min(args.end, total_samples)
    if args.start < 0 or args.start >= end:
        raise ValueError(f"invalid range start={args.start} end={end}")
    writer = PackedShardWriter(args.cache_dir, args.start, end)
    device = torch.device(args.device)
    teacher = VGGTTeacher(
        teacher_repo=str(spatial.teacher_repo),
        input_size=int(spatial.input_size),
        layer=int(spatial.teacher_layer),
        patch_start_idx=int(spatial.patch_start_idx),
        use_vggt_pe=bool(spatial.use_vggt_pe),
        preprocess_mode=str(spatial.preprocess_mode),
        cache_size=0,
        cache_dir="",
        cache_only=False,
    )
    target_tokens = int(spatial.token_count)
    print(
        f"worker={args.worker_id} device={device} gpu={torch.cuda.get_device_name(device)} "
        f"range={args.start}:{end} total_samples={total_samples} batch_size={args.batch_size} "
        f"teacher_repo={spatial.teacher_repo} view_mode=base_only",
        flush=True,
    )
    total_batches = math.ceil((end - args.start) / args.batch_size)
    for batch_index, start in enumerate(
        tqdm(range(args.start, end, args.batch_size), total=total_batches),
        start=1,
    ):
        samples = [
            get_underlying_sample(datasets, offsets, index)
            for index in range(start, min(start + args.batch_size, end))
        ]
        values = encode_base_only(teacher, samples, device, target_tokens)
        if batch_index == 1:
            print(f"worker={args.worker_id} first_target_shape={tuple(values.shape)} dtype={values.dtype}", flush=True)
        keys = [
            teacher.cache_digest(
                [sample["image"][0]],
                target_tokens,
                cache_key=sample["spatial_cache_key"],
            )
            for sample in samples
        ]
        writer.append(keys, values)
    writer.finalize()
    print(
        f"precompute_complete worker={args.worker_id} samples={end - args.start} "
        f"shape={writer.sample_shape} cache_dir={args.cache_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
