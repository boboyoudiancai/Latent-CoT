#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from laravla.dataloader.lerobot_datasets import get_vla_dataset
from laravla.model.modules.vlm.vggt_teacher import VGGTTeacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    spatial = cfg.framework.spatial
    cache_dir = args.cache_dir or str(spatial.target_cache_dir)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train")
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    teacher = VGGTTeacher(
        teacher_repo=str(spatial.teacher_repo),
        input_size=int(spatial.input_size),
        layer=int(spatial.teacher_layer),
        patch_start_idx=int(spatial.patch_start_idx),
        use_vggt_pe=bool(spatial.use_vggt_pe),
        preprocess_mode=str(spatial.preprocess_mode),
        cache_size=1,
        cache_dir=cache_dir,
        cache_only=False,
    )
    end = len(dataset) if args.end < 0 else min(args.end, len(dataset))
    device = torch.device(args.device)
    for start in tqdm(range(args.start, end, args.batch_size), total=(end - args.start + args.batch_size - 1) // args.batch_size):
        samples = [dataset[index] for index in range(start, min(start + args.batch_size, end))]
        images = [sample["image"] for sample in samples]
        keys = [sample["spatial_cache_key"] for sample in samples]
        teacher.encode(images, device=device, target_tokens=int(spatial.token_count), cache_keys=keys)
    print(f"precompute_complete samples={end - args.start} cache_dir={cache_dir}")


if __name__ == "__main__":
    main()
