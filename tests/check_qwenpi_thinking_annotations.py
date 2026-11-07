"""Read-only coverage check for the LaRA LIBERO CoT/BBox files."""
import argparse
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.bridge_annotations import BridgeAnnotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    for root in sorted(args.data_root.glob("libero_*_lerobot")):
        ann = root / "annotations"
        captions = ann / "episode_dense_captions_full.jsonl"
        bboxes = ann / "episode_sam3_bboxes_from_dino_final.jsonl"
        if not captions.is_file() or not bboxes.is_file():
            raise FileNotFoundError(f"Missing thinking annotations in {ann}")
        data = BridgeAnnotations(root, cot_path=captions, bbox_path=bboxes)
        annotated, complete = 0, 0
        for episode, steps in data._cot.items():
            for step, cot in steps.items():
                annotated += 1
                bbox = data.get_step_bbox(episode, step)
                complete += int(bool(cot.subtask and cot.reasoning and bbox and bbox.valid))
        if not complete:
            raise ValueError(f"No complete thinking targets for {root}")
        print(f"{root.name}: complete={complete}/{annotated} ({complete / max(annotated, 1):.1%})")


if __name__ == "__main__":
    main()
