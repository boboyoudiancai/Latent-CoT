import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


def build_dataset():
    dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {
        "state_action_dtype": "float16",
        "image_size": [2, 2],
        "include_state": False,
    }
    dataset.include_future = True
    dataset._modality_keys = {
        "video": ["video.primary_image", "video.wrist_image"],
        "language": ["language.task"],
        "action": ["action.value"],
    }
    dataset.tag = "franka"
    return dataset


def test_future_fallback_uses_first_view_only():
    dataset = build_dataset()
    current_primary = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    future_primary = np.ones((1, 2, 2, 3), dtype=np.uint8)
    stationary_wrist = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    data = {
        "video.primary_image": current_primary,
        "video_next.primary_image": future_primary,
        "video.wrist_image": stationary_wrist,
        "video_next.wrist_image": stationary_wrist.copy(),
        "language.task": np.asarray(["pick up the mug"]),
        "action.value": np.zeros((8, 7), dtype=np.float32),
    }
    sample = dataset._pack_sample(data)
    assert sample["image_next_fallback"] is False

    data["video_next.primary_image"] = current_primary.copy()
    sample = dataset._pack_sample(data)
    assert sample["image_next_fallback"] is True
