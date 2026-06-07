import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class StepCoT:
    subtask: str
    reasoning: str
    gripper_state: Optional[int] = None


@dataclass
class StepBBox:
    bbox: Optional[np.ndarray]
    confidence: Optional[float]

    @property
    def valid(self) -> bool:
        return self.bbox is not None


class BridgeAnnotations:
    """
    Lightweight loader for BRIDGE-LeRobot auxiliary annotations:
    - Dense CoT captions (subtask / reasoning / gripper_state)
    - SAM3 per-frame bboxes

    This class is intentionally kept independent from LeRobot datasets so it can
    be attached only for BRIDGE-style datasets where the annotation files exist.
    """

    def __init__(
        self,
        dataset_root: Path,
        cot_path: Optional[Path] = None,
        bbox_path: Optional[Path] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)

        annotations_dir = self.dataset_root / "annotations"
        if cot_path is None:
            cot_path = annotations_dir / "episode_dense_captions_full_final.jsonl"
        if bbox_path is None:
            bbox_path = annotations_dir / "episode_sam3_bboxes_final.jsonl"

        self.cot_path = Path(cot_path)
        self.bbox_path = Path(bbox_path)

        self._cot: Dict[int, Dict[int, StepCoT]] = {}
        self._bbox: Dict[int, Dict[int, StepBBox]] = {}
        self._bbox2: Dict[int, Dict[int, StepBBox]] = {}
        self._episode_num_steps: Dict[int, int] = {}

        if self.cot_path.exists():
            self._load_cot()
        if self.bbox_path.exists():
            self._load_bbox()

    # --------------------------------------------------------------------- #
    # Loading utilities
    # --------------------------------------------------------------------- #
    def _load_cot(self) -> None:
        with self.cot_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = int(obj.get("episode_index"))
                steps = obj.get("steps", {})
                per_step: Dict[int, StepCoT] = {}
                for k, v in steps.items():
                    try:
                        step_idx = int(k)
                    except Exception:
                        continue
                    subtask = v.get("subtask", "") or ""
                    reasoning = v.get("reasoning", "") or ""
                    gripper_state = v.get("gripper_state", None)
                    if subtask or reasoning:
                        per_step[step_idx] = StepCoT(
                            subtask=subtask,
                            reasoning=reasoning,
                            gripper_state=gripper_state,
                        )
                if per_step:
                    self._cot[ep] = per_step

    def _load_bbox(self) -> None:
        with self.bbox_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ep = int(obj.get("episode_index"))
                num_steps = int(obj.get("num_steps", 0))
                if num_steps > 0:
                    self._episode_num_steps[ep] = num_steps

                per_step: Dict[int, StepBBox] = {}
                per_step2: Dict[int, StepBBox] = {}

                # primary source: dense_labels.active_bbox (aligned with num_steps)
                dense_labels = obj.get("dense_labels") or {}
                active_bboxes = dense_labels.get("active_bbox")
                active_bboxes2 = dense_labels.get("active_bbox2")
                dense_scores = obj.get("dense_scores")
                if isinstance(active_bboxes, list):
                    for idx, bbox_val in enumerate(active_bboxes):
                        if bbox_val is None:
                            continue
                        try:
                            arr = np.asarray(bbox_val, dtype=np.float32)
                        except Exception:
                            arr = None
                        if arr is None:
                            continue
                        conf = None
                        if isinstance(dense_scores, list) and idx < len(dense_scores):
                            conf_val = dense_scores[idx]
                            if conf_val is not None:
                                conf = float(conf_val)
                        per_step[idx] = StepBBox(bbox=arr, confidence=conf)

                # optional secondary bbox: dense_labels.active_bbox2 (if present)
                # NOTE: no dedicated confidence field is assumed; confidence stays None.
                if isinstance(active_bboxes2, list):
                    for idx, bbox_val in enumerate(active_bboxes2):
                        if bbox_val is None:
                            continue
                        try:
                            arr = np.asarray(bbox_val, dtype=np.float32)
                        except Exception:
                            arr = None
                        if arr is None:
                            continue
                        per_step2[idx] = StepBBox(bbox=arr, confidence=None)

                if per_step:
                    self._bbox[ep] = per_step
                if per_step2:
                    self._bbox2[ep] = per_step2

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def has_cot(self, episode_index: int) -> bool:
        """Return True if this episode has at least one CoT step."""
        return episode_index in self._cot and bool(self._cot[episode_index])

    def has_bbox(self, episode_index: int) -> bool:
        """Return True if this episode has at least one valid bbox step."""
        per_step = self._bbox.get(episode_index)
        if not per_step:
            return False
        return any(v.valid for v in per_step.values())

    def get_step_cot(self, episode_index: int, step_index: int) -> Optional[StepCoT]:
        """Return CoT annotation for (episode, step) or None."""
        return self._cot.get(episode_index, {}).get(step_index)

    def get_step_bbox(self, episode_index: int, step_index: int) -> Optional[StepBBox]:
        """Return bbox annotation for (episode, step) or None."""
        return self._bbox.get(episode_index, {}).get(step_index)

    def get_step_bbox2(self, episode_index: int, step_index: int) -> Optional[StepBBox]:
        """Return optional secondary bbox annotation for (episode, step) or None."""
        return self._bbox2.get(episode_index, {}).get(step_index)

    def episode_bbox_coverage(self, episode_index: int) -> Optional[float]:
        """
        Return fraction of steps in this episode that have a valid bbox.
        Useful for episode-level filtering.
        """
        num_steps = self._episode_num_steps.get(episode_index)
        per_step = self._bbox.get(episode_index)
        if not num_steps or not per_step:
            return None
        valid = sum(1 for v in per_step.values() if v.valid)
        return float(valid) / float(num_steps)
