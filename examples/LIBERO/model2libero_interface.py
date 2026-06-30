from collections import deque
from typing import Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.SimplerEnv.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path

from laravla.model.tools import read_mode_config


class M1Inference:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        # Latent reasoning (implicit) parameters (Step 3: prompt formatting only)
        enable_latent_reasoning: bool = False,
        cot_mode: str = "implicit",
        thinking_token_count: int = -1,
        img_next_count: int = -1,
        action_decode_mode: str = "diffusion",
        fast_max_new_tokens: int = 256,
        # Testing utility: allow init without connecting websocket server
        connect_server: bool = True,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port) if connect_server else None
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.action_decode_mode = str(action_decode_mode or "diffusion").lower()
        if self.action_decode_mode not in {"diffusion", "fast"}:
            raise ValueError(f"Unsupported action_decode_mode={action_decode_mode!r}")
        self.fast_max_new_tokens = int(fast_max_new_tokens)

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_decode_mode: {self.action_decode_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Read config once (avoid repeated disk I/O).
        model_config, norm_stats = read_mode_config(Path(policy_ckpt_path))
        self.model_config = model_config
        self.norm_stats = norm_stats

        # Action unnormalization stats
        self.unnorm_key = M1Inference._check_unnorm_key(norm_stats, self.unnorm_key)
        self.action_norm_stats = norm_stats[self.unnorm_key]["action"]

        # Action chunk size (future_action_window_size + 1)
        action_model_cfg = model_config["framework"]["action_model"]
        if "action_horizon" in action_model_cfg:
            self.action_chunk_size = int(action_model_cfg["action_horizon"])
        else:
            self.action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        if self.action_decode_mode == "fast":
            self.action_chunk_size = 1

        # ---- latent reasoning prompt formatting config (strict alignment) ----
        self.enable_latent_reasoning = bool(enable_latent_reasoning)
        self.cot_mode = str(cot_mode)
        self.thinking_token_count = int(thinking_token_count) if int(thinking_token_count) > 0 else 3
        self.img_next_count = int(img_next_count) if int(img_next_count) > 0 else 16

        latent_cfg = ((model_config.get("framework", {}) or {}).get("latent_reasoning", {}) or {})
        self._thinking_token = str(latent_cfg.get("thinking_token", "<|thinking|>"))
        self._start_of_thinking_token = str(latent_cfg.get("start_of_thinking_token", "<|start_of_thinking|>"))
        self._end_of_thinking_token = str(latent_cfg.get("end_of_thinking_token", "<|end_of_thinking|>"))

        img_next_cfg = (model_config.get("framework", {}) or {}).get("img_next", {}) or {}
        self._include_img_next = bool(img_next_cfg.get("enable", False))
        self._img_next_token = str(img_next_cfg.get("token", "<img_next>"))
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def format_instruction(self, instruction: str) -> str:
        return self._format_instruction_with_latent(instruction)

    def _thinking_span(self) -> str:
        """
        Implicit evaluation uses fixed "stage4" behavior:
          - Always insert exactly 3 <|thinking|> tokens by default (SUBTASK+BBOX+REASON),
            unless overridden via `thinking_token_count`.
          - No extra tags/order/stage logic to keep behavior minimal and aligned.
        """
        body = self._thinking_token * int(self.thinking_token_count)
        return f"{self._start_of_thinking_token}{body}{self._end_of_thinking_token}"

    def _img_next_span(self) -> str:
        if (not self._include_img_next) or int(self.img_next_count) <= 0:
            return ""
        return self._img_next_token * int(self.img_next_count)

    def _format_instruction_with_latent(self, instruction: str) -> str:
        """
        Strictly aligned with training formatter and SimplerEnv:
          - delimiter: ". @ " (dot + space + @ + space)
          - thinking/img_next spans are pure token concatenation (no spaces inside)
          - ensure <img_next> is the last segment (no trailing text after it)
        """
        prompt = (instruction or "").strip()
        if (not self.enable_latent_reasoning) or (self.cot_mode != "implicit"):
            return prompt

        span = self._thinking_span()
        text = f"{prompt}. @ {span}"

        img_next_span = self._img_next_span()
        if img_next_span:
            text = f"{text} {img_next_span}" if text else img_next_span
        return text.strip()


    def step(
        self, 
        images, 
        task_description: Optional[str] = None,
        step: int = 0,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        # image: Image.Image = Image.fromarray(image)

        images = [self._resize_image(image) for image in images]
        instruction = self._format_instruction_with_latent(self.task_description or "")
        vla_input = {
            "batch_images": [images],
            "instructions": [instruction],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        # Step 4: Explicitly trigger implicit latent reasoning path on server
        if self.enable_latent_reasoning and self.cot_mode == "implicit":
            vla_input["use_iterative_forward"] = True
            vla_input["cot_mode"] = "implicit"
            vla_input["emit_thinking_tokens"] = False
        elif self.cot_mode == "explicit":
            vla_input["cot_mode"] = "explicit"
        if self.action_decode_mode == "fast":
            vla_input["use_fast_action_tokens"] = True
            vla_input["fast_max_new_tokens"] = self.fast_max_new_tokens

        if self.client is None:
            raise RuntimeError("Websocket client is not initialized (connect_server=False); cannot call step().")



        
        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            response = self.client.infer(vla_input)
            if response.get("status") != "ok":
                raise RuntimeError(f"Policy server returned error: {response}")
            # unnormalize the action
            # import ipdb; ipdb.set_trace()
            normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
            normalized_actions = normalized_actions[0]    
            self.raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
        raw_actions = self.raw_actions[step % action_chunk_size][None]    

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        _, norm_stats = read_mode_config(Path(policy_ckpt_path))  # read config and norm_stats

        unnorm_key = M1Inference._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(Path(policy_ckpt_path))  # read config and norm_stats
        return int(model_config["framework"]["action_model"]["future_action_window_size"]) + 1


    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
