# from collections import deque
# from typing import Optional, Sequence
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# from transforms3d.euler import euler2axangle
# from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

# from examples.SimplerEnv.adaptive_ensemble import AdaptiveEnsembler
# from typing import Dict
# import numpy as np
# from pathlib import Path


# # 独立的配置读取函数，避免导入训练模块
# def read_config_simple(checkpoint_path):
#     """
#     简化版的配置读取，只读取 norm_stats，不依赖训练模块
    
#     Args:
#         checkpoint_path: 模型 checkpoint 路径 (.pt 文件)
    
#     Returns:
#         tuple: (config_dict, norm_stats_dict)
#     """
#     import json
#     from pathlib import Path
    
#     checkpoint_pt = Path(checkpoint_path)
#     if not checkpoint_pt.exists():
#         raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
#     # 获取 run 目录（checkpoint 的父目录的父目录）
#     run_dir = checkpoint_pt.parents[1]
    
#     # 读取配置文件
#     config_yaml = run_dir / "config.yaml"
#     dataset_statistics_json = run_dir / "dataset_statistics.json"
    
#     # 只读取 norm_stats（不需要加载完整配置）
#     if not dataset_statistics_json.exists():
#         raise FileNotFoundError(f"Missing dataset_statistics.json in {run_dir}")
    
#     with open(dataset_statistics_json, "r") as f:
#         norm_stats = json.load(f)
    
#     # config 可以返回 None（SimplerEnv 不需要）
#     config = None
#     if config_yaml.exists():
#         try:
#             # 尝试用 yaml 读取（避免 omegaconf 依赖）
#             import yaml
#             with open(config_yaml, "r") as f:
#                 config = yaml.safe_load(f)
#         except:
#             # 如果没有 yaml 库或读取失败，返回 None
#             pass
    
#     return config, norm_stats


# class M1Inference:
#     def __init__(
#         self,
#         policy_ckpt_path,
#         unnorm_key: Optional[str] = None,
#         policy_setup: str = "widowx_bridge",
#         horizon: int = 0,
#         action_ensemble_horizon: Optional[int] = None,
#         image_size: list[int] = [224, 224],
#         action_scale: float = 1.0,
#         cfg_scale: float = 1.5,
#         use_ddim: bool = True,
#         num_ddim_steps: int = 10,
#         action_ensemble = True,
#         adaptive_ensemble_alpha = 0.1,
#         host="0.0.0.0",
#         port=10093,
#         # ECOT (Implicit Reasoning) parameters
#         enable_latent_reasoning: bool = False,
#         thinking_token_count: int = 4,
#     ) -> None:
        
#         # build client to connect server policy
#         self.client = WebsocketClientPolicy(host, port)

#         os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
#         # 如果没有指定 unnorm_key，尝试从 dataset_statistics.json 自动检测
#         if unnorm_key is None:
#             try:
#                 _, norm_stats = read_config_simple(policy_ckpt_path)
#                 available_keys = list(norm_stats.keys())
                
#                 # 根据 policy_setup 映射到实际的 key
#                 key_mapping = {
#                     "widowx_bridge": ["oxe_bridge", "bridge_data_v2", "bridge"],
#                     "google_robot": ["oxe_rt1", "rt1", "fractal"],
#                 }
                
#                 # 查找匹配的 key
#                 for candidate in key_mapping.get(policy_setup, []):
#                     if candidate in available_keys:
#                         unnorm_key = candidate
#                         print(f"✅ Auto-detected unnorm_key: {unnorm_key} from available keys: {available_keys}")
#                         break
                
#                 # 如果没找到，使用第一个可用的 key
#                 if unnorm_key is None and len(available_keys) > 0:
#                     unnorm_key = available_keys[0]
#                     print(f"⚠️ Using first available unnorm_key: {unnorm_key} from {available_keys}")
#             except Exception as e:
#                 print(f"⚠️ Failed to auto-detect unnorm_key: {e}, falling back to default")
#                 unnorm_key = "oxe_bridge" if policy_setup == "widowx_bridge" else "oxe_rt1"
        
#         if policy_setup == "widowx_bridge":
#             action_ensemble = action_ensemble
#             adaptive_ensemble_alpha = adaptive_ensemble_alpha
#             if action_ensemble_horizon is None:
#                 # Set 7 for widowx_bridge to fix the window size of motion scale between each frame. see appendix in our paper for details
#                 action_ensemble_horizon = 7
#             self.sticky_gripper_num_repeat = 1
#         elif policy_setup == "google_robot":
#             action_ensemble = action_ensemble
#             adaptive_ensemble_alpha = adaptive_ensemble_alpha
#             if action_ensemble_horizon is None:
#                 # Set 2 for google_robot to fix the window size of motion scale between each frame. see appendix in our paper for details
#                 action_ensemble_horizon = 2
#             self.sticky_gripper_num_repeat = 10
#         else:
#             raise NotImplementedError(
#                 f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
#             )
#         self.policy_setup = policy_setup
#         self.unnorm_key = unnorm_key

#         print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
#         self.use_ddim = use_ddim
#         self.num_ddim_steps = num_ddim_steps


#         self.cfg_scale = cfg_scale # 1.5

#         self.image_size = image_size
#         self.action_scale = action_scale # 1.0
#         self.horizon = horizon #0
#         self.action_ensemble = action_ensemble
#         self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
#         self.action_ensemble_horizon = action_ensemble_horizon
#         self.sticky_action_is_on = False
#         self.gripper_action_repeat = 0
#         self.sticky_gripper_action = 0.0
#         self.previous_gripper_action = None

#         self.task_description = None
#         self.image_history = deque(maxlen=self.horizon)
#         if self.action_ensemble:
#             self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
#         else:
#             self.action_ensembler = None
#         self.num_image_history = 0

#         self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        
#         # ECOT (Implicit Reasoning) initialization
#         self.enable_latent_reasoning = enable_latent_reasoning
#         self.thinking_token_count = thinking_token_count
        
#         if self.enable_latent_reasoning:
#             # Define thinking token strings (must match training config)
#             self.thinking_tokens = {
#                 "start": "<|start_of_thinking|>",
#                 "thinking": "<|thinking|>",
#                 "end": "<|end_of_thinking|>",
#             }
            
#             # Pre-construct thinking sequence for efficiency
#             # Format: "<|start_of_thinking|><|thinking|><|thinking|>...<|end_of_thinking|>"
#             # 构造 thinking sequence（与训练时格式保持一致，tokens之间无空格）
#             # Note: No leading/trailing spaces to match training format (reasoning_text.strip())
#             self.thinking_sequence = (
#                 f" {self.thinking_tokens['start']}" +
#                 self.thinking_tokens['thinking'] * self.thinking_token_count +
#                 f"{self.thinking_tokens['end']}"
#             ) # Ensure no leading/trailing spaces (consistent with training: reasoning_text.strip())
            
#             print(f"[ECOT] Implicit reasoning enabled with {thinking_token_count} thinking tokens")
#             # Token 统计：1 (start) + N (thinking) + 1 (end)
#             print(f"[ECOT] Thinking sequence: {thinking_token_count} x <|thinking|> tokens inserted")
#         else:
#             self.thinking_tokens = None
#             self.thinking_sequence = None
        

#     def _add_image_to_history(self, image: np.ndarray) -> None:
#         self.image_history.append(image)
#         self.num_image_history = min(self.num_image_history + 1, self.horizon)

#     def reset(self, task_description: str) -> None:
#         self.task_description = task_description
#         self.image_history.clear()
#         if self.action_ensemble:
#             self.action_ensembler.reset()
#         self.num_image_history = 0

#         self.sticky_action_is_on = False
#         self.gripper_action_repeat = 0
#         self.sticky_gripper_action = 0.0
#         self.previous_gripper_action = None

#     def step(
#         self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
#     ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
#         """
#         Input:
#             image: np.ndarray of shape (H, W, 3), uint8
#             task_description: Optional[str], task description; if different from previous task description, policy state is reset
#         Output:
#             raw_action: dict; raw policy action output
#             action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
#                 - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
#                 - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
#                 - 'gripper': np.ndarray of shape (1,), gripper action
#                 - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
#         """
#         if task_description is not None:
#             if task_description != self.task_description:
#                 self.reset(task_description)

#         assert image.dtype == np.uint8
#         resized_image = self._resize_image(image)
#         self._add_image_to_history(resized_image)
#         image = resized_image
        
#         # Construct instruction (with thinking tokens if ECOT is enabled)
#         instruction = self.task_description
#         if self.enable_latent_reasoning:
#             # Add @ delimiter + thinking token sequence
#             # Format: "Instruction @ <|start_of_thinking|><|thinking|>...<|end_of_thinking|>"
#             # Note: " @ " already contains spaces, thinking tokens have no spaces between them
#             instruction = instruction.strip() + " @ " + self.thinking_sequence.strip()
        
#         vla_input = {
#             "batch_images": [[image]],
#             "instructions": [instruction],  # Extended instruction with thinking tokens (if enabled)
#             "unnorm_key": self.unnorm_key,
#             "do_sample": False,
#             "cfg_scale": self.cfg_scale,
#             "use_ddim": self.use_ddim,
#             "num_ddim_steps": self.num_ddim_steps,
#             "use_iterative_forward": self.enable_latent_reasoning,  # Key flag for forward_latent
#         }
        
#         response = self.client.infer(vla_input)
        
        
#         # unnormalize the action
#         normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
#         normalized_actions = normalized_actions[0]
        
        
#         raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
#         if self.action_ensemble:
#             raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

#         raw_action = {
#             "world_vector": np.array(raw_actions[0, :3]),
#             "rotation_delta": np.array(raw_actions[0, 3:6]),
#             "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
#         }

#         # process raw_action to obtain the action to be sent to the maniskill2 environment
#         action = {}
#         action["world_vector"] = raw_action["world_vector"] * self.action_scale
#         action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

#         roll, pitch, yaw = action_rotation_delta
#         axes, angles = euler2axangle(roll, pitch, yaw)
#         action_rotation_axangle = axes * angles
#         action["rot_axangle"] = action_rotation_axangle * self.action_scale

#         if self.policy_setup == "google_robot":
#             action["gripper"] = 0
#             current_gripper_action = raw_action["open_gripper"]
#             if self.previous_gripper_action is None:
#                 relative_gripper_action = np.array([0])
#                 self.previous_gripper_action = current_gripper_action
#             else:
#                 relative_gripper_action = self.previous_gripper_action - current_gripper_action
#             # fix a bug in the SIMPLER code here
#             # self.previous_gripper_action = current_gripper_action

#             if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
#                 self.sticky_action_is_on = True
#                 self.sticky_gripper_action = relative_gripper_action
#                 self.previous_gripper_action = current_gripper_action

#             if self.sticky_action_is_on:
#                 self.gripper_action_repeat += 1
#                 relative_gripper_action = self.sticky_gripper_action

#             if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
#                 self.sticky_action_is_on = False
#                 self.gripper_action_repeat = 0
#                 self.sticky_gripper_action = 0.0

#             action["gripper"] = relative_gripper_action

#         elif self.policy_setup == "widowx_bridge":
#             action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
#         action["terminate_episode"] = np.array([0.0])
#         return raw_action, action

#     @staticmethod
#     def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
#         mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
#         action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
#         normalized_actions = np.clip(normalized_actions, -1, 1)
#         normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
#         actions = np.where(
#             mask,
#             0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
#             normalized_actions,
#         )
        
#         return actions

#     @staticmethod
#     def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
#         """
#         Duplicate stats accessor (retained for backward compatibility).
#         """
#         policy_ckpt_path = Path(policy_ckpt_path)
#         model_config, norm_stats = read_config_simple(policy_ckpt_path)  # 使用简化版读取函数

#         # unnorm_key = baseframework._check_unnorm_key(norm_stats, unnorm_key) # 其实也是很环境 specific 的
#         return norm_stats[unnorm_key]["action"]



#     def _resize_image(self, image: np.ndarray) -> np.ndarray:
#         """
#         Resize image to target size using PIL BILINEAR interpolation (consistent with training).
        
#         Args:
#             image: numpy array of shape (H, W, 3), dtype uint8
            
#         Returns:
#             numpy array of shape (image_size[0], image_size[1], 3), dtype uint8
#         """
#         # Convert to PIL Image for consistent resize method with training
#         # Training uses: Image.BILINEAR (see ecot_rlds/transforms.py)
#         pil_image = Image.fromarray(image)
#         # PIL resize expects (width, height), but image_size is [height, width]
#         resized_pil = pil_image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
#         # Convert back to numpy array
#         return np.array(resized_pil)

#     def visualize_epoch(
#         self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
#     ) -> None:
#         images = [self._resize_image(image) for image in images]
#         ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

#         img_strip = np.concatenate(np.array(images[::3]), axis=1)

#         # set up plt figure
#         figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
#         plt.rcParams.update({"font.size": 12})
#         fig, axs = plt.subplot_mosaic(figure_layout)
#         fig.set_size_inches([45, 10])

#         # plot actions
#         pred_actions = np.array(
#             [
#                 np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
#                 for a in predicted_raw_actions
#             ]
#         )
#         for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
#             # actions have batch, horizon, dim, in this example we just take the first action for simplicity
#             axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
#             axs[action_label].set_title(action_label)
#             axs[action_label].set_xlabel("Time in one episode")

#         axs["image"].imshow(img_strip)
#         axs["image"].set_xlabel("Time in one episode (subsampled)")
#         plt.legend()
#         plt.savefig(save_path)
from collections import deque
from typing import Dict, Optional, Sequence
import copy
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from transforms3d.euler import euler2axangle
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.SimplerEnv.adaptive_ensemble import AdaptiveEnsembler


# 独立的配置读取函数，避免导入训练模块
def read_config_simple(checkpoint_path):
    """
    简化版的配置读取，只读取 norm_stats，不依赖训练模块
    
    Args:
        checkpoint_path: 模型 checkpoint 路径 (.pt 文件)
    
    Returns:
        tuple: (config_dict, norm_stats_dict)
    """
    import json
    from pathlib import Path
    
    checkpoint_pt = Path(checkpoint_path)
    if not checkpoint_pt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 获取 run 目录（checkpoint 的父目录的父目录）
    run_dir = checkpoint_pt.parents[1]
    
    # 读取配置文件
    config_yaml = run_dir / "config.yaml"
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    
    # 只读取 norm_stats（不需要加载完整配置）
    if not dataset_statistics_json.exists():
        raise FileNotFoundError(f"Missing dataset_statistics.json in {run_dir}")
    
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)
    
    # config 可以返回 None（SimplerEnv 不需要）
    config = None
    if config_yaml.exists():
        try:
            # 尝试用 yaml 读取（避免 omegaconf 依赖）
            import yaml
            with open(config_yaml, "r") as f:
                config = yaml.safe_load(f)
        except:
            # 如果没有 yaml 库或读取失败，返回 None
            pass
    
    return config, norm_stats


BRIDGE_REASONING_DEFAULTS = {
    "stage": 4,
    "include_bbox": True,
    "include_action_tokens": False,
    "tag2think_count": {"BBOX": 1, "SUBTASK": 1, "REASON": 1, "ACTION": 1},
}

BRIDGE_BASE_PROMPT = (
    "You are doing A action in a robot task. First output the target bbox, then list the subtask, then generate the motion reasoning. last, you are required to output the next frame in latent space. Instruction:"
)
BRIDGE_BASE_PROMPT_2 = (
    "Robot task reasoning: first output the Subtask to preform next, then output the BBox of target object, then generate the Motion Reasoning. Instruction:"
)
def extract_bridge_reasoning_settings(config: Optional[dict]) -> dict:
    settings = copy.deepcopy(BRIDGE_REASONING_DEFAULTS)
    if not config:
        return settings
    try:
        datasets_cfg = config.get("datasets", {})
        vla_cfg = datasets_cfg.get("vla_data", {})
        bridge_cfg = vla_cfg.get("bridge_reasoning", {})
        if bridge_cfg:
            settings["stage"] = int(bridge_cfg.get("stage", settings["stage"]))
            settings["include_bbox"] = bridge_cfg.get("include_bbox", settings["include_bbox"])
            settings["include_action_tokens"] = bridge_cfg.get(
                "include_action_tokens", settings["include_action_tokens"]
            )
            tag2think = bridge_cfg.get("tag2think_count")
            if tag2think:
                merged = settings["tag2think_count"]
                merged.update({k.upper(): int(v) for k, v in tag2think.items()})
    except Exception:
        pass
    return settings


class M1Inference:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble = True,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10093,
        # ECOT (Implicit Reasoning) parameters
        enable_latent_reasoning: bool = False,
        thinking_token_count: int = 4,
        img_next_count: int = 16,
        img_next_token: str = "<img_next>",
        cot_mode: str = "implicit",
        think_max_len: int = 64,
        think_temp: float = 0.1,
        think_topp: float = 0.9,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.policy_config = None
        self.norm_stats = None
        try:
            self.policy_config, self.norm_stats = read_config_simple(policy_ckpt_path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Failed to load checkpoint metadata from {policy_ckpt_path}") from exc

        # 如果没有指定 unnorm_key，尝试从 dataset_statistics.json 自动检测
        if unnorm_key is None:
            try:
                available_keys = list(self.norm_stats.keys())

                # 根据 policy_setup 映射到实际的 key
                key_mapping = {
                    "widowx_bridge": ["oxe_bridge", "bridge_data_v2", "bridge"],
                    "google_robot": ["oxe_rt1", "rt1", "fractal"],
                }

                # 查找匹配的 key
                for candidate in key_mapping.get(policy_setup, []):
                    if candidate in available_keys:
                        unnorm_key = candidate
                        print(f"✅ Auto-detected unnorm_key: {unnorm_key} from available keys: {available_keys}")
                        break

                # 如果没找到，使用第一个可用的 key
                if unnorm_key is None and len(available_keys) > 0:
                    unnorm_key = available_keys[0]
                    print(f"⚠️ Using first available unnorm_key: {unnorm_key} from {available_keys}")
            except Exception as e:
                print(f"⚠️ Failed to auto-detect unnorm_key: {e}, falling back to default")
                unnorm_key = "oxe_bridge" if policy_setup == "widowx_bridge" else "oxe_rt1"
        
        if policy_setup == "widowx_bridge":
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 7 for widowx_bridge to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 2 for google_robot to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps


        self.cfg_scale = cfg_scale # 1.5

        self.image_size = image_size
        self.action_scale = action_scale # 1.0
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

        self.bridge_reasoning = extract_bridge_reasoning_settings(self.policy_config)
        stage_requires_latent = self.bridge_reasoning["stage"] >= 2
        self.cot_mode = (cot_mode or "implicit").lower()
        self.think_max_len = think_max_len
        self.think_temp = think_temp
        self.think_topp = think_topp

        # 根据 cot_mode 派生开关（显式关闭 latent；隐式开启）
        if self.cot_mode == "implicit":
            self.enable_latent_reasoning = True
            self.emit_thinking_tokens = False
            self.use_iterative_forward = True
        elif self.cot_mode == "explicit":
            self.enable_latent_reasoning = False
            self.emit_thinking_tokens = True
            self.use_iterative_forward = False
        else:  # none / vlm_seen_no_out / fallback
            self.enable_latent_reasoning = False
            self.emit_thinking_tokens = False
            self.use_iterative_forward = False

        # 若 stage 需求与模式冲突，仅告警提示
        if stage_requires_latent and not self.enable_latent_reasoning:
            print(f"[ECOT] Warning: training stage={self.bridge_reasoning['stage']} expects latent reasoning "
                  f"but cot_mode={self.cot_mode} disables it.")


        self.thinking_tokens = {
            "start": "<|start_of_thinking|>",
            "thinking": "<|thinking|>",
            "end": "<|end_of_thinking|>",
        }
        self.img_next_token = img_next_token
        self.img_next_count = max(0, int(img_next_count))
        self.thinking_gen_times: list[float] = []
        self.action_infer_times: list[float] = []

        self.action_norm_stats = self.get_action_stats(self.unnorm_key)

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

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))
        # image: Image.Image = Image.fromarray(image)

        image = self._resize_image(image)
        
        # Construct instruction aligned with mode
        if self.cot_mode in ("implicit", "explicit"):
            instruction = self._format_instruction_with_reasoning(self.task_description or "")
        else:
            instruction = self.task_description or ""
        use_iterative = self.use_iterative_forward if self.cot_mode == "implicit" else False

        vla_input = {
            "batch_images": [[image]],
            "instructions": [instruction],  # Extended instruction with thinking tokens (if enabled)
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "cfg_scale": self.cfg_scale,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "use_iterative_forward": use_iterative,  # Only implicit uses forward_latent
            "cot_mode": self.cot_mode,
            "emit_thinking_tokens": self.emit_thinking_tokens,
            "think_max_len": self.think_max_len,
            "think_temp": self.think_temp,
            "think_topp": self.think_topp,
        }
        
        t0 = time.perf_counter()
        response = self.client.infer(vla_input)
        t1 = time.perf_counter()
        
        
        thinking_time = response.get("data", {}).get("thinking_gen_time", 0)
        self.thinking_gen_times.append(thinking_time)
        self.action_infer_times.append(max(t1 - t0 - thinking_time, 0))
        
        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
        normalized_actions = normalized_actions[0]
        
        
        raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    def get_action_stats(self, unnorm_key: str) -> dict:
        """
        Fetch action normalization stats from cached dataset_statistics.json content.
        """
        if not self.norm_stats:
            raise RuntimeError("Normalization statistics not loaded; cannot unnormalize actions.")
        if unnorm_key not in self.norm_stats:
            raise KeyError(f"Normalization key '{unnorm_key}' not found. Available: {list(self.norm_stats.keys())}")
        return self.norm_stats[unnorm_key]["action"]

    def _format_instruction_with_reasoning(self, instruction: str) -> str:
        instruction = (instruction or "").strip()
        prompt = f"{instruction}".strip()
        if not self.enable_latent_reasoning:
            return prompt

        thinking_body = self._build_thinking_body()
        if not thinking_body:
            return prompt

        span = f"{self.thinking_tokens['start']}{thinking_body}{self.thinking_tokens['end']}"
        img_next_span = ""
        if self.img_next_token and self.img_next_count > 0:
            img_next_span = self.img_next_token * self.img_next_count
        return f"{prompt}. @ {span} {img_next_span}"

    def _build_thinking_body(self) -> str:
        stage = self.bridge_reasoning["stage"]
        tag_counts = self.bridge_reasoning["tag2think_count"]
        include_bbox = self.bridge_reasoning.get("include_bbox", True)
        include_action = self.bridge_reasoning.get("include_action_tokens", False)

        latent_tags = []
        if stage >= 2 and include_bbox:
            latent_tags.append("BBOX")
        if stage >= 3:
            latent_tags.append("SUBTASK")
        if stage >= 4:
            latent_tags.append("REASON")
        if include_action and stage >= 5:
            latent_tags.append("ACTION")

        body_parts = []
        for tag in latent_tags:
            count = max(1, int(tag_counts.get(tag, 1)))
            body_parts.append(self.thinking_tokens["thinking"] * count)
        return "".join(body_parts)



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
