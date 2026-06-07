import argparse
import json
import logging
import math
import os
import pathlib
import sys
import time

starvla_torch_site = os.environ.get("STARVLA_TORCH_SITE")
if starvla_torch_site and starvla_torch_site not in sys.path:
    sys.path.append(starvla_torch_site)

import imageio
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.model2libero_interface import M1Inference

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val):
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    return np.asarray([1.0 - 2.0 * (v > 0.5)], dtype=np.float32)


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _max_steps(task_suite_name):
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite_name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-path", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10093)
    p.add_argument("--task-suite-name", default="libero_goal")
    p.add_argument("--num-trials-per-task", type=int, default=1)
    p.add_argument("--max-tasks", type=int, default=-1)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--video-out-path", default="experiments/libero/logs")
    p.add_argument("--log-path", default="experiments/libero/logs")
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--unnorm-key", default=None)
    p.add_argument("--enable-latent-reasoning", action="store_true")
    p.add_argument("--cot-mode", default="implicit")
    p.add_argument("--thinking-token-count", type=int, default=-1)
    p.add_argument("--img-next-count", type=int, default=-1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s | %(message)s", datefmt="%m/%d [%H:%M:%S]", force=True)
    logging.info("Arguments: %s", json.dumps(vars(args), indent=2, ensure_ascii=False))
    np.random.seed(args.seed)

    pathlib.Path(args.log_path).mkdir(parents=True, exist_ok=True)
    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = task_suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, task_suite.n_tasks)
    max_steps = _max_steps(args.task_suite_name)

    model = M1Inference(
        policy_ckpt_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        host=args.host,
        port=args.port,
        image_size=[224, 224],
        enable_latent_reasoning=args.enable_latent_reasoning,
        cot_mode=args.cot_mode,
        thinking_token_count=args.thinking_token_count,
        img_next_count=args.img_next_count,
    )

    disturb_res = {}
    id2category = {}
    libero_home = os.environ.get("LIBERO_HOME", "")
    mapping_path = os.path.join(libero_home, "libero/libero/benchmark/task_classification.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            task_mapping = json.load(f)[args.task_suite_name]
        for item in task_mapping:
            category = item["category"]
            id2category[item["id"]] = (category, item["name"])
            disturb_res.setdefault(category, {"total_count": 0, "success_count": 0})
            if item["id"] <= num_tasks:
                disturb_res[category]["total_count"] += args.num_trials_per_task

    log_txt = pathlib.Path(args.log_path) / f"{args.task_suite_name}.log"
    total_episodes = 0
    total_successes = 0
    with open(log_txt, "w", encoding="utf-8") as lf:
        lf.write(f"Task suite: {args.task_suite_name}\n")
        lf.write(f"num_tasks: {num_tasks}\n")
        for task_id in tqdm.tqdm(range(num_tasks)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            task_episodes = 0
            task_successes = 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info("Task %d/%d episode %d: %s", task_id + 1, num_tasks, episode_idx, task_description)
                lf.write(f"\nTask: {task_description}\n")
                model.reset(task_description=task_description)
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                t = 0
                step = 0
                done = False
                replay_images = [] if args.save_videos else None
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    if replay_images is not None:
                        replay_images.append(img)
                    state = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                    response = model.step(images=[img, wrist_img], task_description=str(task_description), step=step)
                    raw_action = response["raw_action"]
                    delta_action = np.concatenate([
                        np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1),
                        np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1),
                        _binarize_gripper_open(raw_action["open_gripper"]),
                    ], axis=0)
                    if delta_action.size != 7:
                        raise ValueError(f"invalid action shape {delta_action.shape}")
                    obs, reward, done, info = env.step(delta_action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        if id2category:
                            category = id2category[task_id + 1][0]
                            disturb_res[category]["success_count"] += 1
                        break
                    t += 1
                    step += 1
                task_episodes += 1
                total_episodes += 1
                if args.save_videos and replay_images is not None:
                    suffix = "success" if done else "failure"
                    item = id2category.get(task_id + 1, ("task", task_description.replace(" ", "_")))[1]
                    imageio.mimwrite(pathlib.Path(args.video_out_path) / f"rollout_{item}_episode{episode_idx}_{suffix}.mp4", [np.asarray(x) for x in replay_images], fps=25)
                cur_total = 100.0 * total_successes / total_episodes
                logging.info("Success: %s", done)
                logging.info("# episodes completed so far: %d", total_episodes)
                logging.info("# successes: %d (%.1f%%)", total_successes, cur_total)
                lf.write(f"Success: {done}\n")
                lf.write(f"# episodes completed so far: {total_episodes}\n")
                lf.write(f"# successes: {total_successes} ({cur_total:.1f}%)\n")
                lf.flush()
            logging.info("Current task success rate: %.3f", float(task_successes) / float(task_episodes))
            logging.info("Current total success rate: %.3f", float(total_successes) / float(total_episodes))
            lf.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
            lf.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
            lf.flush()
    if disturb_res:
        with open(pathlib.Path(args.log_path) / f"{args.task_suite_name}.json", "w", encoding="utf-8") as f:
            json.dump(disturb_res, f, ensure_ascii=False)
    logging.info("Total success rate: %.3f", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %d", total_episodes)


if __name__ == "__main__":
    main()
