import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import queue
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

starvla_torch_site = os.environ.get("STARVLA_TORCH_SITE")
if starvla_torch_site and starvla_torch_site not in sys.path:
    sys.path.append(starvla_torch_site)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import imageio
import numpy as np
try:
    import tqdm
except ImportError:
    class _TqdmFallback:
        @staticmethod
        def tqdm(iterable, *args, **kwargs):
            return iterable

    tqdm = _TqdmFallback()
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
SERVER_PYTHON_FALLBACK = "/home/wangbo/miniconda3/envs/starVLA/bin/python"


def _configure_tokenizers_parallelism(enabled: bool) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if enabled else "false"


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


def _parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_csv_arg(items: Sequence[object]) -> str:
    return ",".join(str(item) for item in items)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _build_model(config: Dict, host: str, port: int):
    from examples.LIBERO.model2libero_interface import M1Inference

    return M1Inference(
        policy_ckpt_path=config["pretrained_path"],
        unnorm_key=config["unnorm_key"],
        host=host,
        port=port,
        image_size=[224, 224],
        enable_latent_reasoning=config["enable_latent_reasoning"],
        cot_mode=config["cot_mode"],
        thinking_token_count=config["thinking_token_count"],
        img_next_count=config["img_next_count"],
    )


def _load_category_map(
    task_suite_name: str,
    num_tasks: int,
    num_trials_per_task: int,
) -> Tuple[Dict[int, Tuple[str, str]], Dict[str, Dict[str, int]]]:
    id2category: Dict[int, Tuple[str, str]] = {}
    disturb_res: Dict[str, Dict[str, int]] = {}
    libero_home = os.environ.get("LIBERO_HOME", "")
    mapping_path = os.path.join(libero_home, "libero/libero/benchmark/task_classification.json")
    if not os.path.exists(mapping_path):
        return id2category, disturb_res

    with open(mapping_path, "r", encoding="utf-8") as f:
        task_mapping = json.load(f).get(task_suite_name, [])

    for item in task_mapping:
        category = item["category"]
        task_id = int(item["id"])
        id2category[task_id] = (category, item["name"])
        if task_id <= num_tasks:
            disturb_res.setdefault(category, {"total_count": 0, "success_count": 0})
            disturb_res[category]["total_count"] += num_trials_per_task

    return id2category, disturb_res


def _safe_video_segment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _write_disturb_json(args, disturb_res: Dict[str, Dict[str, int]]) -> None:
    if not disturb_res:
        return
    with open(pathlib.Path(args.log_path) / f"{args.task_suite_name}.json", "w", encoding="utf-8") as f:
        json.dump(disturb_res, f, ensure_ascii=False)


def _run_episode(
    model,
    env,
    initial_state,
    task_description: str,
    max_steps: int,
    num_steps_wait: int,
    save_videos: bool,
):
    model.reset(task_description=task_description)
    env.reset()
    obs = env.set_init_state(initial_state)
    replay_images = [] if save_videos else None
    t = 0
    step = 0
    success = False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        if replay_images is not None:
            replay_images.append(img)

        response = model.step(images=[img, wrist_img], task_description=str(task_description), step=step)
        raw_action = response["raw_action"]
        delta_action = np.concatenate(
            [
                np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1),
                np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1),
                _binarize_gripper_open(raw_action["open_gripper"]),
            ],
            axis=0,
        )
        if delta_action.size != 7:
            raise ValueError(f"invalid action shape {delta_action.shape}")

        obs, _, done, _ = env.step(delta_action.tolist())
        if done:
            success = True
            break
        t += 1
        step += 1

    return success, replay_images


def _make_assignments(num_tasks: int, num_trials_per_task: int, num_processes: int) -> List[List[Tuple[int, int]]]:
    assignments = [[] for _ in range(num_processes)]
    idx = 0
    for task_id in range(num_tasks):
        for episode_idx in range(num_trials_per_task):
            assignments[idx % num_processes].append((task_id, episode_idx))
            idx += 1
    return assignments


def _resolve_server_gpus(server_gpus: str) -> List[str]:
    if server_gpus:
        return _parse_csv_arg(server_gpus)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return _parse_csv_arg(visible)


def _detect_idle_gpu_ids(max_utilization: int, max_memory_mb: int) -> List[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        logging.warning("Auto parallel GPU detection failed: %s", exc)
        return []

    gpu_ids = []
    for raw_line in output.splitlines():
        parts = [item.strip() for item in raw_line.split(",")]
        if len(parts) != 4:
            continue
        gpu_id, util, used_mem, _total_mem = parts
        try:
            util_value = int(util)
            used_mem_value = int(used_mem)
        except ValueError:
            continue
        if util_value <= max_utilization and used_mem_value <= max_memory_mb:
            gpu_ids.append(gpu_id)
    return gpu_ids


def _auto_configure_parallelism(args, total_episodes: int) -> None:
    if not args.auto_parallel:
        return

    if args.hosts or args.ports or args.server_gpus or args.render_gpu_ids or args.num_processes > 0:
        logging.info("Auto parallel skipped because manual parallel parameters are already provided.")
        return

    gpu_ids = _detect_idle_gpu_ids(args.auto_gpu_max_utilization, args.auto_gpu_max_memory_mb)
    if not gpu_ids:
        logging.info("Auto parallel found no idle GPUs; keeping sequential/single-endpoint settings.")
        return

    if args.auto_max_endpoints > 0:
        gpu_ids = gpu_ids[: args.auto_max_endpoints]

    endpoint_count = min(len(gpu_ids), max(1, total_episodes))
    if endpoint_count <= 1:
        logging.info("Auto parallel found only one usable endpoint; keeping sequential/single-endpoint settings.")
        return

    gpu_ids = gpu_ids[:endpoint_count]
    args.launch_servers = True
    args.server_gpus = _format_csv_arg(gpu_ids)
    args.render_gpu_ids = _format_csv_arg(gpu_ids)
    args.ports = _format_csv_arg(args.port + idx for idx in range(endpoint_count))
    args.hosts = _format_csv_arg([args.host] * endpoint_count)
    args.num_processes = endpoint_count

    cpu_count = os.cpu_count() or endpoint_count
    logging.info(
        "Auto parallel selected %d endpoints on GPUs=%s (cpu_count=%s, total_episodes=%d)",
        endpoint_count,
        gpu_ids,
        cpu_count,
        total_episodes,
    )


def _resolve_endpoints(args) -> List[Tuple[str, int]]:
    ports = [int(item) for item in _parse_csv_arg(args.ports)] if args.ports else [args.port]
    if args.launch_servers and not args.ports:
        gpu_candidates = _resolve_server_gpus(args.server_gpus)
        count = len(gpu_candidates) if gpu_candidates else max(1, args.num_processes)
        ports = [args.port + idx for idx in range(count)]

    hosts = _parse_csv_arg(args.hosts) if args.hosts else [args.host] * len(ports)
    if len(hosts) == 1 and len(ports) > 1:
        hosts = hosts * len(ports)
    if len(hosts) != len(ports):
        raise ValueError("hosts and ports must have the same length")
    return list(zip(hosts, ports))


def _resolve_render_gpu_ids(render_gpu_ids: str, count: int) -> List[str]:
    gpu_ids = _parse_csv_arg(render_gpu_ids)
    if gpu_ids and len(gpu_ids) == 1 and count > 1:
        gpu_ids = gpu_ids * count
    if gpu_ids and len(gpu_ids) != count:
        raise ValueError("render-gpu-ids must be empty, length 1, or match number of endpoints")
    return gpu_ids


def _choose_server_python(server_python: str) -> str:
    candidates = [
        server_python,
        os.environ.get("SERVER_PY"),
        os.environ.get("LARAVLA_PYTHON"),
        SERVER_PYTHON_FALLBACK,
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate) and not os.path.exists(candidate):
            continue
        return candidate
    return sys.executable


def _wait_for_websocket_server(host: str, port: int, timeout: int) -> None:
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)

    import websockets.sync.client

    uri = f"ws://{host}:{port}"
    start = time.time()
    while time.time() - start <= timeout:
        try:
            conn = websockets.sync.client.connect(
                uri,
                compression=None,
                max_size=None,
                open_timeout=5,
                ping_interval=None,
            )
            conn.close()
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server not ready after {timeout}s: {host}:{port}")


def _launch_policy_servers(args, endpoints: Sequence[Tuple[str, int]]):
    repo_root = _repo_root()
    server_python = _choose_server_python(args.server_python)
    server_gpus = _resolve_server_gpus(args.server_gpus)
    if not server_gpus:
        server_gpus = [str(idx) for idx in range(len(endpoints))]
    if len(server_gpus) < len(endpoints):
        raise ValueError("launch-servers needs at least one server GPU per endpoint")

    log_dir = pathlib.Path(args.server_log_path or pathlib.Path(args.log_path) / "server_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    handles = []
    try:
        for idx, ((host, port), gpu_id) in enumerate(zip(endpoints, server_gpus)):
            log_path = log_dir / f"server_{port}.log"
            log_handle = open(log_path, "w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
            cmd = [
                server_python,
                str(repo_root / "deployment/model_server/server_policy.py"),
                "--ckpt_path",
                args.pretrained_path,
                "--port",
                str(port),
            ]
            if args.server_use_bf16:
                cmd.append("--use_bf16")
            logging.info("Starting policy server %d: gpu=%s host=%s port=%s log=%s", idx, gpu_id, host, port, log_path)
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            handles.append({"process": proc, "log_handle": log_handle, "log_path": log_path, "host": host, "port": port})

        for handle in handles:
            proc = handle["process"]
            host = "127.0.0.1" if handle["host"] in ("0.0.0.0", "") else handle["host"]
            port = handle["port"]
            if proc.poll() is not None:
                raise RuntimeError(f"policy server exited early on port {port}; see {handle['log_path']}")
            _wait_for_websocket_server(host, port, args.server_timeout)
            logging.info("Policy server ready: %s:%s", host, port)

        return handles
    except Exception:
        _terminate_policy_servers(handles)
        raise


def _terminate_policy_servers(handles) -> None:
    for handle in handles:
        proc = handle["process"]
        if proc.poll() is None:
            proc.terminate()
    for handle in handles:
        proc = handle["process"]
        if proc.poll() is None:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)
        handle["log_handle"].close()


def _worker_main(
    worker_id: int,
    assignments: Sequence[Tuple[int, int]],
    args_dict: Dict,
    endpoint: Tuple[str, int],
    render_gpu_id: Optional[str],
    result_queue,
) -> None:
    try:
        _configure_tokenizers_parallelism(args_dict["tokenizers_parallelism"])
        if render_gpu_id is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_id)

        log_prefix = f"[worker={worker_id} host={endpoint[0]} port={endpoint[1]}]"
        logger = logging.getLogger(f"parallel_eval.worker.{worker_id}")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.addHandler(logging.StreamHandler(sys.stdout))

        task_suite = benchmark.get_benchmark_dict()[args_dict["task_suite_name"]]()
        model = _build_model(args_dict, endpoint[0], endpoint[1])
        max_steps = _max_steps(args_dict["task_suite_name"])
        save_videos = args_dict["save_videos"]
        video_root = pathlib.Path(args_dict["video_out_path"])
        results = []

        for task_id, episode_idx in assignments:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args_dict["seed"])
            logger.info("%s task=%d episode=%d start", log_prefix, task_id, episode_idx)
            t0 = time.perf_counter()
            try:
                success, replay_images = _run_episode(
                    model=model,
                    env=env,
                    initial_state=initial_states[episode_idx],
                    task_description=task_description,
                    max_steps=max_steps,
                    num_steps_wait=args_dict["num_steps_wait"],
                    save_videos=save_videos,
                )
            finally:
                env.close()

            elapsed = time.perf_counter() - t0
            if save_videos and replay_images is not None:
                suffix = "success" if success else "failure"
                task_segment = _safe_video_segment(task_description)
                imageio.mimwrite(
                    video_root / f"worker{worker_id}_task{task_id}_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=25,
                )

            logger.info(
                "%s task=%d episode=%d done success=%s elapsed=%.2fs",
                log_prefix,
                task_id,
                episode_idx,
                success,
                elapsed,
            )
            results.append(
                {
                    "task_id": task_id,
                    "task": task_description,
                    "episode": episode_idx,
                    "success": bool(success),
                    "elapsed_sec": elapsed,
                    "worker_id": worker_id,
                    "host": endpoint[0],
                    "port": endpoint[1],
                }
            )

        result_queue.put({"worker_id": worker_id, "results": results, "error": None})
    except BaseException:
        result_queue.put({"worker_id": worker_id, "results": [], "error": traceback.format_exc()})
        raise


def _collect_worker_outputs(processes, result_queue):
    outputs = []
    while len(outputs) < len(processes):
        try:
            outputs.append(result_queue.get(timeout=5))
        except queue.Empty:
            if all(not proc.is_alive() for proc in processes):
                break

    for proc in processes:
        proc.join()

    summaries = []
    errors = []
    seen_workers = set()
    for output in outputs:
        seen_workers.add(output.get("worker_id"))
        summaries.extend(output.get("results", []))
        if output.get("error"):
            errors.append(output["error"])

    exit_codes = [proc.exitcode for proc in processes]
    missing_workers = set(range(len(processes))) - seen_workers
    if missing_workers:
        errors.append(f"missing worker outputs: {sorted(missing_workers)}")
    return summaries, exit_codes, errors


def _run_parallel(args, num_tasks: int, endpoints: Sequence[Tuple[str, int]], render_gpu_ids: Sequence[str]) -> None:
    num_processes = args.num_processes if args.num_processes > 0 else len(endpoints)
    assignments = _make_assignments(num_tasks, args.num_trials_per_task, num_processes)
    args_dict = vars(args).copy()
    args_dict["tokenizers_parallelism"] = bool(args.tokenizers_parallelism)

    log_txt = pathlib.Path(args.log_path) / f"{args.task_suite_name}.log"
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for worker_id, worker_assignments in enumerate(assignments):
        if not worker_assignments:
            continue
        endpoint = endpoints[worker_id % len(endpoints)]
        render_gpu_id = None if not render_gpu_ids else render_gpu_ids[worker_id % len(render_gpu_ids)]
        proc = ctx.Process(
            target=_worker_main,
            args=(worker_id, worker_assignments, args_dict, endpoint, render_gpu_id, result_queue),
        )
        proc.start()
        processes.append(proc)

    summaries, exit_codes, errors = _collect_worker_outputs(processes, result_queue)
    total_elapsed = time.perf_counter() - t0
    summaries.sort(key=lambda item: (item["task_id"], item["episode"]))

    total_episodes = len(summaries)
    total_successes = sum(int(item["success"]) for item in summaries)
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    throughput = total_episodes / total_elapsed if total_elapsed > 0 else 0.0

    id2category, disturb_res = _load_category_map(args.task_suite_name, num_tasks, args.num_trials_per_task)
    for summary in summaries:
        info = id2category.get(summary["task_id"] + 1)
        if info is None:
            continue
        disturb_res[info[0]]["success_count"] += int(summary["success"])

    with open(log_txt, "w", encoding="utf-8") as lf:
        lf.write(f"Task suite: {args.task_suite_name}\n")
        lf.write(f"num_tasks: {num_tasks}\n")
        lf.write(f"num_trials_per_task: {args.num_trials_per_task}\n")
        lf.write(f"num_processes: {num_processes}\n")
        lf.write(f"endpoints: {list(endpoints)}\n")
        lf.write(f"total_elapsed_sec: {total_elapsed:.4f}\n")
        lf.write(f"throughput_eps_per_sec: {throughput:.4f}\n")
        lf.write(f"total_success_rate: {total_success_rate}\n")
        for summary in summaries:
            lf.write(json.dumps(summary, ensure_ascii=False) + "\n")
        if errors:
            lf.write("\nworker_errors:\n")
            for error in errors:
                lf.write(error + "\n")

    _write_disturb_json(args, disturb_res)
    with open(pathlib.Path(args.log_path) / "parallel_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_suite_name": args.task_suite_name,
                "num_tasks": num_tasks,
                "num_trials_per_task": args.num_trials_per_task,
                "num_processes": num_processes,
                "endpoints": [{"host": host, "port": port} for host, port in endpoints],
                "total_elapsed_sec": total_elapsed,
                "throughput_eps_per_sec": throughput,
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "total_success_rate": total_success_rate,
                "exit_codes": exit_codes,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("Total success rate: %.3f", total_success_rate)
    logging.info("Total episodes: %d", total_episodes)
    logging.info("Total elapsed: %.2fs", total_elapsed)
    logging.info("Throughput: %.3f episodes/sec", throughput)

    bad_exit_codes = [code for code in exit_codes if code != 0]
    if errors or bad_exit_codes:
        raise RuntimeError(f"parallel evaluation failed: exit_codes={exit_codes}, errors={len(errors)}")


def _run_sequential(args, num_tasks: int) -> None:
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = _max_steps(args.task_suite_name)
    model = _build_model(vars(args), args.host, args.port)
    id2category, disturb_res = _load_category_map(args.task_suite_name, num_tasks, args.num_trials_per_task)

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
            try:
                for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                    logging.info("Task %d/%d episode %d: %s", task_id + 1, num_tasks, episode_idx, task_description)
                    lf.write(f"\nTask: {task_description}\n")
                    success, replay_images = _run_episode(
                        model=model,
                        env=env,
                        initial_state=initial_states[episode_idx],
                        task_description=task_description,
                        max_steps=max_steps,
                        num_steps_wait=args.num_steps_wait,
                        save_videos=args.save_videos,
                    )
                    if success:
                        task_successes += 1
                        total_successes += 1
                        if id2category:
                            category = id2category[task_id + 1][0]
                            disturb_res[category]["success_count"] += 1
                    task_episodes += 1
                    total_episodes += 1
                    if args.save_videos and replay_images is not None:
                        suffix = "success" if success else "failure"
                        item = id2category.get(task_id + 1, ("task", task_description.replace(" ", "_")))[1]
                        imageio.mimwrite(
                            pathlib.Path(args.video_out_path) / f"rollout_{_safe_video_segment(item)}_episode{episode_idx}_{suffix}.mp4",
                            [np.asarray(x) for x in replay_images],
                            fps=25,
                        )
                    cur_total = 100.0 * total_successes / total_episodes if total_episodes else 0.0
                    logging.info("Success: %s", success)
                    logging.info("# episodes completed so far: %d", total_episodes)
                    logging.info("# successes: %d (%.1f%%)", total_successes, cur_total)
                    lf.write(f"Success: {success}\n")
                    lf.write(f"# episodes completed so far: {total_episodes}\n")
                    lf.write(f"# successes: {total_successes} ({cur_total:.1f}%)\n")
                    lf.flush()
            finally:
                env.close()

            task_rate = float(task_successes) / float(task_episodes) if task_episodes else 0.0
            total_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
            logging.info("Current task success rate: %.3f", task_rate)
            logging.info("Current total success rate: %.3f", total_rate)
            lf.write(f"Current task success rate: {task_rate}\n")
            lf.write(f"Current total success rate: {total_rate}\n")
            lf.flush()

    _write_disturb_json(args, disturb_res)
    total_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    logging.info("Total success rate: %.3f", total_rate)
    logging.info("Total episodes: %d", total_episodes)


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-path", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10093)
    p.add_argument("--ports", default="")
    p.add_argument("--hosts", default="")
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
    p.add_argument("--num-processes", type=int, default=0)
    p.add_argument("--render-gpu-ids", default="")
    p.add_argument("--tokenizers-parallelism", action="store_true")
    p.add_argument("--launch-servers", action="store_true")
    p.add_argument("--server-gpus", default="")
    p.add_argument("--server-python", default="")
    p.add_argument("--server-log-path", default="")
    p.add_argument("--server-timeout", type=int, default=600)
    p.add_argument("--server-use-bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-parallel", action="store_true")
    p.add_argument("--auto-gpu-max-utilization", type=int, default=10)
    p.add_argument("--auto-gpu-max-memory-mb", type=int, default=2048)
    p.add_argument("--auto-max-endpoints", type=int, default=0)
    return p


def main():
    args = _build_argparser().parse_args()
    _configure_tokenizers_parallelism(args.tokenizers_parallelism)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    logging.info("Arguments: %s", json.dumps(vars(args), indent=2, ensure_ascii=False))
    np.random.seed(args.seed)

    pathlib.Path(args.log_path).mkdir(parents=True, exist_ok=True)
    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = task_suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, task_suite.n_tasks)
    total_episodes = num_tasks * args.num_trials_per_task
    _auto_configure_parallelism(args, total_episodes)
    endpoints = _resolve_endpoints(args)
    render_gpu_ids = _resolve_render_gpu_ids(args.render_gpu_ids, len(endpoints))

    use_parallel = args.num_processes > 1 or len(endpoints) > 1
    if render_gpu_ids and not use_parallel:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_ids[0])

    server_handles = []
    try:
        if args.launch_servers:
            server_handles = _launch_policy_servers(args, endpoints)
        if use_parallel:
            _run_parallel(args, num_tasks, endpoints, render_gpu_ids)
        else:
            _run_sequential(args, num_tasks)
    finally:
        _terminate_policy_servers(server_handles)


if __name__ == "__main__":
    main()
