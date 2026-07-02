"""propose: benchmark runner entrypoint -- orchestrate training runs.
input: profile name (cli arg), configs/*.json.
output: results csv + manifest json under evals/checkpoints/<profile>.
    """

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.config import settings
from src.config.loader import config_hash, load_all_configs
from src.config.logger import logger
from src.models.training import train_and_evaluate
from src.utils.data_acquisition import make_atari_env, resolve_all_envs


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_profile_lock(lock_path: Path, force: bool = False):
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8-sig"))
            locked_pid = int(payload.get("pid", 0))
        except Exception:
            locked_pid = 0

        if not force and _pid_is_running(locked_pid):
            raise RuntimeError(
                f"profile lock exists at {lock_path}; process {locked_pid} is already running this profile"
            )
        lock_path.unlink(missing_ok=True)

    lock_payload = {
        "pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    lock_path.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")

    def release() -> None:
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8-sig"))
            if int(payload.get("pid", 0)) == os.getpid():
                lock_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("could not release profile lock %s: %s", lock_path, exc)

    return release


def record_playback_video(
    algo: str,
    model_path: str,
    env_id: str,
    seed: int,
    video_dir: str,
    env_overrides: dict[str, Any] | None = None,
    max_steps: int = 5000,
    deterministic: bool = True,
) -> dict[str, Any] | None:
    """record one deterministic episode as mp4. best-effort, logs warning on failure."""
    try:
        from gymnasium.wrappers import RecordVideo

        path = Path(model_path)
        device = "auto"

        if algo.lower() == "discretesac":
            from src.models.discrete_sac import DiscreteSAC

            env = make_atari_env(env_id, seed=seed, env_overrides=env_overrides)
            model = DiscreteSAC(
                "CnnPolicy",
                env,
                seed=seed,
                device=device,
                buffer_size=1,
                learning_starts=1,
                batch_size=1,
            )
            model.load(str(path))
            env.close()
        elif algo.lower() == "dqn":
            from stable_baselines3 import DQN

            model = DQN.load(str(path), device=device)
        elif algo.lower() == "ppo":
            from stable_baselines3 import PPO

            model = PPO.load(str(path), device=device)
        else:
            return None

        env = make_atari_env(env_id, seed=seed, render_mode="rgb_array", env_overrides=env_overrides)
        out = Path(video_dir)
        out.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(out),
            name_prefix=f"{algo.lower()}_{env_id.split('/')[-1]}_seed{seed}",
            episode_trigger=lambda e: True,
        )
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        action_counts: Counter[int] = Counter()
        try:
            action_meanings = list(env.unwrapped.get_action_meanings())
        except Exception:
            action_meanings = []
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_int = int(action.item()) if hasattr(action, "item") else int(action)
            action_counts[action_int] += 1
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        env.close()
        logger.info(
            "playback: %s/%s seed=%d steps=%d actions=%s",
            algo,
            env_id,
            seed,
            steps,
            dict(sorted(action_counts.items())),
        )
        return {
            "video_dir": str(out),
            "steps": steps,
            "action_counts": dict(sorted(action_counts.items())),
            "action_meanings": action_meanings,
            "deterministic": deterministic,
            "truncated_by_step_limit": steps >= max_steps and not done,
        }
    except Exception as exc:
        logger.warning("playback failed for %s/%s seed=%d: %s", algo, env_id, seed, exc)
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atari rl benchmark runner")
    parser.add_argument(
        "profile",
        nargs="?",
        default="300k_3seeds",
        help="profile name from configs/benchmark.json profiles block",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="override timesteps for this run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="override seeds for this run (space-separated)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        nargs="+",
        default=None,
        help="run one or more algos only (dqn ppo discretesac)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="run single env only (pong, breakout, space invaders)",
    )
    parser.add_argument(
        "--force-lock",
        action="store_true",
        help="replace an existing profile lock if it is known to be stale",
    )
    return parser.parse_args(argv)


def build_manifest(
    profile: str,
    profile_cfg: dict[str, Any],
    configs: dict[str, Any],
    resolved_envs: dict[str, str],
    evals_dir: Path,
) -> dict[str, Any]:
    return {
        "profile": profile,
        "profile_config": profile_cfg,
        "config_hash": config_hash(configs),
        "started_at": datetime.now(UTC).isoformat(),
        "timesteps": profile_cfg["timesteps"],
        "evaluation_episodes": profile_cfg["evaluation_episodes"],
        "seeds": profile_cfg["seeds"],
        "checkpoint_freq": profile_cfg.get("checkpoint_freq", 0),
        "checkpoint_root": str(evals_dir / "checkpoints" / profile),
        "playback_root": str(settings.EVALUATION_DIR / "playback" / profile),
        "resolved_envs": resolved_envs,
    }


def build_result_row(
    run_result: dict[str, Any],
    status: str,
    failure_reason: str,
    profile: str,
    cfg_hash: str,
) -> dict[str, Any]:
    row = dict(run_result)
    row["Status"] = status
    row["FailureReason"] = failure_reason
    row["Profile"] = profile
    row["ConfigHash"] = cfg_hash
    return row


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("wrote %d rows to %s", len(results), path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = args.profile

    logger.info("project: %s", settings.PROJECT_NAME)
    logger.info("profile: %s", profile)

    try:
        import torch

        cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda else "n/a"
        device_count = torch.cuda.device_count() if cuda else 0
        logger.info("gpu: cuda=%s count=%d device=%s", cuda, device_count, device_name)
    except ImportError:
        logger.info("gpu: torch not available")

    configs = load_all_configs()
    benchmark_cfg = configs["benchmark"]
    algo_cfgs = configs["algorithms"]["algorithms"]
    cfg_hash = config_hash(configs)

    profiles = benchmark_cfg.get("profiles", {})
    if profile not in profiles:
        logger.error("unknown profile '%s'. available: %s", profile, list(profiles.keys()))
        return 1
    profile_cfg = dict(profiles[profile])

    if args.timesteps is not None:
        profile_cfg["timesteps"] = args.timesteps
    if args.seeds is not None:
        profile_cfg["seeds"] = args.seeds

    timesteps = profile_cfg["timesteps"]
    seeds = profile_cfg["seeds"]
    eval_episodes = profile_cfg["evaluation_episodes"]
    checkpoint_freq = profile_cfg.get("checkpoint_freq", 0)
    resolved_envs = resolve_all_envs(configs)
    logger.info("resolved envs: %s", resolved_envs)

    enabled_algos = {k: v for k, v in algo_cfgs.items() if v.get("enabled", True)}
    if args.algo:
        requested_algos = {name.lower() for name in args.algo}
        enabled_algos = {k: v for k, v in enabled_algos.items() if k.lower() in requested_algos}
        if not enabled_algos:
            logger.error("algos '%s' not found or disabled", " ".join(args.algo))
            return 1

    filtered_envs = dict(resolved_envs)
    if args.env:
        args_env_lower = args.env.lower()
        filtered_envs = {k: v for k, v in filtered_envs.items() if k.lower() == args_env_lower}
        if not filtered_envs:
            logger.error("env '%s' not found", args.env)
            return 1

    runs: list[tuple[str, str, str, int]] = []
    total = 0
    for algo_name in enabled_algos:
        for env_name, env_id in filtered_envs.items():
            for seed in seeds:
                runs.append((algo_name, env_name, env_id, seed))
                total += 1

    logger.info(
        "plan: %d algos x %d envs x %d seeds = %d runs @ %d timesteps each",
        len(enabled_algos),
        len(filtered_envs),
        len(seeds),
        total,
        timesteps,
    )

    evals_dir = settings.BASE_DIR / "evals"
    run_dir = evals_dir / "checkpoints" / profile
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = evals_dir / "checkpoints" / profile
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    playback_root = settings.EVALUATION_DIR / "playback" / profile
    playback_root.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_root / f"{profile}.lock"
    try:
        release_lock = acquire_profile_lock(lock_path, force=args.force_lock)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("checkpoint root: %s", checkpoint_root)
    logger.info("playback root: %s", playback_root)
    logger.info("profile lock: %s", lock_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    result_path = run_dir / f"{profile}_results_{timestamp}.csv"
    manifest_path = run_dir / f"{profile}_manifest_{timestamp}.json"

    manifest = build_manifest(profile, profile_cfg, configs, resolved_envs, evals_dir)
    results: list[dict[str, Any]] = []
    start_global = time.perf_counter()
    completed = 0
    failed = 0

    try:
        for algo_name, env_name, env_id, seed in runs:
            algo_cfg = dict(enabled_algos[algo_name])
            run_label = f"{algo_name}/{env_name}/seed_{seed}"
            logger.info("starting %s ...", run_label)

            chk_dir = checkpoint_root / env_name.lower().replace(" ", "_") / algo_name.lower() / f"seed_{seed}"
            run_start = time.perf_counter()
            status = "completed"
            failure = ""

            try:
                run_result = train_and_evaluate(
                    algo=algo_name,
                    algo_cfg=algo_cfg,
                    env_name=env_name,
                    env_id=env_id,
                    seed=seed,
                    timesteps=timesteps,
                    eval_episodes=eval_episodes,
                    configs=configs,
                    checkpoint_freq=checkpoint_freq,
                    checkpoint_dir=str(chk_dir),
                )
                completed += 1
                fp = run_result.get("FinalModelPath", "")
                if fp:
                    vid_dir = playback_root / env_name.lower().replace(" ", "_") / algo_name.lower()
                    playback_env_overrides = dict(algo_cfg.get("env", {}))
                    playback_env_overrides.update(algo_cfg.get("eval_env", {}))
                    playback = record_playback_video(
                        algo_name,
                        fp,
                        env_id,
                        seed,
                        str(vid_dir),
                        env_overrides=playback_env_overrides,
                        max_steps=int(algo_cfg.get("playback_max_steps", 5000)),
                        deterministic=bool(algo_cfg.get("playback_deterministic", True)),
                    )
                    if playback:
                        run_result["PlaybackDir"] = playback["video_dir"]
                        run_result["PlaybackSteps"] = playback["steps"]
                        run_result["PlaybackActionCounts"] = json.dumps(playback["action_counts"], sort_keys=True)
                        run_result["PlaybackActionMeanings"] = json.dumps(playback["action_meanings"])
                        run_result["PlaybackDeterministic"] = playback["deterministic"]
                        run_result["PlaybackTruncatedByStepLimit"] = playback["truncated_by_step_limit"]
            except Exception as exc:
                logger.exception("run %s failed", run_label)
                run_result = {
                    "Algorithm": algo_name,
                    "Environment": env_name,
                    "EnvID": env_id,
                    "Seed": seed,
                    "Timesteps": timesteps,
                    "EvalEpisodes": eval_episodes,
                    "Final_Reward_Mean": 0.0,
                    "Final_Reward_Std": 0.0,
                    "Final_Reward_Min": 0.0,
                    "Final_Reward_Max": 0.0,
                    "Training_Seconds": time.perf_counter() - run_start,
                    "Reward_Per_Hour": 0.0,
                    "InputShape": "",
                    "InputDType": "",
                    "InputRange": "",
                    "InputLayout": "",
                    "InputContiguous": "",
                    "NumEnvs": int(algo_cfg.get("n_envs", 1)),
                    "CheckpointDir": "",
                    "FinalModelPath": "",
                    "DiagnosticsPath": "",
                }
                status = "failed"
                failure = str(exc)
                failed += 1

            row = build_result_row(run_result, status, failure, profile, cfg_hash)
            results.append(row)
            logger.info(
                "finished %s: %s (%.1fs)",
                run_label,
                status,
                time.perf_counter() - run_start,
            )

            write_csv(results, result_path)
    finally:
        release_lock()

    elapsed = time.perf_counter() - start_global
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["rows"] = len(results)
    manifest["total_runs"] = total
    manifest["completed"] = completed
    manifest["failed"] = failed
    manifest["elapsed_seconds"] = round(elapsed, 2)
    manifest["result_file"] = str(result_path)
    manifest["runtime"] = f"{sys.executable} | python {sys.version}"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("manifest written to %s", manifest_path)

    print(f"\n{'=' * 50}")
    print(f"profile:     {profile}")
    print(f"runs:        {completed} completed, {failed} failed (of {total})")
    print(f"time:        {elapsed:.1f}s total")
    print(f"results:     {result_path}")
    print(f"manifest:    {manifest_path}")
    print(f"{'=' * 50}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
