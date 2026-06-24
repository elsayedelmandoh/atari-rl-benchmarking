"""propose: training harness -- model creation, checkpointing, training loop, evaluation.
input: algorithm config, environment id, hyperparameters.
output: trained model instance, evaluation stats dict."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.evaluation.validation import assert_model_input_contract
from src.utils.data_acquisition import make_atari_env, make_vec_atari_env
from src.utils.helpers import set_global_seed


def get_sb3_class(class_name: str):
    import stable_baselines3 as sb3

    return getattr(sb3, class_name)


def linear_schedule(initial_value: float):
    initial_value = float(initial_value)

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def normalize_sb3_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("lin_"):
            normalized[key] = linear_schedule(float(value.removeprefix("lin_")))
        else:
            normalized[key] = value
    return normalized


def make_model(algo: str, algo_cfg: dict[str, Any], env: Any, seed: int):
    if algo_cfg.get("library") == "custom" and algo_cfg["class_name"] == "DiscreteSAC":
        from src.models.discrete_sac import DiscreteSAC

        return DiscreteSAC(
            algo_cfg["policy"],
            env,
            seed=seed,
            verbose=0,
            device="auto",
            tensorboard_log="./logs",
            **algo_cfg.get("params", {}),
        )

    model_cls = get_sb3_class(algo_cfg["class_name"])
    return model_cls(
        algo_cfg["policy"],
        env,
        seed=seed,
        verbose=0,
        device="auto",
        tensorboard_log="./logs",
        **normalize_sb3_params(algo_cfg.get("params", {})),
    )


def evaluate_model(model: Any, env_id: str, seed: int, episodes: int) -> dict[str, float]:
    import numpy as np

    env = make_atari_env(env_id, seed=seed)
    rewards: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        rewards.append(total)
    env.close()
    return {
        "mean": float(np.mean(rewards)) if rewards else 0.0,
        "std": float(np.std(rewards)) if rewards else 0.0,
        "min": float(np.min(rewards)) if rewards else 0.0,
        "max": float(np.max(rewards)) if rewards else 0.0,
    }


def train_and_evaluate(
    algo: str,
    algo_cfg: dict[str, Any],
    env_name: str,
    env_id: str,
    seed: int,
    timesteps: int,
    eval_episodes: int,
    configs: dict[str, Any],
    checkpoint_freq: int = 0,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    set_global_seed(seed)
    input_env = make_atari_env(env_id, seed=seed)
    obs, _ = input_env.reset(seed=seed)
    input_meta = assert_model_input_contract(algo, obs, configs)
    input_env.close()

    n_envs = int(algo_cfg.get("n_envs", 1))
    env = make_vec_atari_env(env_id, n_envs=n_envs, seed=seed) if n_envs > 1 else make_atari_env(env_id, seed=seed)

    start = time.perf_counter()
    model = make_model(algo, algo_cfg, env, seed)

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_path:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    if algo_cfg.get("library") == "custom" and algo_cfg["class_name"] == "DiscreteSAC":
        model.learn(total_timesteps=timesteps, checkpoint_freq=checkpoint_freq, checkpoint_dir=checkpoint_path)
        if checkpoint_path:
            final_model_path = checkpoint_path / "final_model.pt"
            model.save(final_model_path)
        else:
            final_model_path = None
    else:
        callback = None
        if checkpoint_freq and checkpoint_path:
            from stable_baselines3.common.callbacks import CheckpointCallback

            save_freq = max(int(checkpoint_freq) // max(n_envs, 1), 1)
            callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(checkpoint_path),
                name_prefix=f"{algo.lower()}",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        model.learn(total_timesteps=timesteps, callback=callback)
        if checkpoint_path:
            final_model_path = checkpoint_path / "final_model.zip"
            model.save(str(final_model_path))
        else:
            final_model_path = None

    elapsed = time.perf_counter() - start
    eval_stats = evaluate_model(model, env_id, seed, eval_episodes)
    env.close()

    return {
        "Algorithm": algo,
        "Environment": env_name,
        "EnvID": env_id,
        "Seed": seed,
        "Timesteps": timesteps,
        "EvalEpisodes": eval_episodes,
        "Final_Reward_Mean": eval_stats["mean"],
        "Final_Reward_Std": eval_stats["std"],
        "Final_Reward_Min": eval_stats["min"],
        "Final_Reward_Max": eval_stats["max"],
        "Training_Seconds": elapsed,
        "Reward_Per_Hour": eval_stats["mean"] / (elapsed / 3600.0) if elapsed > 0 else 0.0,
        "InputShape": str(input_meta["shape"]),
        "InputDType": input_meta["dtype"],
        "InputRange": str(input_meta["range"]),
        "InputLayout": input_meta["layout"],
        "InputContiguous": input_meta["contiguous"],
        "NumEnvs": n_envs,
        "CheckpointDir": str(checkpoint_path) if checkpoint_path else "",
        "FinalModelPath": str(final_model_path) if final_model_path else "",
        "Status": "completed",
        "FailureReason": "",
    }
