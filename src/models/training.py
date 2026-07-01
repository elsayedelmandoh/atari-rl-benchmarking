"""propose: training harness -- model creation, checkpointing, training loop, evaluation.
input: algorithm config, environment id, hyperparameters.
output: trained model instance, evaluation stats dict."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from src.evaluation.validation import assert_model_input_contract
from src.utils.data_acquisition import make_atari_env, make_vec_atari_env
from src.utils.helpers import set_global_seed


class PpoDiagnosticsCallback:
    """Write rollout-level PPO diagnostics without changing the optimizer."""

    LOGGER_KEYS = (
        "train/approx_kl",
        "train/clip_fraction",
        "train/clip_range",
        "train/entropy_loss",
        "train/explained_variance",
        "train/learning_rate",
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "rollout/ep_len_mean",
        "rollout/ep_rew_mean",
    )

    FIELDNAMES = (
        "Timesteps",
        "RolloutActions",
        "RolloutActionEntropy",
        "RolloutDominantAction",
        "RolloutDominantActionFraction",
        "train_approx_kl",
        "train_clip_fraction",
        "train_clip_range",
        "train_entropy_loss",
        "train_explained_variance",
        "train_learning_rate",
        "train_loss",
        "train_policy_gradient_loss",
        "train_value_loss",
        "rollout_ep_len_mean",
        "rollout_ep_rew_mean",
    )

    def __init__(self, output_path: str | Path):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: PpoDiagnosticsCallback):
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                return True

            def _on_rollout_end(self) -> None:
                self.outer.write_rollout(self.model, self.num_timesteps)

        self.output_path = Path(output_path)
        self.callback = _Callback(self)

    def write_rollout(self, model: Any, timesteps: int) -> None:
        import numpy as np

        actions = np.asarray(model.rollout_buffer.actions).reshape(-1)
        action_counts: dict[int, int] = {}
        if actions.size:
            unique, counts = np.unique(actions.astype(int), return_counts=True)
            action_counts = {int(action): int(count) for action, count in zip(unique, counts, strict=False)}

        total_actions = sum(action_counts.values())
        entropy = 0.0
        dominant_action = ""
        dominant_fraction = 0.0
        if total_actions:
            entropy = -sum(
                (count / total_actions) * math.log(count / total_actions)
                for count in action_counts.values()
                if count > 0
            )
            dominant_action, dominant_count = max(action_counts.items(), key=lambda item: item[1])
            dominant_fraction = dominant_count / total_actions

        row: dict[str, Any] = {
            "Timesteps": timesteps,
            "RolloutActions": json.dumps(action_counts, sort_keys=True),
            "RolloutActionEntropy": round(entropy, 6),
            "RolloutDominantAction": dominant_action,
            "RolloutDominantActionFraction": round(dominant_fraction, 6),
        }

        logger_values = getattr(getattr(model, "logger", None), "name_to_value", {})
        for key in self.LOGGER_KEYS:
            out_key = key.replace("/", "_")
            row[out_key] = _jsonable_scalar(logger_values.get(key, ""))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists()
        with self.output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, str) and value == "":
        return ""
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def make_model(algo: str, algo_cfg: dict[str, Any], env: Any, seed: int):
    class_name = algo_cfg["class_name"]

    if class_name == "DiscreteSAC":
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

    if class_name == "DQN":
        from src.models.dqn import make_dqn

        return make_dqn(env, seed, algo_cfg.get("params", {}))

    if class_name == "PPO":
        from src.models.ppo import make_ppo

        return make_ppo(env, seed, algo_cfg.get("params", {}))

    msg = f"unknown algorithm class: {class_name}"
    raise ValueError(msg)


def evaluate_model(
    model: Any,
    env_id: str,
    seed: int,
    episodes: int,
    env_overrides: dict[str, Any] | None = None,
    deterministic: bool = True,
) -> dict[str, float]:
    import numpy as np

    env = make_atari_env(env_id, seed=seed, env_overrides=env_overrides)
    rewards: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
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
    env_overrides = algo_cfg.get("env", {})
    eval_env_overrides = dict(env_overrides)
    eval_env_overrides.update(algo_cfg.get("eval_env", {}))
    input_env = make_atari_env(env_id, seed=seed, env_overrides=env_overrides)
    obs, _ = input_env.reset(seed=seed)
    input_meta = assert_model_input_contract(algo, obs, configs)
    input_env.close()

    n_envs = int(algo_cfg.get("n_envs", 1))
    env = (
        make_vec_atari_env(env_id, n_envs=n_envs, seed=seed, env_overrides=env_overrides)
        if n_envs > 1
        else make_atari_env(env_id, seed=seed, env_overrides=env_overrides)
    )

    start = time.perf_counter()
    model = make_model(algo, algo_cfg, env, seed)

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_path:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    diagnostics_path = None
    if algo_cfg.get("library") == "custom" and algo_cfg["class_name"] == "DiscreteSAC":
        if checkpoint_path:
            diagnostics_path = checkpoint_path / "discrete_sac_diagnostics.csv"
        model.learn(
            total_timesteps=timesteps,
            checkpoint_freq=checkpoint_freq,
            checkpoint_dir=checkpoint_path,
            diagnostics_path=diagnostics_path,
        )
        if checkpoint_path:
            final_model_path = checkpoint_path / "final_model.pt"
            model.save(final_model_path)
        else:
            final_model_path = None
    else:
        callbacks = []
        if checkpoint_freq and checkpoint_path:
            from stable_baselines3.common.callbacks import CheckpointCallback

            save_freq = max(int(checkpoint_freq) // max(n_envs, 1), 1)
            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(checkpoint_path),
                    name_prefix=f"{algo.lower()}",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )

        if algo_cfg["class_name"] == "PPO" and checkpoint_path:
            diagnostics_path = checkpoint_path / "ppo_diagnostics.csv"
            callbacks.append(PpoDiagnosticsCallback(diagnostics_path).callback)

        callback = None
        if callbacks:
            from stable_baselines3.common.callbacks import CallbackList

            callback = CallbackList(callbacks)

        model.learn(total_timesteps=timesteps, callback=callback)
        if checkpoint_path:
            final_model_path = checkpoint_path / "final_model.zip"
            model.save(str(final_model_path))
        else:
            final_model_path = None

    elapsed = time.perf_counter() - start
    eval_deterministic = bool(algo_cfg.get("eval_deterministic", True))
    eval_stats = evaluate_model(
        model,
        env_id,
        seed,
        eval_episodes,
        env_overrides=eval_env_overrides,
        deterministic=eval_deterministic,
    )
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
        "Reward_Per_Hour": (eval_stats["mean"] / elapsed * 3600) if elapsed > 0 else 0.0,
        "InputShape": str(input_meta.get("shape", "")),
        "InputDType": str(input_meta.get("dtype", "")),
        "InputRange": str(input_meta.get("range", "")),
        "InputLayout": str(input_meta.get("layout", "")),
        "InputContiguous": str(input_meta.get("contiguous", "")),
        "NumEnvs": n_envs,
        "CheckpointDir": str(checkpoint_dir) if checkpoint_dir else "",
        "FinalModelPath": str(final_model_path) if final_model_path else "",
        "DiagnosticsPath": str(diagnostics_path) if diagnostics_path else "",
        "EvalDeterministic": eval_deterministic,
    }
