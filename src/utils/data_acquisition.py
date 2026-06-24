"""propose: load, create, and validate atari environments for rl benchmarking.
input: env config dicts, env id strings, seed values.
output: initialized gymnasium environments (raw, preprocessed, vectorized)."""

from typing import Any


def import_gymnasium():
    import ale_py
    import gymnasium as gym

    try:
        gym.register_envs(ale_py)
    except Exception:
        pass
    return gym


def resolve_env_id(env_config: dict[str, Any]) -> tuple[str | None, list[str]]:
    gym = import_gymnasium()
    errors: list[str] = []
    for env_id in env_config["candidate_ids"]:
        try:
            env = gym.make(env_id)
            env.close()
            return env_id, errors
        except Exception as exc:
            errors.append(f"{env_id}: {exc}")
    return None, errors


def resolve_all_envs(configs: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for env_config in configs["envs"]["environments"]:
        env_id, errors = resolve_env_id(env_config)
        if env_id is None:
            joined = "\n".join(errors)
            raise RuntimeError(f"Could not resolve environment for {env_config['name']}:\n{joined}")
        resolved[env_config["name"]] = env_id
    return resolved


def make_raw_env(env_id: str, seed: int | None = None, render_mode: str | None = None):
    gym = import_gymnasium()
    kwargs = {}
    if render_mode:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_id, frameskip=1, **kwargs) if env_id.startswith("ALE/") else gym.make(env_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    return env


def make_atari_env(env_id: str, seed: int | None = None, render_mode: str | None = None):
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    from src.config.loader import load_all_configs

    pre = load_all_configs()["preprocessing"]["preprocessing"]
    env = make_raw_env(env_id, seed=seed, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        noop_max=pre["noop_max"],
        frame_skip=pre["frame_skip"],
        screen_size=pre["screen_size"],
        terminal_on_life_loss=pre["terminal_on_life_loss"],
        grayscale_obs=pre["grayscale_obs"],
        grayscale_newaxis=pre["grayscale_newaxis"],
        scale_obs=pre["scale_obs"],
    )
    env = FrameStackObservation(env, stack_size=pre["frame_stack"])
    return env


def make_vec_atari_env(env_id: str, n_envs: int, seed: int | None = None):
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_one(rank: int):
        def _init():
            env_seed = None if seed is None else seed + rank
            return make_atari_env(env_id, seed=env_seed)

        return _init

    return DummyVecEnv([make_one(rank) for rank in range(int(n_envs))])
