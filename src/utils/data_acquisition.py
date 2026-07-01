"""propose: load, create, and validate atari environments for rl benchmarking.
input: env config dicts, env id strings, seed values.
output: initialized gymnasium environments (raw, preprocessed, vectorized)."""

from typing import Any


def _merge_preprocessing(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    from src.config.loader import load_all_configs

    pre = dict(load_all_configs()["preprocessing"]["preprocessing"])
    if overrides:
        pre.update(overrides)
    return pre


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


def make_raw_env(
    env_id: str,
    seed: int | None = None,
    render_mode: str | None = None,
    env_overrides: dict[str, Any] | None = None,
):
    gym = import_gymnasium()
    pre = _merge_preprocessing(env_overrides)
    kwargs = {}
    if render_mode:
        kwargs["render_mode"] = render_mode
    if env_id.startswith("ALE/") and "repeat_action_probability" in pre:
        kwargs["repeat_action_probability"] = float(pre["repeat_action_probability"])
    env = gym.make(env_id, frameskip=1, **kwargs) if env_id.startswith("ALE/") else gym.make(env_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    return env


def _wrap_fire_after_life_loss(env: Any):
    import gymnasium as gym

    class FireAfterLifeLossEnv(gym.Wrapper):
        """Press FIRE once after non-terminal life loss to keep full-game eval moving."""

        def __init__(self, wrapped_env: Any) -> None:
            super().__init__(wrapped_env)
            self.fire_action = self._find_fire_action()
            self.lives = 0

        def _find_fire_action(self) -> int | None:
            try:
                meanings = self.unwrapped.get_action_meanings()
            except Exception:
                return None
            for idx, meaning in enumerate(meanings):
                if meaning == "FIRE":
                    return idx
            for idx, meaning in enumerate(meanings):
                if "FIRE" in meaning:
                    return idx
            return None

        def _ale_lives(self) -> int:
            try:
                return int(self.unwrapped.ale.lives())
            except Exception:
                return 0

        def reset(self, **kwargs: Any):
            obs, info = self.env.reset(**kwargs)
            self.lives = self._ale_lives()
            return obs, info

        def step(self, action: int):
            obs, reward, terminated, truncated, info = self.env.step(action)
            current_lives = self._ale_lives()
            lost_life = self.lives > 0 and current_lives > 0 and current_lives < self.lives
            self.lives = current_lives

            if lost_life and not (terminated or truncated) and self.fire_action is not None:
                fire_obs, fire_reward, fire_terminated, fire_truncated, fire_info = self.env.step(self.fire_action)
                obs = fire_obs
                reward += fire_reward
                terminated = terminated or fire_terminated
                truncated = truncated or fire_truncated
                info.update(fire_info)
                self.lives = self._ale_lives()

            return obs, reward, terminated, truncated, info

    return FireAfterLifeLossEnv(env)


def make_atari_env(
    env_id: str,
    seed: int | None = None,
    render_mode: str | None = None,
    env_overrides: dict[str, Any] | None = None,
):
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    pre = _merge_preprocessing(env_overrides)
    env = make_raw_env(env_id, seed=seed, render_mode=render_mode, env_overrides=pre)
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
    if pre.get("fire_reset", False):
        from stable_baselines3.common.atari_wrappers import FireResetEnv

        env = FireResetEnv(env)
    if pre.get("fire_after_life_loss", False):
        env = _wrap_fire_after_life_loss(env)
    if pre.get("clip_reward", False):
        from stable_baselines3.common.atari_wrappers import ClipRewardEnv

        env = ClipRewardEnv(env)
    env = FrameStackObservation(env, stack_size=pre["frame_stack"])
    return env


def make_vec_atari_env(
    env_id: str,
    n_envs: int,
    seed: int | None = None,
    env_overrides: dict[str, Any] | None = None,
):
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_one(rank: int):
        def _init():
            env_seed = None if seed is None else seed + rank
            return make_atari_env(env_id, seed=env_seed, env_overrides=env_overrides)

        return _init

    return DummyVecEnv([make_one(rank) for rank in range(int(n_envs))])
