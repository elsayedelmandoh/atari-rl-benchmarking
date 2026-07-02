"""Record playback videos from saved benchmark checkpoints.

Usage:
    python -m src.inference.record_playback 1m_1seed_StaDiscSac_diagnostic pong
    python -m src.inference.record_playback 1m_1seed_StaDiscSac_diagnostic all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app import record_playback_video
from src.config.config import settings
from src.config.loader import load_all_configs
from src.utils.data_acquisition import resolve_all_envs


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def _match_env(requested_env: str, resolved_envs: dict[str, str]) -> dict[str, str]:
    requested = requested_env.strip().lower().replace("_", " ")
    if requested in {"all", "*"}:
        return resolved_envs

    matches = {
        env_name: env_id
        for env_name, env_id in resolved_envs.items()
        if requested == env_name.lower() or requested == _slug(env_name)
    }
    if matches:
        return matches

    available = ", ".join(resolved_envs)
    raise ValueError(f"unknown env '{requested_env}'. available: {available}, all")


def _model_extension(algo_cfg: dict[str, Any]) -> str:
    return ".pt" if algo_cfg.get("class_name") == "DiscreteSAC" else ".zip"


def record_profile_playback(
    profile: str,
    requested_env: str,
    algo: str = "DiscreteSAC",
    seed: int = 0,
    output_profile: str | None = None,
) -> list[dict[str, Any]]:
    configs = load_all_configs()
    resolved_envs = resolve_all_envs(configs)
    selected_envs = _match_env(requested_env, resolved_envs)

    algo_cfgs = configs["algorithms"]["algorithms"]
    algo_cfg = next((cfg for name, cfg in algo_cfgs.items() if name.lower() == algo.lower()), None)
    if algo_cfg is None:
        available = ", ".join(algo_cfgs)
        raise ValueError(f"unknown algo '{algo}'. available: {available}")

    env_overrides = dict(algo_cfg.get("env", {}))
    env_overrides.update(algo_cfg.get("eval_env", {}))

    output_name = output_profile or f"{profile}_regenerated"
    extension = _model_extension(algo_cfg)
    results: list[dict[str, Any]] = []

    for env_name, env_id in selected_envs.items():
        env_slug = _slug(env_name)
        algo_slug = algo.lower()
        model_path = (
            settings.BASE_DIR
            / "evals"
            / "checkpoints"
            / profile
            / env_slug
            / algo_slug
            / f"seed_{seed}"
            / f"final_model{extension}"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"missing model checkpoint: {model_path}")

        video_dir = settings.EVALUATION_DIR / "playback" / output_name / env_slug / algo_slug
        print(f"recording {algo}/{env_name}/seed_{seed}")
        print(f"model: {model_path}")
        print(f"video: {video_dir}")

        playback = record_playback_video(
            algo,
            str(model_path),
            env_id,
            seed,
            str(video_dir),
            env_overrides=env_overrides,
            max_steps=int(algo_cfg.get("playback_max_steps", 5000)),
            deterministic=bool(algo_cfg.get("playback_deterministic", True)),
        )
        if playback is None:
            raise RuntimeError(f"playback failed for {algo}/{env_name}/seed_{seed}")

        result = {"Environment": env_name, "ModelPath": str(model_path), **playback}
        results.append(result)
        print(
            "done: "
            f"steps={playback['steps']} "
            f"actions={playback['action_counts']} "
            f"dir={playback['video_dir']}"
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="record playback from a saved benchmark profile")
    parser.add_argument("profile", help="benchmark profile name, e.g. 1m_1seed_StaDiscSac_diagnostic")
    parser.add_argument("env", help="environment name: pong, breakout, space invaders, or all")
    parser.add_argument("--algo", default="DiscreteSAC", help="algorithm name; default: DiscreteSAC")
    parser.add_argument("--seed", type=int, default=0, help="checkpoint seed; default: 0")
    parser.add_argument(
        "--output-profile",
        default=None,
        help="playback output folder name; default: <profile>_regenerated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record_profile_playback(
            profile=args.profile,
            requested_env=args.env,
            algo=args.algo,
            seed=args.seed,
            output_profile=args.output_profile,
        )
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
