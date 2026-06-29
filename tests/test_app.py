from types import SimpleNamespace

import app


def test_main_defaults_to_300k_profile_without_real_training(monkeypatch, tmp_path) -> None:
    fake_settings = SimpleNamespace(
        BASE_DIR=tmp_path,
        EVALUATION_DIR=tmp_path / "artifacts" / "evaluation",
        PROJECT_NAME="atari-rl-benchmarking",
    )
    fake_configs = {
        "benchmark": {
            "profiles": {
                "300k_3seeds": {
                    "timesteps": 300000,
                    "seeds": [0],
                    "evaluation_episodes": 1,
                    "checkpoint_freq": 100000,
                }
            }
        },
        "algorithms": {
            "algorithms": {
                "DQN": {
                    "class_name": "DQN",
                    "library": "stable_baselines3",
                    "enabled": True,
                    "n_envs": 1,
                    "params": {},
                }
            }
        },
        "envs": {"environments": []},
        "preprocessing": {"preprocessing": {}},
        "contracts": {"contracts": {}},
    }

    def fake_train_and_evaluate(**kwargs):
        checkpoint_dir = kwargs["checkpoint_dir"]
        return {
            "Algorithm": kwargs["algo"],
            "Environment": kwargs["env_name"],
            "EnvID": kwargs["env_id"],
            "Seed": kwargs["seed"],
            "Timesteps": kwargs["timesteps"],
            "EvalEpisodes": kwargs["eval_episodes"],
            "Final_Reward_Mean": 0.0,
            "Final_Reward_Std": 0.0,
            "Final_Reward_Min": 0.0,
            "Final_Reward_Max": 0.0,
            "Training_Seconds": 0.1,
            "Reward_Per_Hour": 0.0,
            "InputShape": "(4, 84, 84)",
            "InputDType": "uint8",
            "InputRange": "[0, 255]",
            "InputLayout": "CHW",
            "InputContiguous": "True",
            "NumEnvs": 1,
            "CheckpointDir": checkpoint_dir,
            "FinalModelPath": str(tmp_path / "final_model.zip"),
        }

    playback_calls = []
    monkeypatch.setattr(app, "settings", fake_settings)
    monkeypatch.setattr(app, "load_all_configs", lambda: fake_configs)
    monkeypatch.setattr(app, "resolve_all_envs", lambda configs: {"Pong": "ALE/Pong-v5"})
    monkeypatch.setattr(app, "train_and_evaluate", fake_train_and_evaluate)
    monkeypatch.setattr(app, "record_playback_video", lambda *args: playback_calls.append(args) or "ok")

    exit_code = app.main([])

    assert exit_code == 0
    assert (tmp_path / "evals" / "checkpoints" / "300k_3seeds").exists()
    assert (tmp_path / "artifacts" / "evaluation" / "playback" / "300k_3seeds").exists()
    assert playback_calls
