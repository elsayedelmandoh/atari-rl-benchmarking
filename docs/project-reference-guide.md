# Atari RL Benchmarking Reference Guide

This guide maps the project architecture, configuration flow, data pipeline, and current conventions for improving the Atari benchmark code safely.

## Project Structure

```text
atari-rl-benchmarking/
  app.py                         Benchmark runner entry point.
  configs/                       JSON/YAML experiment configuration.
    algorithms.json              DQN, PPO, DiscreteSAC model settings.
    benchmark.json               Run profiles such as 300k_3seeds and 1m_5seeds.
    envs.json                    Atari game list and fallback Gymnasium ids.
    model_contracts.json         Required model input shapes/layouts.
    preprocessing.json           Atari preprocessing and frame-stack settings.
  src/
    config/                      Settings, config loading, logging.
    evaluation/                  Metrics, metadata, input-contract validation.
    inference/                   Inference helpers.
    models/                      Algorithm factories and custom DiscreteSAC.
    utils/                       Environment construction and reproducibility helpers.
  scripts/
    validate_envs.py             Environment availability checks.
    validate_inputs.py           Per-algorithm input contract checks.
    record_playback.py           Record one episode from a saved checkpoint.
  evals/
    checkpoints/<profile>/       Model checkpoints, final models, CSVs, manifests.
    figures/                     Analysis plots.
  artifacts/
    preparing/input_samples/     Saved input samples and frame-stack images.
    evaluation/playback/<profile>/ One-episode MP4 videos grouped by game/algo.
    training/                    Training artifacts reserved by the project layout.
  logs/                          Project log and TensorBoard event files.
  tests/                         Unit tests for settings/app behavior.
  docs/                          Proposal, runbook, and project documentation.
```

The active benchmark outputs are profile-scoped. For example, `1m_5seeds` contains proposal-scale results, while validation runs write to matching folders such as `evals/checkpoints/300k_3seeds/` and `artifacts/evaluation/playback/300k_3seeds/`.

## Configuration System

`src/config/config.py` defines a Pydantic `Settings` singleton named `settings`.

Available top-level settings:

- `BASE_DIR`: repository root, derived from `src/config/config.py`.
- `PROJECT_NAME`: default `atari-rl-benchmarking`.
- `ENV`: default `development`.
- `DATA_DIR`, `ARTIFACTS_DIR`, `LOGS_DIR`, `FIGURES_DIR`, `CONFIGS_DIR`, `PROMPTS_DIR`.
- Data subdirs: `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `TRAIN_DATA_DIR`, `VAL_DATA_DIR`, `TEST_DATA_DIR`, `SAMPLES_DATA_DIR`, `CURATED_DATA_DIR`, `INSTRUCTIONS_DATA_DIR`.
- Artifact subdirs: `PREPARING_DIR`, `TRAINING_DIR`, `EVALUATION_DIR`, `EXPORTS_DIR`.

Settings are computed as absolute `Path` objects. Importing `settings` also creates the required directories.

`src/config/loader.py` loads JSON config files from `settings.CONFIGS_DIR`:

- `load_all_configs()` returns `envs`, `algorithms`, `preprocessing`, `contracts`, and `benchmark`.
- `config_hash()` creates a short SHA256 hash of the full loaded config bundle. Each benchmark result row and manifest records this hash for reproducibility.

## Main Config Files

`configs/benchmark.json`

- Stores named profiles under `profiles`.
- `300k_3seeds` currently means 300,000 timesteps, seed `[0]`, 50 evaluation episodes, checkpoints every 100,000 steps.
- `1m_1seed_ppo_diagnostic` is the PPO-only diagnostic profile: 1,000,000 timesteps, seed `[0]`, 100 evaluation episodes, checkpoints every 100,000 steps. Run it with `--algo ppo`.
- `1m_1seed_StaDiscSac_diagnostic` is the Stable DiscreteSAC diagnostic profile: 1,000,000 timesteps, seed `[0]`, 100 evaluation episodes, checkpoints every 100,000 steps. Run it with `--algo discretesac`.
- `1m_5seeds` is the original proposal-scale profile name, currently configured for 1,000,000 timesteps and seeds `[0, 1, 2]`.
- To change run size, edit the profile or pass CLI overrides:
  - `python app.py 300k_3seeds --timesteps 100000 --seeds 0`
  - `python app.py 1m_1seed_ppo_diagnostic --algo ppo`
  - `python app.py 1m_1seed_StaDiscSac_diagnostic --algo discretesac`

`configs/algorithms.json`

- Defines enabled algorithms and hyperparameters.
- `DQN` and `PPO` use Stable-Baselines3 `CnnPolicy`.
- `DiscreteSAC` uses the custom implementation in `src/models/discrete_sac.py`.
- `PPO` uses `n_envs: 8`; `DQN` and `DiscreteSAC` use single environments.
- String schedule values like `"lin_0.00025"` are converted by `src/models/sb3.py`.

`configs/envs.json`

- Defines benchmark games: Pong, Breakout, Space Invaders.
- Each game has ordered `candidate_ids`; `resolve_all_envs()` picks the first available Gymnasium/ALE environment.

`configs/preprocessing.json`

- Controls Atari input preparation: noop, frame skip, 84x84 resize, grayscale, frame stack, and expected layouts.
- Current model input path expects stacked grayscale observations as unbatched `CHW`.

`configs/model_contracts.json`

- Defines algorithm-specific input contracts.
- `src/evaluation/validation.py` checks exact shape, layout, and contiguity before training.

## Data Pipeline And Paths

There is no offline image dataset for training. Inputs are generated by Atari environments at runtime.

1. `app.py` loads all configs and resolves environment ids.
2. `src/utils/data_acquisition.py` creates raw Gymnasium/ALE envs with `frameskip=1`.
3. `make_atari_env()` applies `AtariPreprocessing` and `FrameStackObservation`.
4. Before each run, `train_and_evaluate()` resets an input env and calls `assert_model_input_contract()`.
5. The trainer creates either a single env or a `DummyVecEnv` based on algorithm `n_envs`.
6. The model trains, checkpoints periodically, saves a final model, evaluates deterministic episodes, then records one playback video.

PPO runs also write `ppo_diagnostics.csv` in each seed checkpoint folder. Use this to inspect rollout action collapse, action entropy, KL, clip fraction, entropy loss, value loss, and explained variance. DiscreteSAC runs write `discrete_sac_diagnostics.csv` with action entropy, alpha, Q statistics, actor/critic losses, and policy probabilities.

Output path convention:

```text
evals/checkpoints/<profile>/
  <profile>_results_<timestamp>.csv
  <profile>_manifest_<timestamp>.json
  <game>/<algo>/seed_<seed>/
    final_model.zip              Stable-Baselines3 DQN/PPO
    final_model.pt               custom DiscreteSAC
    *_steps.zip or *.pt          periodic checkpoints
    ppo_diagnostics.csv          PPO rollout/action/update diagnostics
    discrete_sac_diagnostics.csv DiscreteSAC action/Q/entropy diagnostics

artifacts/evaluation/playback/<profile>/
  <game>/<algo>/*.mp4            one deterministic episode per model/game/seed
```

For `300k_3seeds`, checkpoints and final model files go under `evals/checkpoints/300k_3seeds/`. Playback videos go under `artifacts/evaluation/playback/300k_3seeds/`.

## Key Entry Points

- `python app.py`: runs the default benchmark profile, now `300k_3seeds`.
- `python app.py 300k_3seeds`: explicit 300k, 3-seed validation run.
- `python app.py 1m_1seed_ppo_diagnostic --algo ppo`: PPO-only 1M diagnostic run after the current 300k comparison finishes.
- `python app.py 1m_1seed_StaDiscSac_diagnostic --algo discretesac`: Stable DiscreteSAC-only 1M diagnostic run.
- `python app.py 300k_3seeds --algo dqn --env pong`: focused run for one algorithm/game.
- `python scripts/validate_envs.py`: check Atari environment resolution.
- `python scripts/validate_inputs.py`: check image size, dtype, range, layout, contiguity, and one model prediction for every algorithm/game pair.
- `python scripts/record_playback.py <algo> <checkpoint> <env_id> --output artifacts/evaluation/playback/manual`: record a single checkpoint manually.

## Constraints And Conventions

- Keep experiment changes in `configs/*.json` whenever possible; avoid hardcoding run budgets or hyperparameters inside model code.
- Preserve profile-scoped outputs. Do not mix smoke, 200k, 300k, and 1m artifacts in the same folder.
- Keep input checks explicit. Before training, validate shape, memory layout, contiguity, dtype, and pixel range.
- Treat model inputs as `CHW` unbatched observations and `NCHW` batched tensors unless the config contracts are intentionally updated.
- DQN/PPO should stay behind the Stable-Baselines3 factories in `src/models/dqn.py` and `src/models/ppo.py`.
- DiscreteSAC improvements belong in `src/models/discrete_sac.py` and should retain `save()`, `load()`, and `predict()` compatibility with the runner.
- Add new games only through `configs/envs.json`; let `resolve_env_id()` discover the available ALE id.
- Add new run sizes through `configs/benchmark.json`; the folder name should match the profile name.
- For reproducibility, keep seeds, profile config, result CSV, manifest, config hash, checkpoints, and playback videos together by profile.
- PPO can finish much faster than DQN or DiscreteSAC in wall-clock time because it uses vectorized environments. Compare algorithms by configured environment timesteps, then use `Training_Seconds` and `ppo_diagnostics.csv` to judge whether PPO received enough update quality and policy diversity.
