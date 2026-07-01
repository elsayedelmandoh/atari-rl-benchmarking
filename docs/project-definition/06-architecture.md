# system architecture

## high-level flow

```
configs/*.json  -->  app.py (runner)  -->  training  -->  evaluation  -->  playback recording
                        |                    |              |
                        v                    v              v
                    profile lock         checkpoints     csv results
                    --algo filter        diagnostics     manifests
                    --env filter         tensorboard     mp4 videos
```

## major components

### entrypoint: app.py
- reads benchmark profiles from `configs/benchmark.json`
- acquires profile lock to prevent overlapping runs
- filters by `--algo` and `--env` flags
- orchestrates: training -> evaluation -> playback recording -> csv/manifest writing

### configuration layer: src/config/
- `settings.py`: path resolution, logging setup
- config loading and hashing for provenance tracking

### environment layer: src/utils/
- atari environment construction with shared preprocessing
- wrapper stack: noop reset, frame skip, grayscale, resize, frame stack
- eval env overrides (post-life fire for breakout)
- seed/device helpers

### model layer: src/models/
- **dqn**: sb3 dqn with cnn policy
- **ppo**: sb3 ppo with cnn policy and subprocvecenv
- **discretesac**: custom categorical-policy sac with:
  - twin q networks + target q networks
  - entropy temperature tuning (learned alpha)
  - min-q backup for target computation
  - delayed policy updates (2:1 ratio)
  - finite-gradient guards and gradient clipping
  - replay buffer with prioritized or uniform sampling

### evaluation layer: src/evaluation/
- input contract validation (shape, dtype, range, layout, contiguity, batch, prediction)
- metadata collection and metrics computation
- diagnostic csv generation

### inference layer: src/inference/
- playback regeneration from saved checkpoints via `python -m src.inference.record_playback <profile> <env>`
- supports stochastic policy sampling for discretesac

## output structure

```
evals/checkpoints/<profile>/
  <profile>_results_<timestamp>.csv
  <profile>_manifest_<timestamp>.json
  <game>/<algo>/seed_<seed>/
    final_model.zip       (dqn/ppo)
    final_model.pt        (discretesac)
    *_steps.zip or *.pt
    ppo_diagnostics.csv
    discrete_sac_diagnostics.csv

artifacts/evaluation/playback/<profile>/
  <game>/<algo>/*.mp4

evals/figures/
  mean_reward.png, max_reward.png, reward_per_hour.png, training_time.png

logs/
  tensorboard event files and process logs
```
