# Benchmark Runbook

## Where To Change Run Size

Edit [configs/benchmark.json](../configs/benchmark.json).

Important profiles:

- `tuned_200k_1seed`
  - `timesteps`: `200000`
  - `seeds`: `[0]`
  - `evaluation_episodes`: `20`
  - `checkpoint_freq`: `100000`
  - `output_subdir`: `pilot_200k_tuned`
- `pilot_100k_1seed`
  - `timesteps`: `100000`
  - `seeds`: `[0]`
  - `evaluation_episodes`: `20`
  - `checkpoint_freq`: `50000`
- `full_1m_5seed`
  - `timesteps`: `1000000`
  - `seeds`: `[0, 1, 2, 3, 4]`
  - `evaluation_episodes`: `100`
  - `checkpoint_freq`: `50000`

Run the current pilot:

```powershell
python scripts/run_benchmark.py --profile pilot_100k_1seed
```

Run the tuned 200k pilot:

```powershell
python scripts/run_benchmark.py --profile tuned_200k_1seed
```

Run the later full benchmark:

```powershell
python scripts/run_benchmark.py --profile full_1m_5seed
```

## Where To Change Algorithm Hyperparameters

Edit [configs/algorithms.json](../configs/algorithms.json).

Sections:

- `DQN.params`
- `PPO.params`
- `DiscreteSAC.params`

These values are copied into each run manifest so a result CSV can be tied back to the exact config.

## Output Layout

- Smoke results: `results/smoke/`
- 100k pilot results: `results/pilot/`
- Full benchmark results: `results/benchmark/`
- Checkpoints: `checkpoints/<profile-output-subdir>/<timestamp>/<algorithm>/<environment>/seed_<seed>/`

Each benchmark run writes:

- a CSV with metrics and input-contract metadata,
- a JSON manifest with config hash, runtime, device, profile, seeds, and full config snapshot,
- model checkpoints at the configured `checkpoint_freq`,
- final model checkpoint after training.

## Watch A Trained Agent Play

Use [scripts/watch_agent.py](../scripts/watch_agent.py). By default it loads the latest pilot final checkpoint for the selected algorithm/game.

Live window:

```powershell
python scripts/watch_agent.py --algorithm DQN --game Breakout --render-mode human --episodes 1
python scripts/watch_agent.py --algorithm PPO --game "Space Invaders" --render-mode human --episodes 1
python scripts/watch_agent.py --algorithm DiscreteSAC --game Pong --render-mode human --episodes 1
```

Save a video instead of opening a live window:

```powershell
python scripts/watch_agent.py --algorithm DQN --game Breakout --render-mode rgb_array --episodes 1 --save-video
```

Videos are written to:

```text
artifacts/playback/videos/
```

Record one episode for every algorithm/game pair:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/record_all_playback_videos.ps1
```

Record one episode for every algorithm/game pair with a filename tag:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/record_all_playback_videos.ps1 -Tag 200k
```

To load a specific checkpoint:

```powershell
python scripts/watch_agent.py --algorithm DQN --game Breakout --checkpoint checkpoints/pilot/20260622_164552/dqn/breakout/seed_0/final_model.zip --render-mode human
```

## Required Preflight

Before benchmark runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all_phase_gates.ps1
```

This now requires CUDA to pass before moving on.
