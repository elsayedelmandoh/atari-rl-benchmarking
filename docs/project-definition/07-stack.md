# stack overview

## languages
- python 3.12

## rl / ml
- **gymnasium** (atari environment wrapper, ale interface)
- **stable-baselines3** (dqn, ppo implementations and cnn policy)
- **pytorch** (neural network backend, tensor ops, gradient computation)

## data / config
- **json** (configs: algorithms, environments, preprocessing, profiles)
- **csv** (results, diagnostics, playback metadata)
- **numpy** (array operations, statistics)

## logging / visualization
- **tensorboard** (training curves, loss plots)
- **matplotlib** + **seaborn** (report figures: mean/max reward, reward/hour, training time)
- **opencv** / **ffmpeg** (mp4 playback recording)

## packaging
- **pyproject.toml** (project metadata, dependencies)
- **uv** (dependency management, lockfile)
- **pytest** (unit tests)
- **ruff** (linting and formatting)

## why each choice

| Component | Choice | Reason |
|---|---|---|
| RL framework | Stable Baselines3 | Mature, well-tested DQN/PPO; project requirement to use existing tools |
| Atari interface | Gymnasium ALE | Standard Atari benchmark wrapper; maintained fork |
| Neural network | PyTorch | Required by SB3; discrete SAC custom implementation |
| Config | JSON | Simple, version-controllable, no extra deps |
| Diagnostic tracking | CSV | Universal, diffable, can be loaded into any analysis tool |
| Figures | Matplotlib/Seaborn | Standard Python plotting; publication-quality output |
