# solution overview

## what we built

an end-to-end reproducible benchmark pipeline that trains, evaluates, and records playback for three rl algorithms on atari 2600 games.

## algorithms

| Algorithm | Implementation | Type | Key features |
|---|---|---|---|
| DQN | Stable Baselines3 `CnnPolicy` | Value-based | Replay buffer, target network, epsilon-greedy |
| PPO | Stable Baselines3 `CnnPolicy` | Policy-gradient | Clipped optimization, GAE, entropy reg, 8 vec envs |
| DiscreteSAC | Custom (categorical policy) | Maximum-entropy | Twin Q networks, target Q, temperature tuning, stochastic policy |

discretesac is custom because sb3 sac targets continuous action spaces. this project implements a categorical policy over the atari action set with min-Q backup, delayed policy updates, and entropy temperature tuning.

## games

pong, breakout, space invaders - chosen for their distinct challenge profiles (tracking, paddle control, aiming + dodging).

## profiles

experiments are organized into named profiles that control timesteps, seeds, eval episodes, and checkpoint frequency:

| Profile | Purpose |
|---|---|
| `100k_1seed` | Early pilot across all algos and games |
| `200k_1seed` | Tuned pilot after stability fixes |
| `300k_1seed_ppo_sac_improved` | PPO + DiscreteSAC comparison |
| `1m_5seeds` | Main DQN multi-seed benchmark |
| `1m_1seed_ppo_diagnostic` | PPO diagnostic (1M steps) |
| `1m_1seed_StaDiscSac_diagnostic` | Stable DiscreteSAC diagnostic (1M steps) |

## key mechanisms

- profile locking prevents overlapping runs
- `--algo` flag for single-algorithm diagnostics
- input contract validation before training (shape, dtype, range, layout)
- per-profile checkpoint folders, result CSVs, manifests
- mp4 playback regeneration from saved checkpoints
- ppo diagnostic csv (entropy, loss, approx kl)
- discretesac diagnostic csv (q-values, clipped gradients, entropy temperature)
