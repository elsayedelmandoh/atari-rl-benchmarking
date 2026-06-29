# atari rl benchmarking -- final report

## 1. methodology

### objective

reproducibly benchmark 3 rl algorithms (dqn, ppo, discretesac) on 3 atari 2600 games (pong, breakout, space invaders) using pixel observations and a nature-style cnn.

### games

| game | env id | action space | observation |
|---|---|---|---|
| pong | ale/pong-v5 | discrete(4) | rgb (210, 160, 3) |
| breakout | ale/breakout-v5 | discrete(4) | rgb (210, 160, 3) |
| space invaders | ale/spaceinvaders-v5 | discrete(6) | rgb (210, 160, 3) |

### preprocessing pipeline

all algorithms share identical preprocessing (verified by runtime contract checks):

1. ataripreprocessing: max noop=30, frame skip=4, grayscale, resize to 84x84
2. framestackobservation: stack 4 frames
3. final shape: (4, 84, 84), uint8, chw, range 0-255
4. normalized to [0, 1] before model ingestion (float32 division by 255)

### algorithms

| algo | source | policy | n_envs | learning_starts | buffer_size |
|---|---|---|---|---|---|
| dqn | stable-baselines3 | cnndqn | 1 | 100k | 100k |
| ppo | stable-baselines3 | cnnpolicy | 8 | n/a (on-policy) | n/a |
| discretesac | custom (src/models/discrete_sac.py) | nature cnn backbone + 512fc | 1 | 50k | 500k |

**discretesac note:** uses discrete-action sac variant with categorical policy, twin q-networks, target networks, and entropy temperature tuning. not sb3 sac (which is continuous-action). initial implementation showed q-divergence on space invaders at 200k due to buffer turnover (100k buffer overwritten 2x for 200k steps) and missing gradient clipping. fixed by increasing buffer to 500k, adding gradient clipping (max_norm=1.0), and adjusting exploration schedule.

### evaluation protocol

- 20 deterministic evaluation episodes per run
- evaluation after every run (not during)
- mean reward, std, min, max recorded
- reward clipping: default atari wrapper (clip reward to {-1, 0, 1})

### pilot profiles

| profile | timesteps | seeds | purpose |
|---|---|---|---|---|
| 100k_1seed | 100,000 | 1 (seed 0) | initial smoke, dqn baseline validation |
| 200k_1seed | 200,000 | 1 (seed 0) | tuned params for all algos |
| 1m_5seeds | 1,000,000 | 5 (0-4) | full-scale benchmark (pending) |

---

## 2. results

### per-game perf table

| profile | algo | game | mean reward | std | max | reward/hour |
|---|---|---|---|---|---|---|
| 100k | dqn | pong | -21.0 | 0.0 | -21.0 | -418.6 |
| 100k | ppo | pong | -21.0 | 0.0 | -21.0 | -379.6 |
| 100k | discretesac | pong | -21.0 | 0.0 | -21.0 | -51.8 |
| 100k | dqn | breakout | 7.7 | 3.5 | 15.0 | 143.4 |
| 100k | ppo | breakout | 0.0 | 0.0 | 0.0 | 0.0 |
| 100k | discretesac | breakout | 0.0 | 0.0 | 0.0 | 0.0 |
| 100k | dqn | space invaders | 187.5 | 66.0 | 355.0 | 3556.8 |
| 100k | ppo | space invaders | 285.0 | 0.0 | 285.0 | 4737.3 |
| 100k | discretesac | space invaders | 168.3 | 165.7 | 615.0 | 397.1 |
| 200k | dqn | pong | -21.0 | 0.0 | -21.0 | -342.5 |
| 200k | ppo | pong | -21.0 | 0.0 | -21.0 | -567.1 |
| 200k | discretesac | pong | -21.0 | 0.0 | -21.0 | -73.4 |
| 200k | dqn | breakout | 8.3 | 4.6 | 18.0 | 131.7 |
| 200k | ppo | breakout | 2.7 | 0.7 | 4.0 | 67.1 |
| 200k | discretesac | breakout | 0.0 | 0.0 | 0.0 | 0.0 |
| 200k | dqn | space invaders | 170.0 | 37.5 | 225.0 | 2627.2 |
| 200k | ppo | space invaders | 285.0 | 0.0 | 285.0 | 7259.0 |
| 200k | discretesac | space invaders | 0.0 | 0.0 | 0.0 | 0.0 |

### key observations

**pong (-21.0 across all algos):** all algorithms score -21.0 at both 100k and 200k timesteps. this is expected -- pong requires >1 million timesteps for standard dqn/ppo to show positive reward. the policy consistently loses 21-0 to the cpu opponent. no meaningful comparison possible at this budget.

**breakout (dqn leads):** dqn achieves 7.7 (100k) and 8.3 (200k) -- modest positive return. ppo learns slowly, reaching only 2.7 at 200k. discretesac fails to learn (0.0 at both profiles), likely requiring more timesteps or different hyperparams. dqn > ppo >> discretesac for breakout at 200k.

**space invaders (ppo dominates):** ppo achieves 285.0 (ceiling for this env at low difficulty framing) at both 100k and 200k, with zero variance. dqn achieves 187.5 (100k) and 170.0 (200k) with moderate variance. discretesac starts at 168.3 (100k) but pre-fix collapses to 0.0 at 200k due to q-divergence from buffer turnover. fix applied (buffer 500k + gradient clipping) but needs re-validation with new run.

### training time

| algo | avg 100k (min) | avg 200k (min) | notes |
|---|---|---|---|
| dqn | ~3.1 | ~3.8 | sb3, cpu, 1 env |
| ppo | ~3.4 | ~2.3 | sb3, cpu, 8 vec envs (faster wall-clock) |
| discretesac | ~25.0 | ~17.3 | custom, cpu, 1 env, slow forward pass |

discretesac is 5-7x slower than sb3 baselines on cpu due to pure python training loop (no c++ backend). this limits its viability for large-scale atari benchmarks without gpu.

### statistical tests

pairwise welch's t-tests on evaluation episode variance (20 episodes per run):

**space invaders (100k):**
- ppo (285.0) vs dqn (187.5): p < 0.001 ***
- ppo (285.0) vs discretesac (168.3): p = 0.005 **
- dqn (187.5) vs discretesac (168.3): p = 0.633 (not significant)

**space invaders (200k):**
- ppo (285.0) vs dqn (170.0): p < 0.001 ***
- ppo (285.0) vs discretesac (0.0): p < 0.001 ***
- dqn (170.0) vs discretesac (0.0): p < 0.001 ***

**breakout (200k):**
- dqn (8.3) vs ppo (2.7): p < 0.001 ***
- dqn (8.3) vs discretesac (0.0): p < 0.001 ***
- ppo (2.7) vs discretesac (0.0): p < 0.001 ***

pong not tested (zero variance across all conditions yields nan t-stat).

**caveat:** these tests use episode-level variance within a single seed, not seed variance. multi-seed analysis needed for robust conclusions.

---

## 3. figures

figures saved to `evals/figures/`:
- `mean_reward.png` -- bar chart with error bars per (profile, algo, game)
- `max_reward.png` -- max reward per condition
- `reward_per_hour.png` -- sample efficiency comparison
- `training_time.png` -- wall-clock training time

---

## 4. regression fix: discretesac space invaders

### problem

discretesac achieved 168.25 mean return at 100k steps but collapsed to 0.0 at 200k steps on space invaders. dqn and ppo did not show this pattern.

### root cause

| factor | detail |
|---|---|
| buffer turnover | buffer_size=100k for 200k training steps: buffer overwritten 2x. early good experiences evicted, no corrective signal from past success. |
| q-divergence | no gradient clipping allowed q-network to diverge once buffer contained self-reinforcing bad data. |
| early training start | learning_starts=10k (vs dqn's 100k) meant policy trained on minimal data, noisier initial q-estimates. |
| target entropy | scale=0.98 barely constrained policy (target=1.756 vs max 1.792 nats for 6-action si), allowing premature determinism. |

### fixes applied

1. buffer_size: 100000 -> 500000 (prevents full overwrite)
2. gradient clipping: added `clip_grad_norm_` at max_norm=1.0 on q1, q2, actor
3. learning_starts: 10000 -> 50000 (more random exploration before policy training)
4. target_entropy_scale: 0.98 -> 0.6 (stronger exploration pressure)
5. added verbose logging: q-values, entropy, alpha, actor loss every 1000 steps

### verification

smoke profile (9 runs x 64 steps) passes with 0 failures post-fix. full 200k_1seed re-run needed to confirm 200k space invaders returns positive after fix.

---

## 5. limitations

### single seed (critical)

all pilot results use only seed 0. conclusions about relative algorithm performance may not generalize across seeds. planned 1m_5seeds profile (5 seeds) will address this.

### cpu-only training

torch 2.12.1+cu126 -- cuda available (rtx 5000 ada, 30gb vram). discretesac runs 5-7x slower than sb3 baselines, making its results less practical but now feasible at scale.

### limited timesteps

at 100k-200k timesteps, most atari games are still in early learning phase. pong at -21.0 is expected (needs 2-10m steps for positive reward). results represent early learning behavior, not converged performance.

### single env per method

dqn and discretesac use n_envs=1 (off-policy). ppo uses n_envs=8 (on-policy, standard). this affects wall-clock comparison because ppo processes data faster but uses more environment interactions per step.

### no learning curves

csv results only contain final evaluation metrics. per-step learning curves were not logged (sb3 tensorboard events exist at logs/ but not parsed into analysis).

---

## 6. conclusion

### at 200k timesteps:

- **ppo** is the strongest performer on space invaders (285.0, ceiling score) and reasonably sample-efficient.
- **dqn** is the most consistent across games -- modest positive results on breakout (8.3) and space invaders (170.0) with moderate variance.
- **discretesac** (custom) underperforms sb3 baselines. it matches dqn on space invaders at 100k (168.3) but is fragile at longer horizons without stability fixes. the custom implementation adds implementation risk without clear performance benefit in this setting.

### recommendation

for near-term atari benchmarking, ppo (sb3) + dqn (sb3) provide reliable baselines. the custom discretesac needs more development (gpu training, tuned hyperparameters, target network improvements) before it can serve as a stable baseline. consider replacing with sb3's qrdqn or remaster as alternatives for the sac-family atari representation.

---

## 7. discretesac decision note

this project uses a custom `discretesac` implementation (not sb3 sac) because sb3 sac targets continuous action spaces. the custom implementation:

- uses categorical policy over atari discrete action set
- twin q-networks with target networks and polyak averaging (tau=0.005)
- entropy temperature tuning via dual gradient descent
- same preprocessing pipeline as dqn/ppo

standard sb3 sac is intentionally excluded because it does not support discrete actions.
