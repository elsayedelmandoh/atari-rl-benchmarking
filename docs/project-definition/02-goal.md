# goal and success criteria

## measurable outcome

build a reproducible atari rl benchmarking pipeline that compares dqn, ppo, and discretesac across three games (pong, breakout, space invaders) with:
- shared preprocessing and environment configuration
- profile-based experiment tracking (timesteps, seeds, eval episodes)
- checkpointed models, csv results, manifests, diagnostics
- mp4 playback videos for qualitative inspection
- tensorboard logs

## success criteria

1. **reproducibility** - any run can be re-executed from config alone (profile name + algo + env)
2. **multi-algorithm** - at least 3 algorithms compared (dqn, ppo, discretesac)
3. **multi-game** - results across 3 atari games with different challenge profiles
4. **diagnostics** - per-algorithm diagnostic csvs (ppo entropy, discretesac q-values, action counts)
5. **qualitative evidence** - playback videos that let reviewers see agent behavior, not just reward tables
6. **honest reporting** - failures documented alongside successes (collapsed ppo, weak space invaders)
7. **profile isolation** - profile locking to prevent overlapping runs
8. **algorithm filtering** - ability to run single-algo diagnostics (`--algo ppo`)

## how we will know it worked

- result csvs with mean/std/max reward, wall-clock time, playback paths
- manifests with full provenance (config hashes, timestamps, env specs)
- mp4 files viewable per game/algo/seed
- diagnostic csvs with action distributions, entropy, q-statistics
- this report itself, which documents all findings honestly
