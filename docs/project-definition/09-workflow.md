# development workflow

## phase 1: define (weeks 1-2)
- select games (pong, breakout, space invaders) and algorithms (dqn, ppo, discretesac)
- write proposal with problem, approach, timeline
- get feedback -> add discrete sac as third algorithm
- set up repo structure and config system

## phase 2: build baseline pipeline (weeks 3-4)
- implement environment wrappers and preprocessing pipeline
- integrate sb3 dqn and ppo
- implement custom discretesac with categorical policy
- add input contract validation
- run pilot profiles (100k_1seed, 200k_1seed)

## phase 3: diagnose and fix (weeks 5-6)
- discover collapsed ppo and discretesac playback
- add mechanism: problem-found table grew iteratively
- add mp4 playback recording
- add algorithm filtering (`--algo`)
- add profile locking
- add ppo diagnostic csv
- fix discretesac stability (gradient guards, clipping, recipe update)
- fix discretesac deterministic playback (stochastic sampling)
- fix breakout post-life fire handling

## phase 4: final runs (weeks 7-8)
- run dqn 1m_5seeds (main benchmark)
- run ppo 1m_1seed_ppo_diagnostic
- run discretesac 1m_1seed_StaDiscSac_diagnostic (after recipe fixes)
- record all playback videos
- generate report figures

## phase 5: document (week 9)
- write final report (02-report.md)
- convert to latex (02-report.tex)
- populate project-definition files (01-10)
- commit all evals, artifacts, figures

## iteration pattern

each profile run follows this loop:
1. acquire profile lock (prevent overlap)
2. load configs (algo, env, preproc, profile)
3. for each (game, algo, seed):
   a. validate input contracts
   b. train agent for profile-defined timesteps
   c. evaluate over profile-defined episodes
   d. record diagnostics (action counts, entropy, q-values)
   e. save checkpoint
   f. record playback mp4
4. write result csv and manifest

when playback or metrics look wrong: stop, diagnose, fix, re-run affected profile. the diagnostic profile concept (`1m_1seed_*`) was created specifically for this.
