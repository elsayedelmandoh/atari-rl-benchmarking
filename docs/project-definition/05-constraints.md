# constraints and assumptions

## timeline constraints
- fixed semester schedule for cisc 856
- no time for multi-algorithm multi-seed runs across all profiles (ppo and discretesac diagnostics are 1-seed)
- long training runs (1m steps) must complete within days, not weeks

## compute constraints
- consumer-grade gpu (no multi-gpu or distributed training)
- no cloud compute budget - runs on local hardware
- evaluation limited to 100 episodes per seed

## algorithm constraints
- dqn and ppo must use stable baselines3 (project requirement)
- sac must be discrete-action variant (sb3 sac is continuous-only)
- must share preprocessing pipeline across all algorithms

## scope boundaries
- no custom game environments (use gymnasium/ale)
- no novel algorithm development (compare existing methods)
- no massive hyperparameter sweeps
- no multi-agent or multi-task learning
- no distributed or parallel training beyond sb3 subprocvecenv for ppo

## quality constraints
- must validate model inputs before training (shape, dtype, range, contiguity)
- must produce checkpoints, csv results, and manifests per profile
- must support playback regeneration from checkpoints
- must record action distributions for diagnostics
