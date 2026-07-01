# problem statement

## what pain exists

benchmarking rl algorithms on atari is well-trodden ground, but most published comparisons lack reproducibility. papers report aggregate scores without sharing configs, checkpoints, or diagnostics. researchers and practitioners cannot verify claims or build on prior work without reimplementing from scratch.

## who feels it

- students learning rl who need grounded, inspectable comparisons
- practitioners choosing between value-based (dqn), policy-gradient (ppo), and maximum-entropy (sac) methods
- researchers who want a shared pipeline to extend with new algorithms

## current workaround

most projects either:
- report single-seed runs with no diagnostics (hides variance)
- use only one algorithm family (no cross-paradigm comparison)
- omit playback or qualitative inspection (misses behavior collapse that numbers hide)

## why solve this now

this is a cisc 856 final project with a fixed timeline. atari 2600 environments are a compact, well-understood benchmark that lets us compare three algorithm families (value-based, policy-gradient, maximum-entropy) under a shared, reproducible pipeline within the semester constraints.
