# Scope Decisions

## Included Games

- Pong
- Breakout
- Space Invaders

## Included Algorithms

- DQN
- PPO
- DiscreteSAC

## DiscreteSAC Decision

Atari games use discrete action spaces. Standard Stable Baselines3 SAC is intended for continuous action spaces, so the project uses a custom `DiscreteSAC` implementation instead of SB3 SAC.

`DiscreteSAC` uses categorical policies over the Atari action set, twin Q networks, target Q networks, entropy temperature tuning, and the same validated Atari image contract as DQN/PPO.

The old standard `SAC` entry is intentionally not used.

## Input Validation Policy

Before training, every environment and model pair must pass checks for:

- raw image shape, dtype, range, and layout,
- preprocessed image shape, dtype, range, and layout,
- final model memory layout,
- batch dimension,
- tensor contiguity,
- and model forward/predict compatibility.
