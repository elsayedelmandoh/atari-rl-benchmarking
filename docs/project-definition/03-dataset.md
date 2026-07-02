# dataset scope and source

## environments

the benchmark uses gymnasium with the arcade learning environment (ale):

| Game | Resolved environment | Action count | Challenge |
|---|---|---|---|
| Pong | `ALE/Pong-v5` | 6 | track ball, move paddle reactively |
| Breakout | `ALE/Breakout-v5` | 4 | keep ball alive, break bricks |
| Space Invaders | `ALE/SpaceInvaders-v5` | 6 | shoot enemies, dodge bullets |

## observation space

raw state: `s_raw in {0..255}^(210 x 160 x 3)` (rgb frame)

preprocessing pipeline (shared across all algorithms):
1. no-op reset with `noop_max = 30`
2. frame skip / action repeat with `frame_skip = 4`
3. grayscale conversion
4. resize to `84 x 84`
5. stack 4 most recent processed frames

final model input: `s in {0..255}^(4 x 84 x 84)` (channel-first, uint8)
models normalize to `[0, 1]` internally.

## action space

`a in {0, 1, ..., n-1}` where n = 4 (breakout) or 6 (pong, space invaders).
actions correspond to combos like NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE.

## reward

environment reward = score delta from ale: `r_t = R(s_t, a_t, s_{t+1})`
training clips reward to `[-1, 1]` where configured.
evaluation reports accumulated game reward across full episodes.
no custom shaping reward added.

## dataset size and shape

there is no static dataset. the agent generates experience online through environment interaction. each profile defines total timesteps (100k to 1m), seeds (1 or 5), and evaluation episodes (typically 100).
