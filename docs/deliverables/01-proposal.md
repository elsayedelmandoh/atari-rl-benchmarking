### Benchmarking DQN vs PPO on Atari Pong and Breakout

<table border="0" cellspacing="0" cellpadding="4" style="border: none; margin: 0 auto; font-size: 0.70em;">
  <tr style="border: none;">
    <td style="border: none; text-align: center; vertical-align: top;">
      <strong>Elsayed Elmandoh</strong><br>
      School of Computing<br>
      Queen's University<br>
      Kingston, Canada<br>
      elsayed.elmandoua@queensu.ca
    </td>
    <td style="border: none; text-align: center; vertical-align: top;">
      <strong>Mohamed Kamal</strong><br>
      School of Computing<br>
      Queen's University<br>
      Kingston, Canada<br>
      25qbd3@queensu.ca
    </td>
    <td style="border: none; text-align: center; vertical-align: top;">
      <strong>Mostafa Elofy</strong><br>
      School of Computing<br>
      Queen's University<br>
      Kingston, Canada<br>
      25djt3@queensu.ca
    </td>
    <td style="border: none; text-align: center; vertical-align: top;">
      <strong>Mohamed Zidan</strong><br>
      School of Computing<br>
      Queen's University<br>
      Kingston, Canada<br>
      25kbhb@queensu.ca
    </td>
  </tr>
</table>

### I. Project Description

We compare **DQN** (Mnih et al., 2015) and **PPO** (Schulman et al., 2017) on **Pong** (easy baseline) and **Breakout** (harder test) using **pixel-based** environments (`PongNoFrameskip-v4`, `BreakoutNoFrameskip-v4`) the same setting as the original papers. Both algorithms use an identical **Nature CNN** (3 conv + 2 FC) and preprocessing pipeline. Each (algo, env) combination runs with **5 seeds** = **20 total runs** at 1M timesteps each. Evaluation: **100 deterministic episodes**, reporting mean and standard deviation (μ ± σ). **Deliverable** IEEE-format report, learning curves, tables, statistical analysis. Results'd be interpreted as fixed-budget comparison

### II. State Space

Raw observations are Atari RGB frames of shape (210, 160, 3), represented as uint8 pixel values in [0, 255]. The final processed state is a normalized 4-frame stack of shape (4, 84, 84)

| # | Operation | Output |
|---|-----------|--------|
| 1 | Grayscale conversion | (210, 160, 1) |
| 2 | Resize to 84×84 | (84, 84, 1) |
| 3 | Stack 4 frames | (4, 84, 84) |
| 4 | Normalize pixels and clip rewards | scaled pixels; rewards clipped to +/-1 |

### III. Action Space

| Environment | Action Space | Actions |
|---|---|---|
| Pong | Discrete(6) | NOOP, FIRE, RIGHT, LEFT, RIGHT+FIRE, LEFT+FIRE |
| Breakout | Discrete(4) | NOOP, FIRE, LEFT, RIGHT |

**Action repeat** = 4. With Gymnasium ALE v5, use ALE/Pong-v5 and ALE/Breakout-v5 with explicit configuration, use PongNoFrameskip-v4 and BreakoutNoFrameskip-v4 with wrapper-controlled frame skip.

### IV. Proposed Solution

| Component | Choice |
|---|---|
| Framework | Stable Baselines3, PyTorch via TensorBoard, CSV log, 50k-step checkpoints |
| Environments | Gymnasium `PongNoFrameskip-v4`, `BreakoutNoFrameskip-v4` |
| Algorithms | DQN, PPO - Canonical values from each paper and SB3 documentation |
| Policy | Nature CNN (3 conv - 2 FC, same for both) |
| Preprocessing | Grayscale - 84×84 → 4-frame stack - reward clip ±1 via `AtariWrapper` |
| Timesteps | 1,000,000 per run across 5 random seeds per (algo, env) = 20 runs|
| Evaluation | 100 deterministic episodes, learning curves, reward/hour, (μ ± σ), effect size |
| Analysis | Learning curves, bar charts, Tukey HSD, reward/hour, seed-variance |