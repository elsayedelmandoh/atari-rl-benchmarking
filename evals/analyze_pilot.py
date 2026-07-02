"""Generate final benchmark figures from the report evidence profiles.

Inputs:
  - evals/checkpoints/1m_5seeds/*_results_*.csv for DQN
  - evals/checkpoints/1m_1seed_ppo_diagnostic/*_results_*.csv for PPO
  - evals/checkpoints/1m_1seed_StaDiscSac_diagnostic/*_results_*.csv for DiscreteSAC

Outputs:
  - evals/figures/mean_reward.png
  - evals/figures/max_reward.png
  - evals/figures/reward_per_hour.png
  - evals/figures/training_time.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

EVALS_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = EVALS_DIR / "checkpoints"
FIGURES_DIR = EVALS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GAMES = ["Pong", "Breakout", "Space Invaders"]
ALGOS = ["DQN", "PPO", "DiscreteSAC"]
ALGO_COLORS = {"DQN": "#4c72b0", "PPO": "#dd8452", "DiscreteSAC": "#55a868"}

FINAL_SOURCES = {
    "DQN": "1m_5seeds",
    "PPO": "1m_1seed_ppo_diagnostic",
    "DiscreteSAC": "1m_1seed_StaDiscSac_diagnostic",
}


def latest_results_csv(profile: str) -> Path:
    files = sorted((CHECKPOINTS_DIR / profile).glob("*_results_*.csv"))
    if not files:
        raise FileNotFoundError(f"no results CSV found for profile {profile}")
    return files[-1]


def load_final_table() -> pd.DataFrame:
    rows: list[dict] = []
    for algo, profile in FINAL_SOURCES.items():
        df = pd.read_csv(latest_results_csv(profile))
        df = df[df["Algorithm"] == algo].copy()
        if df.empty:
            raise ValueError(f"profile {profile} does not contain rows for {algo}")

        for game in GAMES:
            game_rows = df[df["Environment"] == game]
            if game_rows.empty:
                raise ValueError(f"profile {profile} does not contain {algo}/{game}")

            means = game_rows["Final_Reward_Mean"].astype(float)
            maxes = game_rows["Final_Reward_Max"].astype(float)
            train_secs = game_rows["Training_Seconds"].astype(float)

            # DQN has five seeds in the final evidence profile, so this is a seed-level std.
            # PPO and DiscreteSAC diagnostic profiles have one seed, so we keep their eval-episode std.
            if len(game_rows) > 1:
                spread = float(means.std(ddof=0))
                spread_label = "seed_std"
            else:
                spread = float(game_rows["Final_Reward_Std"].astype(float).iloc[0])
                spread_label = "eval_episode_std"

            train_sec = float(train_secs.mean())
            mean_reward = float(means.mean())
            rows.append(
                {
                    "Algorithm": algo,
                    "Game": game,
                    "Profile": profile,
                    "Runs": int(len(game_rows)),
                    "Mean": mean_reward,
                    "Spread": spread,
                    "SpreadType": spread_label,
                    "Max": float(maxes.max()),
                    "TrainMin": train_sec / 60.0,
                    "RewardPerHour": (mean_reward / train_sec * 3600.0) if train_sec > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def grouped_bar_chart(table: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str, error_bars: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(GAMES))
    width = 0.25

    for algo_idx, algo in enumerate(ALGOS):
        subset = table[table["Algorithm"] == algo]
        vals = []
        errs = []
        for game in GAMES:
            row = subset[subset["Game"] == game].iloc[0]
            vals.append(float(row[metric]))
            errs.append(float(row["Spread"]) if error_bars else 0.0)

        offset = (algo_idx - 1) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=f"{algo} ({FINAL_SOURCES[algo]})",
            color=ALGO_COLORS[algo],
            yerr=errs if error_bars else None,
            capsize=3 if error_bars else 0,
            error_kw={"lw": 1},
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(GAMES)
    ax.axhline(y=0, color="gray", lw=0.6)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    table = load_final_table()
    print("=== final evidence table used for figures ===")
    print(table.to_string(index=False))
    print()

    grouped_bar_chart(
        table,
        metric="Mean",
        title="Final Evidence Mean Evaluation Reward",
        ylabel="Mean Reward",
        filename="mean_reward.png",
        error_bars=True,
    )
    grouped_bar_chart(
        table,
        metric="Max",
        title="Final Evidence Best Evaluation Episode",
        ylabel="Maximum Reward",
        filename="max_reward.png",
        error_bars=False,
    )
    grouped_bar_chart(
        table,
        metric="RewardPerHour",
        title="Final Evidence Reward per Training Hour",
        ylabel="Reward / Hour",
        filename="reward_per_hour.png",
        error_bars=False,
    )
    grouped_bar_chart(
        table,
        metric="TrainMin",
        title="Final Evidence Training Time",
        ylabel="Minutes",
        filename="training_time.png",
        error_bars=False,
    )

    print(f"\ndone. figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
