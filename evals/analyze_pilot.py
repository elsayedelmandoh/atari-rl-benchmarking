"""propose: generate performance tables, bar charts, and stats from pilot csvs.
input: evals/pilot/*.csv and evals/pilot_200k_tuned/*.csv.
output: figures in evals/figures/, summary printed to stdout."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

matplotlib.use("Agg")

EVALS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = EVALS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GAMES = ["Pong", "Breakout", "Space Invaders"]
ALGOS = ["DQN", "PPO", "DiscreteSAC"]
GAME_ORDER = {g: i for i, g in enumerate(GAMES)}
ALGO_COLORS = {"DQN": "#4c72b0", "PPO": "#dd8452", "DiscreteSAC": "#55a868"}
ALGO_MARKERS = {"DQN": "o", "PPO": "s", "DiscreteSAC": "D"}


def load_pilot_data() -> dict[str, pd.DataFrame]:
    csvs = {
        "100k": EVALS_DIR / "results" / "100k_1seed" / "100k_1seed_results_20260622_164552.csv",
        "200k": EVALS_DIR / "results" / "200k_1seed" / "200k_1seed_results_20260623_103011.csv",
    }
    return {k: pd.read_csv(v) for k, v in csvs.items()}


def perf_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for label, df in data.items():
        for algo in ALGOS:
            subset = df[df["Algorithm"] == algo]
            for game in GAMES:
                row = subset[subset["Environment"] == game]
                if row.empty:
                    continue
                r = row.iloc[0]
                rows.append(
                    {
                        "Profile": label,
                        "Algorithm": algo,
                        "Game": game,
                        "Mean": float(r["Final_Reward_Mean"]),
                        "Std": float(r["Final_Reward_Std"]),
                        "Min": float(r["Final_Reward_Min"]),
                        "Max": float(r["Final_Reward_Max"]),
                        "Reward/Hour": float(r["Reward_Per_Hour"]),
                        "TrainSec": float(r["Training_Seconds"]),
                    }
                )
    result = pd.DataFrame(rows)
    result["GameOrder"] = result["Game"].map(GAME_ORDER)
    result["AlgoOrder"] = result["Algorithm"].apply(lambda x: ALGOS.index(x) if x in ALGOS else 99)
    result = result.sort_values(["Profile", "GameOrder", "AlgoOrder"]).reset_index(drop=True)
    for c in ["GameOrder", "AlgoOrder"]:
        del result[c]
    return result


def bar_chart(table: pd.DataFrame, metric: str, title: str, ylabel: str, fname: str) -> None:
    profiles = table["Profile"].unique()
    n_profiles = len(profiles)
    fig, axes = plt.subplots(1, n_profiles, figsize=(5 * n_profiles, 4), sharey=False)
    if n_profiles == 1:
        axes = [axes]

    for idx, profile in enumerate(profiles):
        ax = axes[idx]
        subset = table[table["Profile"] == profile]
        x = np.arange(len(GAMES))
        width = 0.25

        for algo_idx, algo in enumerate(ALGOS):
            algo_data = subset[subset["Algorithm"] == algo]
            vals = []
            errs = []
            for game in GAMES:
                row = algo_data[algo_data["Game"] == game]
                if not row.empty:
                    vals.append(float(row[metric].iloc[0]))
                    errs.append(float(row["Std"].iloc[0]) if metric == "Mean" else 0.0)
                else:
                    vals.append(0.0)
                    errs.append(0.0)

            offset = (algo_idx - 1) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=algo,
                color=ALGO_COLORS[algo],
                yerr=errs if metric == "Mean" else None,
                capsize=3,
                error_kw={"lw": 1},
            )

        ax.set_title(f"{title} ({profile})")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(GAMES, fontsize=9)
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", lw=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def reward_hour_chart(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, profile in enumerate(table["Profile"].unique()):
        ax = axes[idx]
        subset = table[table["Profile"] == profile]
        x = np.arange(len(GAMES))
        width = 0.25

        for algo_idx, algo in enumerate(ALGOS):
            algo_data = subset[subset["Algorithm"] == algo]
            vals = []
            for game in GAMES:
                row = algo_data[algo_data["Game"] == game]
                vals.append(float(row["Reward/Hour"].iloc[0]) if not row.empty else 0.0)
            offset = (algo_idx - 1) * width
            ax.bar(x + offset, vals, width, label=algo, color=ALGO_COLORS[algo])

        ax.set_title(f"Reward/Hour ({profile})")
        ax.set_ylabel("Reward / Hour")
        ax.set_xticks(x)
        ax.set_xticklabels(GAMES, fontsize=9)
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", lw=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / "reward_per_hour.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def train_time_chart(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, profile in enumerate(table["Profile"].unique()):
        ax = axes[idx]
        subset = table[table["Profile"] == profile]
        x = np.arange(len(GAMES))
        width = 0.25

        for algo_idx, algo in enumerate(ALGOS):
            algo_data = subset[subset["Algorithm"] == algo]
            vals = []
            for game in GAMES:
                row = algo_data[algo_data["Game"] == game]
                vals.append(float(row["TrainSec"].iloc[0]) / 60.0 if not row.empty else 0.0)
            offset = (algo_idx - 1) * width
            ax.bar(x + offset, vals, width, label=algo, color=ALGO_COLORS[algo])

        ax.set_title(f"Training Time ({profile})")
        ax.set_ylabel("Minutes")
        ax.set_xticks(x)
        ax.set_xticklabels(GAMES, fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "training_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def pairwise_stats(raw_data: dict[str, pd.DataFrame]) -> None:
    print("\n--- pairwise t-tests (final reward, between algos per game per profile) ---")
    print("note: only 1 seed -- results reflect eval-episode variance, not seed variance.")
    print("tukey hsd not meaningful with 1 seed. use these as directional indicators.\n")

    eval_map = {}
    for profile, df in raw_data.items():
        for _, row in df.iterrows():
            key = (profile, str(row["Algorithm"]), str(row["Environment"]))
            eval_map[key] = int(row["EvalEpisodes"])

    for profile in raw_data:
        df = raw_data[profile]
        print(f"\nprofile: {profile}")
        for game in GAMES:
            game_data = df[df["Environment"] == game]
            print(f"\n  {game}:")
            for i, a1 in enumerate(ALGOS):
                for a2 in ALGOS[i + 1:]:
                    r1 = game_data[game_data["Algorithm"] == a1]
                    r2 = game_data[game_data["Algorithm"] == a2]
                    if r1.empty or r2.empty:
                        print(f"    {a1} vs {a2}: insufficient data")
                        continue
                    m1, s1 = float(r1["Final_Reward_Mean"].iloc[0]), float(r1["Final_Reward_Std"].iloc[0])
                    m2, s2 = float(r2["Final_Reward_Mean"].iloc[0]), float(r2["Final_Reward_Std"].iloc[0])
                    n_episodes = eval_map.get((profile, a1, game), 20)
                    t_stat, p_val = scipy_stats.ttest_ind_from_stats(
                        m1, s1, n_episodes, m2, s2, n_episodes, equal_var=False
                    )
                    sig = " ***" if p_val < 0.001 else " **" if p_val < 0.01 else " *" if p_val < 0.05 else ""
                    print(f"    {a1} ({m1:.1f}) vs {a2} ({m2:.1f}): t={t_stat:.3f}, p={p_val:.4f}{sig}")


def main() -> None:
    data = load_pilot_data()
    table = perf_table(data)

    print("=== performance table ===")
    print(table.to_string(index=False))
    print()

    # bar charts
    bar_chart(table, "Mean", "Mean Eval Reward", "Mean Reward", "mean_reward.png")
    bar_chart(table, "Max", "Max Eval Reward", "Max Reward", "max_reward.png")
    reward_hour_chart(table)
    train_time_chart(table)

    # pairwise stats
    pairwise_stats(data)

    print("\ndone. figures in", FIGURES_DIR)


if __name__ == "__main__":
    main()
