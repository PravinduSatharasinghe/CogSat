import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("logs/federated_a2c/round_logs.json")
REFRESH_SECONDS = 3

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 11,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

plt.ion()


def moving_average(values, window=5):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    valid = np.convolve(values, kernel, mode="valid")
    pad_left = window // 2
    pad_right = len(values) - len(valid) - pad_left
    left = [np.mean(values[:i + 1]) for i in range(pad_left)]
    right = [np.mean(values[len(values) - (pad_right - i):]) for i in range(pad_right)]
    return np.array(left + list(valid) + right)


def load_logs():
    if not LOG_PATH.exists():
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
        if not isinstance(logs, list):
            return []
        logs = sorted(logs, key=lambda x: x["round_idx"])
        return logs
    except Exception:
        return []


fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Federated A2C Live Training Monitor", fontsize=14)


def update_plot():
    logs = load_logs()
    if not logs:
        return

    rounds = [x["round_idx"] for x in logs]
    global_reward = [x["global_eval_mean_reward"] for x in logs]
    leo_cap = [x["global_eval_avg_leo_capacity"] for x in logs]
    geo_cap = [x["global_eval_avg_geo_capacity"] for x in logs]
    interference = [max(x["global_eval_avg_leo_to_geo_interference"], 1e-30) for x in logs]
    param_delta = [x["avg_param_delta_norm"] for x in logs]
    train_time = [x["avg_client_train_time_sec"] for x in logs]

    reward_s = moving_average(global_reward, 5)
    leo_s = moving_average(leo_cap, 5)
    geo_s = moving_average(geo_cap, 5)
    int_s = moving_average(interference, 5)
    delta_s = moving_average(param_delta, 5)
    time_s = moving_average(train_time, 5)

    for ax in axes.flat:
        ax.clear()

    axes[0, 0].plot(rounds, global_reward, alpha=0.4, label="Raw")
    axes[0, 0].plot(rounds, reward_s, linewidth=2, label="Smoothed")
    axes[0, 0].set_title("Global Reward")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    axes[0, 1].plot(rounds, leo_cap, alpha=0.4, label="LEO")
    axes[0, 1].plot(rounds, leo_s, linewidth=2)
    axes[0, 1].plot(rounds, geo_cap, alpha=0.4, label="GEO")
    axes[0, 1].plot(rounds, geo_s, linewidth=2)
    axes[0, 1].set_title("Capacities")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    axes[1, 0].plot(rounds, interference, alpha=0.4, label="Raw")
    axes[1, 0].plot(rounds, int_s, linewidth=2, label="Smoothed")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("LEO→GEO Interference")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    axes[1, 1].plot(rounds, param_delta, alpha=0.4, label="Raw")
    axes[1, 1].plot(rounds, delta_s, linewidth=2, label="Smoothed")
    axes[1, 1].set_title("Parameter Delta Norm")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    axes[2, 0].plot(rounds, train_time, alpha=0.4, label="Raw")
    axes[2, 0].plot(rounds, time_s, linewidth=2, label="Smoothed")
    axes[2, 0].set_title("Avg Client Train Time")
    axes[2, 0].set_xlabel("Round")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    axes[2, 0].legend()

    axes[2, 1].axis("off")
    last = logs[-1]
    summary = (
        f"Last round: {last['round_idx']}\n"
        f"Selected clients: {last['selected_client_ids']}\n"
        f"Global reward: {last['global_eval_mean_reward']:.6f}\n"
        f"LEO capacity: {last['global_eval_avg_leo_capacity']:.6f}\n"
        f"GEO capacity: {last['global_eval_avg_geo_capacity']:.6f}\n"
        f"Interference: {last['global_eval_avg_leo_to_geo_interference']:.3e}\n"
        f"Param delta norm: {last['avg_param_delta_norm']:.6f}"
    )
    axes[2, 1].text(0.02, 0.98, summary, va="top", ha="left", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    print(f"Watching: {LOG_PATH.resolve()}")
    while plt.fignum_exists(fig.number):
        update_plot()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()