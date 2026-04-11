import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PLOT CONFIGURATION (IEEE Paper Style)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (7, 5),
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

OUTPUT_DIR = Path("paper_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_DATA_PATH = Path("logs/federated_a2c/round_logs.json")


# ==========================================
# HELPERS
# ==========================================
def load_round_logs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find round log file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    if not isinstance(logs, list) or len(logs) == 0:
        raise ValueError("round_logs.json must be a non-empty list of round records.")

    # Ensure sorted by round index
    logs = sorted(logs, key=lambda x: x["round_idx"])
    return logs


def moving_average(values, window=5):
    """
    Simple moving average with edge-preserving behavior.
    Returns an array of the same length as input.
    """
    values = np.asarray(values, dtype=float)

    if window <= 1 or len(values) < window:
        return values.copy()

    kernel = np.ones(window) / window
    valid = np.convolve(values, kernel, mode="valid")

    pad_left = window // 2
    pad_right = len(values) - len(valid) - pad_left

    left_vals = [np.mean(values[:i + 1]) for i in range(pad_left)]
    right_vals = [np.mean(values[len(values) - (pad_right - i):]) for i in range(pad_right)]

    return np.array(left_vals + list(valid) + right_vals)


def extract_series(logs):
    rounds = [entry["round_idx"] for entry in logs]

    series = {
        "rounds": rounds,
        "global_reward": [entry["global_eval_mean_reward"] for entry in logs],
        "global_leo_capacity": [entry["global_eval_avg_leo_capacity"] for entry in logs],
        "global_geo_capacity": [entry["global_eval_avg_geo_capacity"] for entry in logs],
        "global_interference": [entry["global_eval_avg_leo_to_geo_interference"] for entry in logs],
        "avg_client_reward": [entry["avg_client_reward"] for entry in logs],
        "avg_client_leo_capacity": [entry["avg_client_leo_capacity"] for entry in logs],
        "avg_client_geo_capacity": [entry["avg_client_geo_capacity"] for entry in logs],
        "avg_client_interference": [entry["avg_client_leo_to_geo_interference"] for entry in logs],
        "avg_param_delta_norm": [entry["avg_param_delta_norm"] for entry in logs],
        "avg_client_train_time_sec": [entry["avg_client_train_time_sec"] for entry in logs],
        "num_selected_clients": [entry["num_selected_clients"] for entry in logs],
    }

    return series


# ==========================================
# PLOTTING FUNCTIONS
# ==========================================
def plot_global_reward(series, smooth_window=5):
    rounds = series["rounds"]
    reward = np.array(series["global_reward"], dtype=float)
    reward_smooth = moving_average(reward, window=smooth_window)

    plt.figure()
    plt.plot(rounds, reward, label="Raw Global Mean Reward", linewidth=1.5, alpha=0.45)
    plt.plot(rounds, reward_smooth, label=f"Smoothed (MA={smooth_window})", linewidth=2.2)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Global Mean Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.title("Global Reward over FL Rounds")

    plt.savefig(OUTPUT_DIR / "fig_global_reward_real.png")
    plt.savefig(OUTPUT_DIR / "fig_global_reward_real.pdf")
    plt.close()


def plot_interference_convergence(series, smooth_window=5):
    rounds = series["rounds"]
    interference = np.array(series["global_interference"], dtype=float)
    interference_smooth = moving_average(interference, window=smooth_window)

    # Avoid log-scale issues if any zeros appear
    eps = 1e-30
    interference_safe = np.maximum(interference, eps)
    interference_smooth_safe = np.maximum(interference_smooth, eps)

    plt.figure()
    plt.plot(rounds, interference_safe, label="Raw Global LEO-to-GEO Interference", linewidth=1.5, alpha=0.45)
    plt.plot(rounds, interference_smooth_safe, label=f"Smoothed (MA={smooth_window})", linewidth=2.2)

    plt.yscale("log")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Average Interference (Watts) [Log Scale]")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.title("Convergence of LEO-to-GEO Interference")

    plt.savefig(OUTPUT_DIR / "fig_interference_convergence_real.png")
    plt.savefig(OUTPUT_DIR / "fig_interference_convergence_real.pdf")
    plt.close()


def plot_capacities(series, smooth_window=5):
    rounds = series["rounds"]

    leo_cap = np.array(series["global_leo_capacity"], dtype=float)
    geo_cap = np.array(series["global_geo_capacity"], dtype=float)

    leo_cap_smooth = moving_average(leo_cap, window=smooth_window)
    geo_cap_smooth = moving_average(geo_cap, window=smooth_window)

    plt.figure()
    plt.plot(rounds, leo_cap, label="Raw Global LEO Capacity", linewidth=1.3, alpha=0.35)
    plt.plot(rounds, geo_cap, label="Raw Global GEO Capacity", linewidth=1.3, alpha=0.35)

    plt.plot(rounds, leo_cap_smooth, label=f"LEO Capacity Smoothed (MA={smooth_window})", linewidth=2.2)
    plt.plot(rounds, geo_cap_smooth, label=f"GEO Capacity Smoothed (MA={smooth_window})", linewidth=2.2)

    plt.xlabel("Communication Rounds")
    plt.ylabel("System Capacity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.title("Global Capacity Evolution over FL Rounds")

    plt.savefig(OUTPUT_DIR / "fig_capacities_real.png")
    plt.savefig(OUTPUT_DIR / "fig_capacities_real.pdf")
    plt.close()


def plot_training_stability(series, smooth_window=5):
    rounds = series["rounds"]
    param_delta = np.array(series["avg_param_delta_norm"], dtype=float)
    param_delta_smooth = moving_average(param_delta, window=smooth_window)

    plt.figure()
    plt.plot(rounds, param_delta, label="Raw Avg Parameter Delta Norm", linewidth=1.5, alpha=0.45)
    plt.plot(rounds, param_delta_smooth, label=f"Smoothed (MA={smooth_window})", linewidth=2.2)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Avg Parameter Delta Norm")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.title("Training Stability across FL Rounds")

    plt.savefig(OUTPUT_DIR / "fig_training_stability_real.png")
    plt.savefig(OUTPUT_DIR / "fig_training_stability_real.pdf")
    plt.close()


def plot_client_vs_global_reward(series, smooth_window=5):
    rounds = series["rounds"]

    client_reward = np.array(series["avg_client_reward"], dtype=float)
    global_reward = np.array(series["global_reward"], dtype=float)

    client_reward_smooth = moving_average(client_reward, window=smooth_window)
    global_reward_smooth = moving_average(global_reward, window=smooth_window)

    plt.figure()
    plt.plot(rounds, client_reward_smooth, label="Avg Client Reward", linewidth=2.2)
    plt.plot(rounds, global_reward_smooth, label="Global Eval Reward", linewidth=2.2)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.title("Client vs Global Reward")

    plt.savefig(OUTPUT_DIR / "fig_client_vs_global_reward_real.png")
    plt.savefig(OUTPUT_DIR / "fig_client_vs_global_reward_real.pdf")
    plt.close()


def print_summary(series):
    reward = np.array(series["global_reward"], dtype=float)
    leo_cap = np.array(series["global_leo_capacity"], dtype=float)
    geo_cap = np.array(series["global_geo_capacity"], dtype=float)
    interf = np.array(series["global_interference"], dtype=float)
    param_delta = np.array(series["avg_param_delta_norm"], dtype=float)

    print("\n===== TRAINING SUMMARY =====")
    print(f"Rounds: {len(series['rounds'])}")
    print(f"Global Reward   | start={reward[0]:.6f}, end={reward[-1]:.6f}, best={reward.max():.6f}")
    print(f"LEO Capacity    | start={leo_cap[0]:.6f}, end={leo_cap[-1]:.6f}, best={leo_cap.max():.6f}")
    print(f"GEO Capacity    | start={geo_cap[0]:.6f}, end={geo_cap[-1]:.6f}, best={geo_cap.max():.6f}")
    print(f"Interference    | start={interf[0]:.6e}, end={interf[-1]:.6e}, min={interf.min():.6e}")
    print(f"Param Delta Norm| start={param_delta[0]:.6f}, end={param_delta[-1]:.6f}, min={param_delta.min():.6f}")
    print("============================\n")


def main():
    logs = load_round_logs(REAL_DATA_PATH)
    series = extract_series(logs)

    print(f"Loaded {len(logs)} rounds from: {REAL_DATA_PATH.resolve()}")

    plot_global_reward(series, smooth_window=5)
    plot_interference_convergence(series, smooth_window=5)
    plot_capacities(series, smooth_window=5)
    plot_training_stability(series, smooth_window=5)
    plot_client_vs_global_reward(series, smooth_window=5)

    print_summary(series)

    print(f"Real-data plots saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()