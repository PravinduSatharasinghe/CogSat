# results.py
"""
Generate all result plots from federated training logs and trained models.

Usage:
    python results.py                          # saves to plots/run_<timestamp>/
    python results.py --output-dir my_plots    # saves to my_plots/
    python results.py --no-model-eval          # skip Figure 8 (no model loading)

Expects logs in:
    logs/FedAvg/round_logs.json
    logs/FedProx/round_logs.json
    logs/FedNova/round_logs.json
    logs/FedAvg_DRL/round_logs.json
    logs/FedProx_DRL/round_logs.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

LOG_DIRS: Dict[str, Path] = {
    "FedAvg":      Path("logs/FedAvg"),
    "FedProx":     Path("logs/FedProx"),
    "FedNova":     Path("logs/FedNova"),
    "FedAvg_DRL":  Path("logs/FedAvg_DRL"),
    "FedProx_DRL": Path("logs/FedProx_DRL"),
}

COLORS: Dict[str, str] = {
    "FedAvg":      "tab:blue",
    "FedProx":     "tab:orange",
    "FedNova":     "tab:green",
    "FedAvg_DRL":  "tab:red",
    "FedProx_DRL": "tab:purple",
}

LINESTYLES: Dict[str, str] = {
    "FedAvg":      "-",
    "FedProx":     "--",
    "FedNova":     "-.",
    "FedAvg_DRL":  "-",
    "FedProx_DRL": "--",
}

NUM_CLIENTS        = 20
SMOOTH_WINDOW      = 10    # moving-average window for convergence plots
IMPROVEMENT_WINDOW = 10    # rounds used for baseline / final averages
PEAK_THRESHOLD     = 0.90  # fraction of peak for "rounds to X%" metric
LAST_N_ROUNDS      = 20    # window for box-plot distributions

plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman"],
    "axes.labelsize": 11,
    "font.size":      10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":     150,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_output_dir(cli_arg: str | None) -> Path:
    if cli_arg:
        return Path(cli_arg)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("result_plots") / f"run_{timestamp}"


def load_logs() -> Dict[str, List[dict]]:
    all_logs: Dict[str, List[dict]] = {}
    for name, log_dir in LOG_DIRS.items():
        path = log_dir / "round_logs.json"
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            all_logs[name] = sorted(data, key=lambda x: x["round_idx"])
            print(f"  [ok]   {name}: {len(data)} rounds")
        else:
            print(f"  [skip] {name}: empty log")
    return all_logs


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window or window <= 1:
        return arr
    kernel = np.ones(window) / window
    valid  = np.convolve(arr, kernel, mode="valid")
    pad_l  = window // 2
    pad_r  = len(arr) - len(valid) - pad_l
    left   = [np.mean(arr[:i + 1]) for i in range(pad_l)]
    right  = [np.mean(arr[len(arr) - (pad_r - i):]) for i in range(pad_r)] if pad_r > 0 else []
    return np.array(left + list(valid) + right)


def extract(logs: List[dict], key: str) -> List[float]:
    return [float(r[key]) for r in logs if key in r]


def rounds_to_pct_of_peak(values: List[float], pct: float = PEAK_THRESHOLD) -> int:
    sm     = moving_average(values, SMOOTH_WINDOW)
    target = pct * float(np.max(sm))
    for i, v in enumerate(sm):
        if v >= target:
            return i + 1
    return len(values)


def pct_change(values: List[float], higher_is_better: bool = True) -> float:
    w = IMPROVEMENT_WINDOW
    if len(values) < 2 * w:
        return 0.0
    baseline = float(np.mean(values[:w]))
    final    = float(np.mean(values[-w:]))
    if abs(baseline) < 1e-12:
        return 0.0
    raw = (final - baseline) / abs(baseline) * 100.0
    return raw if higher_is_better else -raw


def jains_fairness(counts: np.ndarray) -> float:
    n = len(counts)
    s = float(np.sum(counts))
    if n == 0 or s == 0:
        return 0.0
    return s ** 2 / (n * float(np.sum(counts ** 2)))


def save_fig(fig: plt.Figure, name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"  → {path}")
    plt.close(fig)


# ── Figure 1: Convergence Curves ─────────────────────────────────────────────

def plot_convergence(all_logs: Dict[str, List[dict]], output_dir: Path):
    metrics = [
        ("global_eval_mean_reward",                "Global Reward"),
        ("global_eval_avg_leo_capacity",            "LEO Capacity (bps/Hz)"),
        ("global_eval_avg_geo_capacity",            "GEO Capacity (bps/Hz)"),
        ("global_eval_avg_leo_to_geo_interference", "LEO→GEO Interference"),
    ]
    log_scale = {"global_eval_avg_leo_to_geo_interference"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Convergence Curves — All Methods", fontsize=13)

    for ax, (key, title) in zip(axes.flat, metrics):
        for name, logs in all_logs.items():
            vals = extract(logs, key)
            if not vals:
                continue
            rounds = list(range(1, len(vals) + 1))
            sm = moving_average(vals, SMOOTH_WINDOW)
            ax.plot(rounds, vals, alpha=0.15, color=COLORS[name], linewidth=0.8)
            ax.plot(rounds, sm, label=name, color=COLORS[name],
                    linestyle=LINESTYLES[name], linewidth=2)
        if key in log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig1_convergence_curves", output_dir)


# ── Figure 2: Throughput & Interference Improvements ─────────────────────────

def plot_improvements(all_logs: Dict[str, List[dict]], output_dir: Path):
    methods = list(all_logs.keys())
    x = np.arange(len(methods))
    w = 0.25

    leo_imp = [pct_change(extract(all_logs[m], "global_eval_avg_leo_capacity"),           True)  for m in methods]
    geo_imp = [pct_change(extract(all_logs[m], "global_eval_avg_geo_capacity"),            True)  for m in methods]
    int_red = [pct_change(extract(all_logs[m], "global_eval_avg_leo_to_geo_interference"), False) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))

    def _labeled_bars(pos, vals, label, color):
        bars = ax.bar(pos, vals, w, label=label, color=color, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -10),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    _labeled_bars(x - w, leo_imp, "LEO Throughput Improvement (%)", "tab:blue")
    _labeled_bars(x,     geo_imp, "GEO Throughput Improvement (%)", "tab:orange")
    _labeled_bars(x + w, int_red, "Interference Reduction (%)",     "tab:red")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Improvement / Reduction (%)")
    ax.set_title(f"Performance Changes: First {IMPROVEMENT_WINDOW} vs Last {IMPROVEMENT_WINDOW} Rounds")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    save_fig(fig, "fig2_performance_improvements", output_dir)


# ── Figure 3: Training Efficiency ────────────────────────────────────────────

def plot_training_efficiency(all_logs: Dict[str, List[dict]], output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Efficiency", fontsize=13)

    ax = axes[0]
    for name, logs in all_logs.items():
        vals = extract(logs, "avg_param_delta_norm")
        if not vals:
            continue
        rounds = list(range(1, len(vals) + 1))
        sm = moving_average(vals, SMOOTH_WINDOW)
        ax.plot(rounds, vals, alpha=0.15, color=COLORS[name], linewidth=0.8)
        ax.plot(rounds, sm, label=name, color=COLORS[name],
                linestyle=LINESTYLES[name], linewidth=2)
    ax.set_title("Parameter Delta Norm")
    ax.set_xlabel("Round")
    ax.set_ylabel("‖Δθ‖")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    for name, logs in all_logs.items():
        vals = extract(logs, "avg_client_train_time_sec")
        if not vals:
            continue
        rounds = list(range(1, len(vals) + 1))
        ax.plot(rounds, np.cumsum(vals), label=name, color=COLORS[name],
                linestyle=LINESTYLES[name], linewidth=2)
    ax.set_title("Cumulative Training Time")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Time (s)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[2]
    methods = list(all_logs.keys())
    r_to_peak = []
    for name in methods:
        vals = extract(all_logs[name], "global_eval_mean_reward")
        r_to_peak.append(rounds_to_pct_of_peak(vals) if vals else 0)
    bars = ax.bar(methods, r_to_peak,
                  color=[COLORS[m] for m in methods],
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, r_to_peak):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, str(val),
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Rounds to {int(PEAK_THRESHOLD * 100)}% of Peak Reward")
    ax.set_ylabel("Rounds")
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig3_training_efficiency", output_dir)


# ── Figure 4: DDPG Selector Reward Components ────────────────────────────────

def plot_selector_rewards(all_logs: Dict[str, List[dict]], output_dir: Path):
    drl = {k: v for k, v in all_logs.items() if "DRL" in k}
    if not drl:
        print("  [skip] no DRL logs for selector reward plot")
        return

    n = len(drl)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False)
    fig.suptitle("DDPG Selector Reward Components", fontsize=13)

    for ax, (name, logs) in zip(axes[0], drl.items()):
        rounds = [r["round_idx"] for r in logs]
        for key, label, color in [
            ("fl_reward",       "FL Reward",       "tab:blue"),
            ("snr_comm_reward", "SNR/Comm Reward",  "tab:orange"),
            ("selector_reward", "Total Reward",     "tab:red"),
        ]:
            vals = [r.get(key, np.nan) for r in logs]
            ax.plot(rounds, vals, alpha=0.2, color=color, linewidth=0.8)
            ax.plot(rounds, moving_average(vals, SMOOTH_WINDOW),
                    label=label, color=color, linewidth=2)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_title(name)
        ax.set_xlabel("Round")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig4_selector_rewards", output_dir)


# ── Figure 5: Client Selection Heatmaps ──────────────────────────────────────

def plot_selection_heatmaps(all_logs: Dict[str, List[dict]], output_dir: Path):
    n_methods = len(all_logs)
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 3.2 * n_methods),
                              gridspec_kw={"width_ratios": [4, 1]})
    if n_methods == 1:
        axes = np.array([axes])
    fig.suptitle("Client Selection Patterns", fontsize=13)

    for row, (name, logs) in enumerate(all_logs.items()):
        ax_heat = axes[row, 0]
        ax_freq = axes[row, 1]

        num_rounds = len(logs)
        matrix = np.zeros((NUM_CLIENTS, num_rounds), dtype=np.float32)
        for col, r in enumerate(logs):
            for cid in r.get("selected_client_ids", []):
                if 0 <= cid < NUM_CLIENTS:
                    matrix[cid, col] = 1.0

        ax_heat.imshow(matrix, aspect="auto", cmap="Blues",
                       interpolation="nearest", vmin=0, vmax=1)
        ax_heat.set_title(f"{name} — Selection Heatmap")
        ax_heat.set_xlabel("Round")
        ax_heat.set_ylabel("Client ID")
        ax_heat.set_yticks(range(0, NUM_CLIENTS, 4))

        freq = matrix.sum(axis=1)
        jf   = jains_fairness(freq)
        ax_freq.barh(range(NUM_CLIENTS), freq, color=COLORS.get(name, "gray"), alpha=0.8)
        ax_freq.set_title(f"Freq  J={jf:.3f}")
        ax_freq.set_xlabel("Times Selected")
        ax_freq.set_yticks(range(0, NUM_CLIENTS, 4))
        ax_freq.invert_yaxis()
        ax_freq.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, "fig5_selection_heatmaps", output_dir)


# ── Figure 6: Final Performance Box Plots ────────────────────────────────────

def plot_final_performance_boxplots(all_logs: Dict[str, List[dict]], output_dir: Path):
    metrics = [
        ("global_eval_mean_reward",                "Global Reward"),
        ("global_eval_avg_leo_capacity",            "LEO Capacity"),
        ("global_eval_avg_geo_capacity",            "GEO Capacity"),
        ("global_eval_avg_leo_to_geo_interference", "Interference"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f"Final Performance Distribution (Last {LAST_N_ROUNDS} Rounds)", fontsize=13)

    for ax, (key, title) in zip(axes, metrics):
        data, labels = [], []
        for name, logs in all_logs.items():
            vals = extract(logs, key)
            if len(vals) >= LAST_N_ROUNDS:
                data.append(vals[-LAST_N_ROUNDS:])
                labels.append(name)
        if not data:
            ax.axis("off")
            continue
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, name in zip(bp["boxes"], labels):
            patch.set_facecolor(COLORS.get(name, "gray"))
            patch.set_alpha(0.75)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig6_final_performance_boxplots", output_dir)


# ── Figure 7: Client Selection Fairness ──────────────────────────────────────

def plot_fairness_comparison(all_logs: Dict[str, List[dict]], output_dir: Path):
    methods, fairness_vals = [], []
    for name, logs in all_logs.items():
        counts = np.zeros(NUM_CLIENTS, dtype=float)
        for r in logs:
            for cid in r.get("selected_client_ids", []):
                if 0 <= cid < NUM_CLIENTS:
                    counts[cid] += 1
        methods.append(name)
        fairness_vals.append(jains_fairness(counts))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(methods, fairness_vals,
                  color=[COLORS[m] for m in methods],
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, fairness_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Perfect Fairness (J=1)")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Client Selection Fairness")
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    save_fig(fig, "fig7_selection_fairness", output_dir)


# ── Figure 8: Model Generalisation Across Orbital Angles ─────────────────────

def plot_generalisation(all_logs: Dict[str, List[dict]], output_dir: Path):
    try:
        from stable_baselines3 import A2C
        from geoleo_env.env import LeoGeoEnv
    except ImportError as e:
        print(f"  [skip] generalisation plot: {e}")
        return

    eval_angles = list(range(-90, 181, 15))
    results: Dict[str, List[float]] = {}

    for name in LOG_DIRS:
        if name not in all_logs:
            continue
        model_path = LOG_DIRS[name] / "global_model_final.zip"
        if not model_path.exists():
            print(f"  [skip] {name}: model not found")
            continue

        print(f"  Evaluating {name} across {len(eval_angles)} angles...")
        rewards = []
        for angle in eval_angles:
            env = LeoGeoEnv(env_config={
                "initial_angle": angle,
                "angular_rate": 0.005,
                "speed": 1.508,
                "enable_gui": False,
                "max_steps": 850,
            })
            try:
                model = A2C.load(str(model_path), env=env)
                obs, _ = env.reset(seed=9999)
                ep_rewards = []
                for _ in range(300):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_rewards.append(float(reward))
                    if terminated or truncated:
                        obs, _ = env.reset()
                rewards.append(float(np.mean(ep_rewards)))
            except Exception as exc:
                rewards.append(np.nan)
                print(f"    Warning at {angle}°: {exc}")
            finally:
                env.close()
        results[name] = rewards

    if not results:
        print("  [skip] no trained models found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Generalisation Across Orbital Positions", fontsize=13)

    ax = axes[0]
    for name, rews in results.items():
        ax.plot(eval_angles, rews, label=name, color=COLORS.get(name, "gray"),
                linestyle=LINESTYLES.get(name, "-"), linewidth=2, marker="o", markersize=4)
    ax.axvspan(-45, 145, alpha=0.07, color="green", label="Training range")
    ax.set_xlabel("Initial Orbital Angle (°)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward vs Orbital Angle")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    for name, rews in results.items():
        arr          = np.array(rews, dtype=float)
        in_dist_mask = (np.array(eval_angles) >= -45) & (np.array(eval_angles) <= 145)
        baseline     = float(np.nanmean(arr[in_dist_mask])) if in_dist_mask.any() else 1.0
        if abs(baseline) > 1e-8:
            norm = arr / baseline * 100.0
            ax.plot(eval_angles, norm, label=name, color=COLORS.get(name, "gray"),
                    linestyle=LINESTYLES.get(name, "-"), linewidth=2, marker="o", markersize=4)
    ax.axvspan(-45, 145, alpha=0.07, color="green", label="Training range")
    ax.axhline(100, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Initial Orbital Angle (°)")
    ax.set_ylabel("Reward (% of in-distribution mean)")
    ax.set_title("Normalised Reward (Generalisation Gap)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig8_generalisation", output_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate FL-DRL result plots.")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save plots (default: plots/run_<timestamp>/)",
    )
    parser.add_argument(
        "--no-model-eval", action="store_true",
        help="Skip Figure 8 (requires loading trained model files)",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)

    print("Loading logs...")
    all_logs = load_logs()

    if not all_logs:
        print("No logs found — run training first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to: {output_dir.resolve()}\n")

    steps = [
        ("Figure 1: Convergence curves",         lambda: plot_convergence(all_logs, output_dir)),
        ("Figure 2: Performance improvements",    lambda: plot_improvements(all_logs, output_dir)),
        ("Figure 3: Training efficiency",         lambda: plot_training_efficiency(all_logs, output_dir)),
        ("Figure 4: DDPG selector rewards",       lambda: plot_selector_rewards(all_logs, output_dir)),
        ("Figure 5: Client selection heatmaps",   lambda: plot_selection_heatmaps(all_logs, output_dir)),
        ("Figure 6: Final performance box plots", lambda: plot_final_performance_boxplots(all_logs, output_dir)),
        ("Figure 7: Client selection fairness",   lambda: plot_fairness_comparison(all_logs, output_dir)),
    ]

    for label, fn in steps:
        print(label)
        fn()

    if not args.no_model_eval:
        print("Figure 8: Model generalisation (loading trained models...)")
        plot_generalisation(all_logs, output_dir)
    else:
        print("Figure 8: Skipped (--no-model-eval)")

    print(f"\nAll done. Plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
