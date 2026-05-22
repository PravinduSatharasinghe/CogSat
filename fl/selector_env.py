# fl/selector_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


@dataclass
class SelectorEnvConfig:
    num_clients: int
    clients_per_round: int
    alpha_reward: float = 1.0
    beta_leo_capacity: float = 1.0
    gamma_interference: float = 1e12
    eta_time: float = 0.01
    zeta_snr: float = 0.5    # reward for selecting high-LEO-capacity clients
    nu_comm: float = 0.1     # penalty for high communication cost (param_delta / leo_cap)


class ClientSelectorEnv(gym.Env):
    """
    Server-side Gymnasium environment for FL client selection.

    This environment does NOT train local A2C agents itself.
    It only:
      1. builds selector observations from client stats,
      2. converts selector actions into selected client IDs,
      3. computes selector reward after a FL round.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: SelectorEnvConfig):
        super().__init__()

        self.config = config
        self.num_clients = config.num_clients
        self.clients_per_round = config.clients_per_round

        # Per-client features:
        # 0 last local reward
        # 1 last train time
        # 2 last parameter drift
        # 3 last avg LEO capacity
        # 4 last avg GEO capacity
        # 5 last LEO-to-GEO interference
        # 6 rounds since last selected
        self.num_features = 7

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_clients, self.num_features),
            dtype=np.float32,
        )

        # Selector outputs one score per client.
        # The server then selects top-K clients.
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_clients,),
            dtype=np.float32,
        )

        self.round_idx = 0
        self.last_global_reward = None
        self.last_global_leo_capacity = None
        self.rounds_since_selected = np.zeros(self.num_clients, dtype=np.float32)

        self.current_obs = np.zeros(
            (self.num_clients, self.num_features),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.round_idx = 0
        self.last_global_reward = None
        self.last_global_leo_capacity = None
        self.rounds_since_selected[:] = 0.0
        self.current_obs[:] = 0.0

        return self.current_obs.copy(), {}

    def build_observation(self, clients: Sequence[Any]) -> np.ndarray:
        """
        Build selector state from FederatedClient.last_stats.
        """
        obs = np.zeros((self.num_clients, self.num_features), dtype=np.float32)

        for i, client in enumerate(clients):
            stats = client.last_stats

            obs[i, 0] = float(stats.get("last_mean_reward", 0.0))
            obs[i, 1] = float(stats.get("last_train_time_sec", 0.0))
            obs[i, 2] = float(stats.get("last_param_delta_norm", 0.0))
            obs[i, 3] = float(stats.get("last_avg_leo_capacity", 0.0))
            obs[i, 4] = float(stats.get("last_avg_geo_capacity", 0.0))
            obs[i, 5] = float(stats.get("last_avg_leo_to_geo_interference", 0.0))
            obs[i, 6] = float(self.rounds_since_selected[i])

        self.current_obs = obs
        return obs.copy()

    def select_top_k(self, action: np.ndarray) -> List[int]:
        """
        Convert selector action scores into selected client IDs.
        """
        scores = np.asarray(action, dtype=np.float32).reshape(-1)

        if scores.shape[0] != self.num_clients:
            raise ValueError(
                f"Expected action shape ({self.num_clients},), got {scores.shape}"
            )

        selected = np.argsort(scores)[-self.clients_per_round:]
        selected = selected[::-1]  # descending score
        return selected.tolist()

    def update_selection_counters(self, selected_client_ids: Sequence[int]) -> None:
        self.rounds_since_selected += 1.0
        for cid in selected_client_ids:
            self.rounds_since_selected[cid] = 0.0

    def compute_selector_reward(self, round_log: Dict[str, Any]) -> float:
        """
        Satellite-aware selector reward:

        r_t = alpha * ΔR_t
            + beta  * ΔC_LEO_t
            - gamma * I_LEO->GEO_t
            - eta   * T_t
        """

        global_reward = float(round_log["global_eval_mean_reward"])
        global_leo_capacity = float(round_log["global_eval_avg_leo_capacity"])
        interference = float(round_log["global_eval_avg_leo_to_geo_interference"])
        train_time = float(round_log["avg_client_train_time_sec"])

        if self.last_global_reward is None:
            delta_reward = 0.0
        else:
            delta_reward = global_reward - self.last_global_reward

        if self.last_global_leo_capacity is None:
            delta_leo_capacity = 0.0
        else:
            delta_leo_capacity = global_leo_capacity - self.last_global_leo_capacity

        reward = (
            self.config.alpha_reward * delta_reward
            + self.config.beta_leo_capacity * delta_leo_capacity
            - self.config.gamma_interference * interference
            - self.config.eta_time * train_time
        )

        self.last_global_reward = global_reward
        self.last_global_leo_capacity = global_leo_capacity

        return float(reward)

    def compute_snr_comm_reward(
        self,
        selected_client_ids: List[int],
        clients: Sequence[Any],
    ) -> float:
        """
        Two-part shaped reward based on channel quality of selected clients.

        r_snr  = zeta_snr  * (mean_leo_cap_selected / mean_leo_cap_all - 1)
                 Positive when selected clients have above-average LEO capacity.

        r_comm = -nu_comm * mean(param_delta_norm / leo_cap)  for selected clients
                 Penalises selecting clients whose parameter updates are expensive
                 to transmit given their current channel quality.
        """
        eps = 1e-8

        all_leo_caps = np.array(
            [float(c.last_stats.get("last_avg_leo_capacity", 0.0)) for c in clients],
            dtype=np.float32,
        )
        sel_leo_caps = all_leo_caps[selected_client_ids]
        mean_all = float(np.mean(all_leo_caps)) + eps

        r_snr = self.config.zeta_snr * (float(np.mean(sel_leo_caps)) / mean_all - 1.0)

        sel_param_deltas = np.array(
            [float(clients[i].last_stats.get("last_param_delta_norm", 0.0))
             for i in selected_client_ids],
            dtype=np.float32,
        )
        comm_costs = sel_param_deltas / (sel_leo_caps + eps)
        r_comm = -self.config.nu_comm * float(np.mean(comm_costs))

        return r_snr + r_comm

    def step(self, action):
        """
        This env does not directly execute FL rounds inside step().
        The main FL loop should:
          1. call select_top_k(action),
          2. run server.run_round(...),
          3. call compute_selector_reward(round_log).

        This placeholder keeps the class Gym-compatible.
        """
        selected_client_ids = self.select_top_k(action)

        info = {
            "selected_client_ids": selected_client_ids,
        }

        terminated = False
        truncated = False
        reward = 0.0

        return self.current_obs.copy(), reward, terminated, truncated, info