# fl/server.py
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import torch as th
from stable_baselines3 import A2C

from env import LeoGeoEnv


@dataclass
class ServerConfig:
    seed: int = 42
    min_clients_per_round: int = 3
    aggregation: str = "fedavg"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_env_config: Dict[str, Any] = field(default_factory=lambda: {
        "initial_angle": 0,
        "enable_gui": False,
    })


class FederatedServer:
    """
    Central server for federated A2C.
    Keeps:
    - global policy parameters
    - a lightweight global evaluation model
    - round logs
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.rng = random.Random(config.seed)

        self.eval_env = LeoGeoEnv(env_config=config.eval_env_config)
        self.global_model = self._make_global_model()
        self.round_logs: List[Dict[str, Any]] = []

    def _make_global_model(self):
        default_kwargs = {
            "policy": "MultiInputPolicy",
            "env": self.eval_env,
            "verbose": 0,
            "seed": self.config.seed,
            "device": "auto",
            "learning_rate": 1e-4,
            "ent_coef": 0.01,
        }
        default_kwargs.update(self.config.model_kwargs)
        model = A2C(**default_kwargs)
        return model

    def get_global_parameters(self) -> Dict[str, th.Tensor]:
        state_dict = self.global_model.policy.state_dict()
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def set_global_parameters(self, params: Dict[str, th.Tensor]) -> None:
        self.global_model.policy.load_state_dict(params, strict=True)

    def select_clients_random(self, clients: Sequence[Any], fraction: float) -> List[Any]:
        num_clients = len(clients)
        m = max(self.config.min_clients_per_round, int(np.ceil(fraction * num_clients)))
        m = min(m, num_clients)
        return self.rng.sample(list(clients), m)

    def select_clients_topk_reward(self, clients: Sequence[Any], fraction: float) -> List[Any]:
        """
        Heuristic selector baseline:
        choose clients with best last observed local reward.
        Useful before adding PPO/DDPG selector.
        """
        num_clients = len(clients)
        m = max(self.config.min_clients_per_round, int(np.ceil(fraction * num_clients)))
        m = min(m, num_clients)

        ranked = sorted(
            clients,
            key=lambda c: c.last_stats.get("last_mean_reward", 0.0),
            reverse=True
        )
        return ranked[:m]

    def aggregate_fedavg(
        self,
        client_updates: Sequence[Dict[str, Any]],
    ) -> Dict[str, th.Tensor]:
        """
        Weighted average by local num_steps.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation.")

        total_weight = sum(update["num_steps"] for update in client_updates)
        if total_weight <= 0:
            raise ValueError("Total aggregation weight must be positive.")

        aggregated = None

        for update in client_updates:
            client_params = update["parameters"]
            weight = update["num_steps"] / total_weight

            if aggregated is None:
                aggregated = {
                    k: client_params[k].float() * weight
                    for k in client_params.keys()
                }
            else:
                for k in aggregated.keys():
                    aggregated[k] += client_params[k].float() * weight

        return aggregated

    def run_round(
        self,
        round_idx: int,
        selected_clients: Sequence[Any],
    ) -> Dict[str, Any]:
        """
        One synchronous FL round:
        - broadcast global params
        - local client updates
        - aggregate
        - evaluate new global model
        """
        global_params = self.get_global_parameters()

        client_updates = []
        for client in selected_clients:
            client.set_parameters(global_params)
            update = client.train_local()
            client_updates.append(update)

        new_global_params = self.aggregate_fedavg(client_updates)
        self.set_global_parameters(new_global_params)

        global_eval = self.evaluate_global(num_steps=300)

        round_log = {
            "round_idx": round_idx,
            "selected_client_ids": [u["client_id"] for u in client_updates],
            "num_selected_clients": len(client_updates),
            "avg_client_train_time_sec": float(np.mean([u["train_time_sec"] for u in client_updates])),
            "avg_client_reward": float(np.mean([u["stats"]["last_mean_reward"] for u in client_updates])),
            "avg_client_leo_capacity": float(np.mean([u["stats"]["last_avg_leo_capacity"] for u in client_updates])),
            "avg_client_geo_capacity": float(np.mean([u["stats"]["last_avg_geo_capacity"] for u in client_updates])),
            "avg_client_leo_to_geo_interference": float(
                np.mean([u["stats"]["last_avg_leo_to_geo_interference"] for u in client_updates])
            ),
            "avg_param_delta_norm": float(np.mean([u["stats"]["last_param_delta_norm"] for u in client_updates])),
            "global_eval_mean_reward": global_eval["mean_reward"],
            "global_eval_avg_leo_capacity": global_eval["avg_leo_capacity"],
            "global_eval_avg_geo_capacity": global_eval["avg_geo_capacity"],
            "global_eval_avg_leo_to_geo_interference": global_eval["avg_leo_to_geo_interference"],
        }

        self.round_logs.append(copy.deepcopy(round_log))
        return round_log

    def evaluate_global(self, num_steps: int = 300) -> Dict[str, float]:
        obs, _ = self.eval_env.reset(seed=self.config.seed + 999)

        rewards = []
        leo_caps = []
        geo_caps = []
        leo_to_geo_int = []

        for _ in range(num_steps):
            action, _ = self.global_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)

            rewards.append(float(reward))

            if "avg_leo_user_capacity" in info and len(info["avg_leo_user_capacity"]) > 0:
                leo_caps.append(float(np.mean(info["avg_leo_user_capacity"])))
            if "avg_geo_user_capacity" in info:
                geo_caps.append(float(info["avg_geo_user_capacity"]))
            if "leo_to_geo_interference" in info and len(info["leo_to_geo_interference"]) > 0:
                leo_to_geo_int.append(float(np.mean(info["leo_to_geo_interference"])))

            if terminated or truncated:
                obs, _ = self.eval_env.reset()

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "avg_leo_capacity": float(np.mean(leo_caps)) if leo_caps else 0.0,
            "avg_geo_capacity": float(np.mean(geo_caps)) if geo_caps else 0.0,
            "avg_leo_to_geo_interference": float(np.mean(leo_to_geo_int)) if leo_to_geo_int else 0.0,
        }

    def close(self):
        self.eval_env.close()