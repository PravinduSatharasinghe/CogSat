# fl/client.py
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import gymnasium
import numpy as np
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from geoleo_env.env import LeoGeoEnv


@dataclass
class ClientConfig:
    client_id: int
    env_config: Dict[str, Any]
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    local_timesteps: int = 2000
    seed: int = 42
    device: str = "auto"


class FederatedClient:
    """
    One federated RL client:
    - owns a local env
    - owns a local A2C model
    - can receive global weights
    - trains locally
    - returns updated weights + training stats
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.client_id = config.client_id
        self.env = self._make_env()
        self.model = self._make_model()

        self.last_stats: Dict[str, Any] = {
            "client_id": self.client_id,
            "selected_rounds": 0,
            "last_mean_reward": 0.0,
            "last_train_time_sec": 0.0,
            "last_num_steps": 0,
            "last_avg_leo_capacity": 0.0,
            "last_avg_geo_capacity": 0.0,
            "last_avg_leo_to_geo_interference": 0.0,
            "last_param_delta_norm": 0.0,
        }

    def _make_env(self):
        env = LeoGeoEnv(env_config=self.config.env_config)
        env = Monitor(env)
        env.reset(seed=self.config.seed + self.client_id)
        return env

    def _make_model(self):
        default_kwargs = {
            "policy": "MultiInputPolicy",
            "env": self.env,
            "verbose": 0,
            "seed": self.config.seed + self.client_id,
            "device": self.config.device,
            "learning_rate": 1e-4,
            "ent_coef": 0.01,
        }
        default_kwargs.update(self.config.model_kwargs)
        model = A2C(**default_kwargs)
        return model

    def get_parameters(self) -> Dict[str, th.Tensor]:
        """
        Return a CPU copy of the policy state dict.
        For SB3 A2C, the trainable parameters live under model.policy.
        """
        state_dict = self.model.policy.state_dict()
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def set_parameters(self, global_parameters: Dict[str, th.Tensor]) -> None:
        """
        Load global weights into the local model.
        """
        self.model.policy.load_state_dict(global_parameters, strict=True)

    def _parameter_delta_norm(
        self,
        before: Dict[str, th.Tensor],
        after: Dict[str, th.Tensor],
    ) -> float:
        sq_sum = 0.0
        for key in before.keys():
            diff = after[key].float() - before[key].float()
            sq_sum += float(th.sum(diff * diff).item())
        return float(np.sqrt(sq_sum))

    def train_local(self) -> Dict[str, Any]:
        """
        Train locally for config.local_timesteps and return:
        - updated parameters
        - client stats
        """
        params_before = self.get_parameters()

        start_time = time.perf_counter()
        self.model.learn(total_timesteps=self.config.local_timesteps, reset_num_timesteps=False)
        train_time = time.perf_counter() - start_time

        params_after = self.get_parameters()
        delta_norm = self._parameter_delta_norm(params_before, params_after)

        eval_stats = self.evaluate_local(num_steps=200)

        self.last_stats.update({
            "client_id": self.client_id,
            "selected_rounds": self.last_stats["selected_rounds"] + 1,
            "last_mean_reward": eval_stats["mean_reward"],
            "last_train_time_sec": train_time,
            "last_num_steps": self.config.local_timesteps,
            "last_avg_leo_capacity": eval_stats["avg_leo_capacity"],
            "last_avg_geo_capacity": eval_stats["avg_geo_capacity"],
            "last_avg_leo_to_geo_interference": eval_stats["avg_leo_to_geo_interference"],
            "last_param_delta_norm": delta_norm,
        })

        return {
            "client_id": self.client_id,
            "num_steps": self.config.local_timesteps,
            "train_time_sec": train_time,
            "parameters": params_after,
            "stats": copy.deepcopy(self.last_stats),
        }

    def evaluate_local(self, num_steps: int = 200) -> Dict[str, float]:
        """
        Lightweight deterministic-ish rollout for logging.
        This is not meant to be a rigorous benchmark.
        """
        obs, _ = self.env.reset(seed=self.config.seed + 10_000 + self.client_id)

        rewards = []
        leo_caps = []
        geo_caps = []
        leo_to_geo_int = []

        for _ in range(num_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)

            rewards.append(float(reward))

            if "avg_leo_user_capacity" in info and len(info["avg_leo_user_capacity"]) > 0:
                leo_caps.append(float(np.mean(info["avg_leo_user_capacity"])))
            if "avg_geo_user_capacity" in info:
                geo_caps.append(float(info["avg_geo_user_capacity"]))
            if "leo_to_geo_interference" in info and len(info["leo_to_geo_interference"]) > 0:
                leo_to_geo_int.append(float(np.mean(info["leo_to_geo_interference"])))

            if terminated or truncated:
                obs, _ = self.env.reset()

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "avg_leo_capacity": float(np.mean(leo_caps)) if leo_caps else 0.0,
            "avg_geo_capacity": float(np.mean(geo_caps)) if geo_caps else 0.0,
            "avg_leo_to_geo_interference": float(np.mean(leo_to_geo_int)) if leo_to_geo_int else 0.0,
        }

    def close(self):
        self.env.close()