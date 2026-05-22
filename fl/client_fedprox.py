# fl/client_fedprox.py
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch as th

from fl.client import ClientConfig, FederatedClient


@dataclass
class FedProxClientConfig(ClientConfig):
    mu: float = 0.05
    prox_interval_timesteps: int = 500

    def __post_init__(self):
        if self.prox_interval_timesteps < 1:
            raise ValueError(
                f"prox_interval_timesteps must be >= 1, got {self.prox_interval_timesteps}"
            )


class FedProxClient(FederatedClient):
    """
    FedProx-style client without modifying SB3 internals.

    After every prox_interval_timesteps of local A2C training, parameters are
    pulled toward the received global parameters via the proximal operator:

        theta <- theta_global + (1 / (1 + mu)) * (theta_local - theta_global)

    This is equivalent to one proximal-gradient step that minimises
    (mu/2) * ||theta - theta_global||^2, approximating the FedProx objective.
    Only floating-point parameters are projected; integer buffers are kept as-is.
    """

    def __init__(self, config: FedProxClientConfig):
        super().__init__(config)
        self.config: FedProxClientConfig = config
        self.global_reference_params: dict | None = None

    def set_parameters(self, global_parameters):
        super().set_parameters(global_parameters)
        # Infer the model device once and store global params on that device
        # so apply_proximal_projection avoids repeated .to() calls.
        device = next(self.model.policy.parameters()).device
        self.global_reference_params = {
            k: v.detach().to(device).clone()
            for k, v in global_parameters.items()
        }

    def apply_proximal_projection(self):
        if self.global_reference_params is None:
            return

        shrink = 1.0 / (1.0 + self.config.mu)
        current_params = self.model.policy.state_dict()
        new_params = {}

        with th.no_grad():
            for k, local_tensor in current_params.items():
                global_tensor = self.global_reference_params[k]

                if not local_tensor.is_floating_point():
                    # Integer buffers (e.g. num_batches_tracked) are not projected.
                    new_params[k] = local_tensor
                    continue

                new_params[k] = global_tensor + shrink * (local_tensor - global_tensor)

        self.model.policy.load_state_dict(new_params, strict=True)

    def train_local(self):
        params_before = self.get_parameters()

        start_time = time.perf_counter()

        remaining = self.config.local_timesteps
        interval = self.config.prox_interval_timesteps

        while remaining > 0:
            steps = min(interval, remaining)
            self.model.learn(total_timesteps=steps, reset_num_timesteps=False)
            self.apply_proximal_projection()
            remaining -= steps

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
            "fedprox_mu": self.config.mu,
        })

        return {
            "client_id": self.client_id,
            "num_steps": self.config.local_timesteps,
            "train_time_sec": train_time,
            "parameters": params_after,
            "stats": copy.deepcopy(self.last_stats),
        }
