# FedAvg_DRL.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium
import numpy as np
import torch as th
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from fl.client import ClientConfig, FederatedClient
from fl.server import FederatedServer, ServerConfig
from fl.selector_env import ClientSelectorEnv, SelectorEnvConfig


SEED = 42
NUM_CLIENTS = 20
CLIENT_FRACTION = 0.5
NUM_ROUNDS = 200
LOCAL_TIMESTEPS = 2000
SELECTOR_BUFFER_SIZE = 200    # replay buffer size (covers all FL rounds)
SELECTOR_LEARNING_STARTS = 20 # random exploration rounds before DDPG learns
SELECTOR_BATCH_SIZE = 32      # mini-batch size for DDPG gradient updates
OUTPUT_DIR = Path("logs/FedAvg_DRL")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


def register_env():
    try:
        gymnasium.register(
            id="LeoGeoEnv-v3.1",
            entry_point="geoleo_env.env:LeoGeoEnv",
        )
    except Exception:
        pass


def make_client_env_config(client_id: int) -> dict:
    return {
        "initial_angle": -45 + (client_id * 10),
        "angular_rate": 0.005,
        "speed": 1.508,
        "enable_gui": False,
        "max_steps": 850,
    }


def build_clients(num_clients: int) -> list[FederatedClient]:
    clients = []
    for client_id in range(num_clients):
        config = ClientConfig(
            client_id=client_id,
            env_config=make_client_env_config(client_id),
            local_timesteps=LOCAL_TIMESTEPS,
            seed=SEED,
            model_kwargs={
                "learning_rate": 1e-4,
                "ent_coef": 0.01,
            },
        )
        clients.append(FederatedClient(config))
    return clients


def save_round_logs_atomic(logs, output_dir: Path):
    tmp_path = output_dir / "round_logs.tmp.json"
    final_path = output_dir / "round_logs.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    tmp_path.replace(final_path)


class FLSelectorEnv(ClientSelectorEnv):
    """
    Extends ClientSelectorEnv with a live step() that executes one FL round.
    SB3 drives this env; each env.step() is one full FL round.
    """

    def __init__(
        self,
        config: SelectorEnvConfig,
        server: FederatedServer,
        clients: List[FederatedClient],
        output_dir: Path,
    ):
        super().__init__(config)
        self.server = server
        self.clients = clients
        self.output_dir = output_dir
        self._fl_round = 0
        self.last_round_log: Optional[Dict[str, Any]] = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self.build_observation(self.clients)
        return obs, info

    def step(self, action: np.ndarray):
        self._fl_round += 1

        selected_client_ids = self.select_top_k(action)
        selected_clients = [self.clients[i] for i in selected_client_ids]
        self.update_selection_counters(selected_client_ids)

        round_log = self.server.run_round(
            round_idx=self._fl_round,
            selected_clients=selected_clients,
        )

        fl_reward = self.compute_selector_reward(round_log)
        snr_comm_reward = self.compute_snr_comm_reward(selected_client_ids, self.clients)
        total_reward = fl_reward + snr_comm_reward

        round_log["fl_reward"] = fl_reward
        round_log["snr_comm_reward"] = snr_comm_reward
        round_log["selector_reward"] = total_reward
        round_log["selector_selected_client_ids"] = selected_client_ids
        self.last_round_log = round_log

        next_obs = self.build_observation(self.clients)

        return next_obs, total_reward, False, False, {"round_log": round_log}


class FLRoundCallback(BaseCallback):
    """Logs round metrics and saves checkpoints after each FL round."""

    def __init__(self, fl_env: FLSelectorEnv, output_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.fl_env = fl_env
        self.output_dir = output_dir

    def _on_step(self) -> bool:
        log = self.fl_env.last_round_log
        if log is None:
            return True

        round_idx = self.fl_env._fl_round
        print(
            f"[Round {round_idx:03d}] "
            f"selected={log['selected_client_ids']} | "
            f"total_reward={log['selector_reward']:.6f} "
            f"(fl={log['fl_reward']:.4f}, snr_comm={log['snr_comm_reward']:.4f}) | "
            f"global_reward={log['global_eval_mean_reward']:.6f} | "
            f"leo_cap={log['global_eval_avg_leo_capacity']:.6f} | "
            f"geo_cap={log['global_eval_avg_geo_capacity']:.6f} | "
            f"leo_to_geo_int={log['global_eval_avg_leo_to_geo_interference']:.6e}"
        )

        save_round_logs_atomic(self.fl_env.server.round_logs, self.output_dir)

        if round_idx % 10 == 0:
            self.fl_env.server.global_model.save(
                str(self.output_dir / f"global_model_round_{round_idx}.zip")
            )

        return True


def main():
    set_global_seed(SEED)
    register_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clients = build_clients(NUM_CLIENTS)

    server = FederatedServer(
        ServerConfig(
            seed=SEED,
            min_clients_per_round=3,
            model_kwargs={
                "learning_rate": 1e-4,
                "ent_coef": 0.01,
            },
            eval_env_config={
                "initial_angle": 5,
                "angular_rate": 0.005,
                "speed": 1.508,
                "enable_gui": False,
                "max_steps": 850,
            },
        )
    )

    clients_per_round = max(
        server.config.min_clients_per_round,
        int(np.ceil(NUM_CLIENTS * CLIENT_FRACTION)),
    )

    fl_env = FLSelectorEnv(
        config=SelectorEnvConfig(
            num_clients=NUM_CLIENTS,
            clients_per_round=clients_per_round,
            alpha_reward=1.0,
            beta_leo_capacity=1.0,
            gamma_interference=1e12,
            eta_time=0.01,
            zeta_snr=0.5,
            nu_comm=0.1,
        ),
        server=server,
        clients=clients,
        output_dir=OUTPUT_DIR,
    )
    fl_env.reset(seed=SEED)

    # Flatten (20, 7) obs to (140,) for SB3 MlpPolicy
    selector_env = FlattenObservation(fl_env)

    n_actions = selector_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    selector_model = DDPG(
        "MlpPolicy",
        selector_env,
        learning_rate=1e-3,
        buffer_size=SELECTOR_BUFFER_SIZE,
        learning_starts=SELECTOR_LEARNING_STARTS,
        batch_size=SELECTOR_BATCH_SIZE,
        tau=0.005,
        gamma=0.99,
        action_noise=action_noise,
        seed=SEED,
        verbose=0,
    )

    callback = FLRoundCallback(fl_env=fl_env, output_dir=OUTPUT_DIR)

    try:
        print("Starting federated A2C training with DDPG selector...")
        print(
            f"NUM_CLIENTS={NUM_CLIENTS}, "
            f"CLIENT_FRACTION={CLIENT_FRACTION}, "
            f"CLIENTS_PER_ROUND={clients_per_round}, "
            f"NUM_ROUNDS={NUM_ROUNDS}, "
            f"SELECTOR_BUFFER_SIZE={SELECTOR_BUFFER_SIZE}, "
            f"SELECTOR_LEARNING_STARTS={SELECTOR_LEARNING_STARTS}"
        )

        selector_model.learn(total_timesteps=NUM_ROUNDS, callback=callback)

        server.global_model.save(str(OUTPUT_DIR / "global_model_final.zip"))
        selector_model.save(str(OUTPUT_DIR / "selector_model_final.zip"))
        print("Federated training complete.")

    finally:
        for client in clients:
            client.close()
        server.close()


if __name__ == "__main__":
    main()
