# train_fed_a2c.py
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import gymnasium
import numpy as np
import torch as th

from env import LeoGeoEnv
from fl.client import ClientConfig, FederatedClient
from fl.server import FederatedServer, ServerConfig


SEED = 42
NUM_CLIENTS = 10
CLIENT_FRACTION = 0.4
NUM_ROUNDS = 10
LOCAL_TIMESTEPS = 2000
OUTPUT_DIR = Path("logs/federated_a2c")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


def register_env():
    try:
        gymnasium.register(
            id="LeoGeoEnv-v3.1",
            entry_point="env:LeoGeoEnv",
        )
    except Exception:
        # safe if already registered
        pass


def make_client_env_config(client_id: int) -> dict:
    """
    Heterogeneity hook:
    each client gets a slightly different environment config.
    """
    return {
        "initial_angle": -45 + (client_id * 10),
        "angular_rate": 0.005,
        "speed": 1.508,
        "enable_gui": False,
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
            }
        )
    )

    try:
        print("Starting federated A2C training...")
        print(f"NUM_CLIENTS={NUM_CLIENTS}, CLIENT_FRACTION={CLIENT_FRACTION}, NUM_ROUNDS={NUM_ROUNDS}")

        for round_idx in range(1, NUM_ROUNDS + 1):
            # Baseline selector: random
            selected_clients = server.select_clients_random(
                clients=clients,
                fraction=CLIENT_FRACTION,
            )

            round_log = server.run_round(
                round_idx=round_idx,
                selected_clients=selected_clients,
            )

            print(
                f"[Round {round_idx:03d}] "
                f"selected={round_log['selected_client_ids']} | "
                f"global_reward={round_log['global_eval_mean_reward']:.6f} | "
                f"leo_cap={round_log['global_eval_avg_leo_capacity']:.6f} | "
                f"geo_cap={round_log['global_eval_avg_geo_capacity']:.6f} | "
                f"leo_to_geo_int={round_log['global_eval_avg_leo_to_geo_interference']:.6e}"
            )

            # save round log incrementally
            # with open(OUTPUT_DIR / "round_logs.json", "w", encoding="utf-8") as f:
            #     json.dump(server.round_logs, f, indent=2)

            tmp_path = OUTPUT_DIR / "round_logs.tmp.json"
            final_path = OUTPUT_DIR / "round_logs.json"

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(server.round_logs, f, indent=2)

            tmp_path.replace(final_path)

            # periodic global checkpoint
            if round_idx % 10 == 0:
                server.global_model.save(str(OUTPUT_DIR / f"global_model_round_{round_idx}.zip"))

        # final save
        server.global_model.save(str(OUTPUT_DIR / "global_model_final.zip"))
        print("Federated training complete.")

    finally:
        for client in clients:
            client.close()
        server.close()


if __name__ == "__main__":
    main()