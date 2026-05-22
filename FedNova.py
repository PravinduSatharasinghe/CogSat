# FedNova.py
from FedAvg import (
    SEED,
    NUM_CLIENTS,
    CLIENT_FRACTION,
    NUM_ROUNDS,
    set_global_seed,
    register_env,
    build_clients,
    save_round_logs_atomic,
)
from fl.server import FederatedServer, ServerConfig
from pathlib import Path


OUTPUT_DIR = Path("logs/FedNova")


def main():
    set_global_seed(SEED)
    register_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clients = build_clients(NUM_CLIENTS)

    server = FederatedServer(
        ServerConfig(
            seed=SEED,
            min_clients_per_round=3,
            aggregation="fednova",
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

    try:
        print("Starting Federated A2C with FedNova + random selection...")
        print(f"NUM_CLIENTS={NUM_CLIENTS}, CLIENT_FRACTION={CLIENT_FRACTION}, NUM_ROUNDS={NUM_ROUNDS}")

        for round_idx in range(1, NUM_ROUNDS + 1):
            selected_clients = server.select_clients_random(
                clients=clients,
                fraction=CLIENT_FRACTION,
            )

            round_log = server.run_round(
                round_idx=round_idx,
                selected_clients=selected_clients,
            )

            round_log["method"] = "FedNova + random selection"

            print(
                f"[Round {round_idx:03d}] "
                f"selected={round_log['selected_client_ids']} | "
                f"global_reward={round_log['global_eval_mean_reward']:.6f} | "
                f"leo_cap={round_log['global_eval_avg_leo_capacity']:.6f} | "
                f"geo_cap={round_log['global_eval_avg_geo_capacity']:.6f} | "
                f"leo_to_geo_int={round_log['global_eval_avg_leo_to_geo_interference']:.6e} | "
                f"delta={round_log['avg_param_delta_norm']:.6f}"
            )

            save_round_logs_atomic(server.round_logs, OUTPUT_DIR)

            if round_idx % 10 == 0:
                server.global_model.save(str(OUTPUT_DIR / f"global_model_round_{round_idx}.zip"))

        server.global_model.save(str(OUTPUT_DIR / "global_model_final.zip"))
        print("FedNova training complete.")

    finally:
        for client in clients:
            client.close()
        server.close()


if __name__ == "__main__":
    main()
