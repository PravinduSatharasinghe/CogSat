import gymnasium
import torch
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from env import LeoGeoEnv

# --- FL + DRL Configuration (Based on Paper Section V) ---
NUM_CLIENTS = 10        # Number of LEO Satellites
SELECTION_RATIO = 0.5   # 'C' in the paper (e.g., select 5 out of 10 clients per round) /partial client selection
FL_ROUNDS = 50          # Number of Communication Rounds
LOCAL_STEPS = 884       # Local training epochs before aggregation (B/E in paper)
# ---------------------------------------------------------

def create_local_satellite(client_id):
    """Creates an independent environment and A2C model for a specific LEO satellite."""
    # Each satellite gets a different orbit angle to simulate Heterogeneous (non-IID) data
    initial_angle = -45 + (client_id * 10)
    env_config = {
        "initial_angle": initial_angle,
        "enable_gui": False # Must be False for parallel multi-agent training
    }
    
    env = LeoGeoEnv(env_config=env_config)
    env.reset(seed=42 + client_id)
    
    # Local Model (The edge LEO brain)
    model = A2C("MultiInputPolicy", env, ent_coef=0.01, verbose=0, learning_rate=0.0001)
    return env, model

if __name__ == '__main__':
    print("Initializing Satellite FL Clients...")
    clients =[]
    
    # 1. INITIALIZATION STAGE (Paper Section IV-B.1)
    for i in range(NUM_CLIENTS):
        env, model = create_local_satellite(i)
        clients.append({"id": i, "env": env, "model": model})
        
    # Extract the global architecture from the first client
    global_model = clients[0]["model"]
    global_weights = global_model.policy.state_dict()

    print(f"Starting Federated Learning over {FL_ROUNDS} rounds...")

    for fl_round in range(FL_ROUNDS):
        print(f"\n========== FL Communication Round {fl_round + 1}/{FL_ROUNDS} ==========")
        
        # --- CLIENT SELECTION (Paper Section IV-C: Node Selection) ---
        # The paper selects nodes based on Training Quality (Cost Function).
        # In DRL, "Training Quality" = High average reward (or low interference penalty).
        # We evaluate all clients quickly to find the best candidates.
        client_scores =[]
        for client in clients:
            # Evaluate current model performance to simulate Equation (9) "Training Quality"
            mean_reward, _ = evaluate_policy(client["model"], client["env"], n_eval_episodes=1)
            client_scores.append((client["id"], mean_reward))
        
        # Sort clients by highest reward (lowest loss/cost)
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top 'C' ratio of clients (e.g., Top 5)
        num_to_select = max(int(NUM_CLIENTS * SELECTION_RATIO), 1)
        selected_client_ids = [score[0] for score in client_scores[:num_to_select]]
        print(f"Server Selected Clients for Aggregation: {selected_client_ids}")

        # --- LOCAL TRAINING STAGE (Paper Section IV-B.2) ---
        # Only the selected clients train on their local data
        local_weights =[]
        for client_id in selected_client_ids:
            client = clients[client_id]
            print(f"  -> Client {client_id} training locally for {LOCAL_STEPS} steps...")
            
            # Train the local A2C model
            client["model"].learn(total_timesteps=LOCAL_STEPS, reset_num_timesteps=False)
            
            # Extract the trained weights (Gradients/Parameters)
            local_weights.append(client["model"].policy.state_dict())

        # --- AGGREGATION STAGE (Paper Section IV-B.3, Equation 4) ---
        print("Central Server: Aggregating local models (FedAvg)...")
        
        # Average the weights of the selected clients
        for key in global_weights.keys():
            # Stack the tensors from the selected local models and take the mean
            stacked_weights = torch.stack([weights[key] for weights in local_weights])
            global_weights[key] = stacked_weights.mean(dim=0)
            
        # --- GLOBAL BROADCAST ---
        # Distribute the newly averaged Global Model to ALL 10 LEO satellites
        for client in clients:
            client["model"].policy.load_state_dict(global_weights)

        print(f"Round {fl_round + 1} completed successfully.")

    # Save the final Global Model
    print("\nTraining Complete. Saving Global Model...")
    global_model.save("fl_global_a2c_leogeo")