import gymnasium
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from env import LeoGeoEnv

# set the seed
seed = 42

gymnasium.register(
    id='LeoGeoEnv-v3.1',  # Use the same ID here as you used in the script
    entry_point='env:LeoGeoEnv',
)

# --- Configuration ---
N_ENVS = 10
ENABLE_GUI = False  # Switch to enable/disable GUI
# ---------------------

def make_env(rank, seed=0, gui=False):
    """
    Utility function for multiprocessed env.
    
    :param rank: index of the environment
    :param seed: the initial seed for RNG
    :param gui: boolean to enable GUI
    """
    def _init():
        # Define arbitrary initial angles. 
        # Example: 10 angles starting from -45, spaced by 10 degrees.
        initial_angle = -45 + (rank * 10) 
        
        env_config = {
            "initial_angle": initial_angle,
            "enable_gui": gui
        }
        
        env = LeoGeoEnv(env_config=env_config)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == '__main__':
    # Initialize the environments
    env_id = "LeoGeoEnv-v3.1"
    
    # If GUI is enabled, we cannot use SubprocVecEnv easily because Turtle is not thread-safe/process-safe for GUI windows.
    # We use DummyVecEnv for GUI debugging (slow, sequential) and SubprocVecEnv for training (fast, parallel).
    vec_env_cls = DummyVecEnv if ENABLE_GUI else SubprocVecEnv
    
    # Create the vectorized environment
    env = vec_env_cls([make_env(i, seed=seed, gui=ENABLE_GUI) for i in range(N_ENVS)])

    epoch_length = 884 ## got through experiment
    epoch_numbers = 100

    # Set up the checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=epoch_length, save_path='./logs/', name_prefix='rl_model_A2C')

    # Specify the policy network architecture, here we are using the default MIP
    model = A2C("MultiInputPolicy", env, ent_coef=0.01, verbose=1, tensorboard_log="./a2c_leogeo_tensorboard/",
                seed=seed, learning_rate=0.0001)

    # Define the total number of timesteps to train the model
    # Note: total_timesteps in SB3 is the total steps across all environments.
    total_timesteps = epoch_length * epoch_numbers

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the model
    model.save("a2c_leogeoenv_1")

    env.close()

    # If you want to load the saved model, you can use the following:
    # model = A2C.load("a2c_leogeoenv", env=env)









# import gymnasium
# from stable_baselines3 import A2C
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import CheckpointCallback
# from env import LeoGeoEnv

# # set the seed
# seed = 42

# gymnasium.register(
#     id='LeoGeoEnv-v3.1',  # Use the same ID here as you used in the script
#     entry_point='env:LeoGeoEnv',
# )

# # Initialize the environment
# env_id = "LeoGeoEnv-v3.1"
# env = make_vec_env(env_id, n_envs=1, seed=seed)

# epoch_length = 884 ## got through experiment
# epoch_numbers = 100

# # Set up the checkpoint callback
# checkpoint_callback = CheckpointCallback(save_freq=epoch_length, save_path='./logs/', name_prefix='rl_model_A2C')

# # Specify the policy network architecture, here we are using the default MIP
# model = A2C("MultiInputPolicy", env, ent_coef=0.01, verbose=1, tensorboard_log="./a2c_leogeo_tensorboard/",
#             seed=seed, learning_rate=0.0001)

# # Define the total number of timesteps to train the model
# total_timesteps = epoch_length*epoch_numbers

# # Train the model
# model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# # Save the model
# model.save("a2c_leogeoenv_1")

# env.close()

# # If you want to load the saved model, you can use the following:
# # model = A2C.load("a2c_leogeoenv", env=env)


