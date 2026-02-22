import gymnasium
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env2 import LeoGeoEnv

# set the seed
seed = 42

gymnasium.register(
    id='LeoGeoEnv-v3.2',  # Use the same ID here as you used in the script
    entry_point='env2:LeoGeoEnv',
)

# Initialize the environment
env_id = "LeoGeoEnv-v3.2"
env = make_vec_env(env_id, n_envs=1)

epoch_length = 884 ## got through experiment
epoch_numbers = 100

# Set up the checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=epoch_length, save_path='./logs/', name_prefix='rl_model_A2C_2')

# Define a function for a linearly decreasing learning rate
def linear_schedule(initial_value):
    """
    Creates a schedule for a linearly decreasing parameter value.
    """
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


# Specify the policy network architecture, here we are using the default MLP

# Load the model with a dynamic learning rate schedule
model = A2C.load("a2c_leogeoenv_1", env=env, ent_coef=0.01, seed=seed, learning_rate=0.0001)

# Define the total number of timesteps to train the model
total_timesteps = epoch_length*epoch_numbers

# Train the model
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Save the model
model.save("a2c_leogeoenv_2")

# Close the environment
env.close()

# If you want to load the saved model, you can use the following:
# model = A2C.load("a2c_leogeoenv_2", env=env)
