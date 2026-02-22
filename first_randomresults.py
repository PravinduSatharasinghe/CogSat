import gymnasium
from leouser import *
import math
import matplotlib.pyplot as plt
import pathloss

# Example usage (commented to prevent execution here):
# print(average_by_position([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1], [2, 3, 4, 5, 6, 7, 8]]))


# gymnasium.register(
#     id='LeoGeoEnv-v3.1',  # Use the same ID here as you used in the script
#     entry_point='env:LeoGeoEnv',
# )
#
# # Initialize the environment
# env = gymnasium.make('LeoGeoEnv-v3.1')


gymnasium.register(
    id='LeoGeoEnv-v3.1',  # Use the same ID here as you used in the script
    entry_point='env:LeoGeoEnv',
)

# Initialize the environment
env = gymnasium.make('LeoGeoEnv-v3.1')


state = env.reset()
done = False
score = 0

leo_average_capacity = []
geo_average_capacity = []
leo_to_geo_interference = []
leo_user_interference = []
leo1_per_beam_capacity_list = []
leo2_per_beam_capacity_list = []


for i in range(884):  # 884 is the episode length
    leo_capacity = 0
    geo_capacity = 0
    action = env.action_space.sample()
    n_state, reward, done, some, info = env.step(action)

    leo_average_capacity.append(info.get('avg_leo_user_capacity', None))
    leo_user_interference.append(info.get('leo_user_interference', None))
    geo_average_capacity.append(info.get('avg_geo_user_capacity', None))
    leo_to_geo_interference.append(info.get('leo_to_geo_interference', None))
    leo1_per_beam_capacity_list.append(info.get('leo1_per_beam_capacity'))
    leo2_per_beam_capacity_list.append(info.get('leo2_per_beam_capacity'))


env.close()
