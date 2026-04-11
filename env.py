'''Changed to have only 1 Leo satelitte'''

import gymnasium
from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete
import numpy as np
import random
import math

import geointeference
import leointeference
import pathloss
from leo import *
from geo import *
from leouser import *
import turtle

# 1 pixel = 5 km

FREQUENCY = {0: "red", 1:"orange", 2: "magenta", 3: "green", 4:"blue", 5:"purple", 6:"pink", 7:'aqua', 8:'khaki', 9:'mistyrose'}

# --- Mock Classes for Headless Training ---
class MockScreen:
    def setup(self, width, height): pass
    def title(self, title): pass
    def tracer(self, n): pass
    def update(self): pass

class MockTurtle:
    def __init__(self):
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._color = "black"
        self._fillcolor = "black"
        self._pen_down = False
        
    def penup(self): self._pen_down = False
    def pendown(self): self._pen_down = True
    def speed(self, x): pass
    def shape(self, x): pass
    def shapesize(self, x): pass
    
    def color(self, c=None): 
        if c is not None:
            self._color = c
        return self._color

    def fillcolor(self, c=None): 
        if c is not None:
            self._fillcolor = c
        return self._fillcolor
    
    def goto(self, x, y):
        self._x = float(x)
        self._y = float(y)
        
    def setheading(self, h): self._heading = float(h)
    def heading(self): return self._heading
    
    def forward(self, distance):
        # Calculate new position based on heading
        rad = math.radians(self._heading)
        self._x += distance * math.cos(rad)
        self._y += distance * math.sin(rad)
        
    def xcor(self): return self._x
    def ycor(self): return self._y
    def position(self): return (self._x, self._y)
    
    # --- FIXED DISTANCE METHOD ---
    def distance(self, x, y=None):
        if y is not None:
            # Case 1: distance(x, y) - Two arguments
            ox, oy = x, y
        elif hasattr(x, 'xcor'):
            # Case 2: distance(turtle_object)
            ox, oy = x.xcor(), x.ycor()
        else:
            # Case 3: distance((x, y)) - Tuple argument
            ox, oy = x
        
        return math.sqrt((self._x - float(ox))**2 + (self._y - float(oy))**2)
    # -----------------------------
    
    def isdown(self): return self._pen_down

# ------------------------------------------

class LeoGeoEnv(gymnasium.Env):
    # env_config=None, render_mode=None added to match with SB3 configurations
    # to use with Ray RLlib "self, env_config" is enough
    def __init__(self, env_config=None, render_mode=None):
        super(LeoGeoEnv, self).__init__()
        self.leo_speed = 1.508
        if env_config is None:
            env_config = {}
        
        # GUI Switch Logic
        self.enable_gui = env_config.get("enable_gui", False)
        
        if self.enable_gui:
            self.turtle_provider = turtle
            self.turtle_class = turtle.Turtle
            self.screen = turtle.Screen()
            self.screen.setup(800, 800)
            self.screen.title(f"LEO GEO CoExisting Use Case - Angle {env_config.get('initial_angle', -45)}")
            # self.screen.tracer(0)
        else:
            self.turtle_provider = MockScreen() 
            self.turtle_class = MockTurtle
            self.screen = MockScreen() 
            self.screen.setup(800, 800)

        self.orbit_config = {
            "initial_angle": env_config.get("initial_angle", -45),
            "angular_rate": env_config.get("angular_rate", 0.005),
            "speed": env_config.get("speed", self.leo_speed)
        }
        if not hasattr(self, 'spec') or self.spec is None:
            self.spec = gymnasium.envs.registration.EnvSpec("LeoGeoEnv-v3.1")
        self.geo_user_int_threshold = 0
        self.leo_user_SINR_threshold = 5

        # Actions
        self.action_space = MultiDiscrete([10]*7) #reduced to 10*7 because of single LEO satelitte

        # Observation space
        # shape in beam_positions changed from 28 to 7
        self.observation_space = Dict({
            "time_step": Box(low=0, high=500, shape=(1,), dtype=np.int64),
            "beam_positions": Box(low=-1000, high=1000, shape=(14,), dtype=np.float64),
            "previous_actions": Box(low=0, high=9, shape=(7,), dtype=np.int64)
        })

        
        self.leo_step = 0  # to define an angular movement
        self.screen_edge = 450  # to stop the simulation from moving beyond the define edge

        # geo users - Pass turtle provider
        self.geo_system = GeoBeams(turtle_provider=self.turtle_class, screen_provider=self.screen)

        # LEOs (Keeping only L1)
        self.leo1_direction = 90
        # Pass turtle provider
        self.leo1 = LEOTurtle(x=-285, y=285, direction=self.leo1_direction, turtle_provider=self.turtle_class)
        self.leo1_xy = self.leo1.get_coordinates()

        # LEO users
        # Pass turtle provider
        self.leo_users = LeoUserConfig(l1_all_turtles=self.leo1.all_turtles, turtle_provider=self.turtle_class)
        self.leo1_users = len(self.leo_users.LEO_A_USER_COORDINATES)

        self.time_step = 0

        self.terminated = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.leo_step = 0  # to define an angular movement
        self.time_step = 0 # reset the time counter
        self.terminated = False

        # Observation space
        # Apply actions for each agent, return observations, rewards, dones, and optional info
        # shape in beam_positions changed from 28 to 7
        #beam positions reduced to 14
        observation = {
            "time_step": np.array([self.time_step], dtype=np.int64),
            "beam_positions": np.random.uniform(low=-1000, high=1000, size=(14,)),
            "previous_actions": np.random.randint(low=0, high=9, size=(7,), dtype=np.int64)
        }

        for leo_beam in range(7):
            for beam, l1_xy in zip(range(7), self.leo1_xy):
                self.leo1.all_turtles[beam].penup()  
                self.leo1.all_turtles[beam].goto(l1_xy[0], l1_xy[1]) 
                self.leo1.all_turtles[leo_beam].setheading(self.leo1_direction)
                self.leo1.all_turtles[leo_beam].pendown()

        self.leo_users = LeoUserConfig(
            l1_all_turtles=self.leo1.all_turtles,
            turtle_provider=self.turtle_class
        )

        return observation, {}

    def save_env_state(self):
        state = {
            'time_step': self.time_step,
            'leo1_positions': [(t.xcor(), t.ycor()) for t in self.leo1.all_turtles],
            'leo1_directions': [t.heading() for t in self.leo1.all_turtles],
            'leo1_pen': [t.isdown() for t in self.leo1.all_turtles],
            'leo1_direction': self.leo1_direction,
            'leo_step': self.leo_step,
            'terminated': self.terminated,
        }
        return state

    def restore_env_state(self, state):
        self.time_step = state['time_step']

        for t, pos, heading, pen in zip(self.leo1.all_turtles, state['leo1_positions'], state['leo1_directions'],
                                             state['leo1_pen']):
            t.penup()
            t.goto(pos[0], pos[1])
            t.setheading(heading)
            if pen:
                t.pendown()

        self.leo1_direction = state['leo1_direction']
        self.leo_step = state['leo_step']
        self.terminated = state['terminated']

    def step(self, actions):
        leo_beam_positions = []

        distance = self.leo_speed

        angle = (
            self.orbit_config["initial_angle"]
            + self.leo_step * self.orbit_config["angular_rate"]
        )

        self.leo1.move(
            self.orbit_config["speed"],
            angle=angle
        )

        for leo_beam in range(7):
            leo_beam_positions.append(float(self.leo1.all_turtles[leo_beam].xcor()))
            leo_beam_positions.append(float(self.leo1.all_turtles[leo_beam].ycor()))

        self.leo_step += 1   # define the intensity of angle/ *****changed from 0.005 to 1*****
        self.time_step += 1

        observation = {
        "time_step": np.array([self.time_step], dtype=np.int64),
        "beam_positions":np.array(leo_beam_positions, dtype=np.float64),
        "previous_actions": np.array(actions,dtype=np.int64)
        }

        ## Reward calculation
        leo_bandwidth = 0.2  # beam bandwidth in MHz

        # Assigning actions to LEO beams
        for a, leo_beam in zip(actions[0:7], range(7)):
            self.leo1.all_turtles[leo_beam].color(FREQUENCY[a])

        # list with leo user capacity
        self.leo_user_capacity = []

        # list with geo user capacity
        self.geo_user_capacity = []

        # list with leo interference to GEO users
        self.leo_to_geo_user_interference = []

        self.leo_user_interference = []

        leo1_per_beam_capacity = {}

        # Calculate the SINR of GEO users
        # This is were work is ongoing
        # P_watts = 10 ** (dBW / 10)  # convert dBW to watts

        for user, geo_beam in self.geo_system.geo_user_beam.items():
            
            # Use parentheses () to get the string value of the color
            current_beam_color = geo_beam.fillcolor() 
            bandwidth_ratio = len(self.geo_system.freq_sub[current_beam_color]) 

            leo_interference, geo_sinr = geointeference.geo_user_interference(user=user, beam=geo_beam, \
            l1_turtles=self.leo1.all_turtles, leo_freq_sub=self.leo1.freq_sub, geo_freq_sub=self.geo_system.freq_sub,
            bandwidth_ratio = bandwidth_ratio)
            
            self.leo_to_geo_user_interference.append(leo_interference)
            self.geo_user_capacity.append(leo_bandwidth*len(self.geo_system.freq_sub[current_beam_color]) * math.log2(1 + geo_sinr))

        # Calculate the SINR of LEO-1 users (rmeoved the role of L1 in here)
        for user, leo_beam in self.leo_users.leo1_for_user.items():
            user_leo1_sinr, user_leo1_interference = leointeference.leo_user_interference(user=user, leo_beam=leo_beam, \
            sat_altitude=pathloss.leo_satellite_altitude_A,int_sat_altitude=pathloss.leo_satellite_altitude_B, \
            geo_beams=self.geo_system.geo_beams, \
            leo_freq_sub=self.leo1.freq_sub, geo_freq_sub=self.geo_system.freq_sub)
            leo1_capacity_cal = leo_bandwidth * math.log2(1 + user_leo1_sinr)
            self.leo_user_capacity.append(leo1_capacity_cal)
            self.leo_user_interference.append(user_leo1_interference)

            if leo_beam not in leo1_per_beam_capacity:
                leo1_per_beam_capacity[leo_beam] = [leo1_capacity_cal]
            else:
                leo1_per_beam_capacity[leo_beam].append(leo1_capacity_cal)

        new_leo1_per_beam_capacity = {i: value for i, (_, value) in enumerate(leo1_per_beam_capacity.items(), start=1)}
        new_leo1_per_beam_capacity = {key: float(sum(values)) / len(values) for key, values in
                                      new_leo1_per_beam_capacity.items()}


        # weights for reward function
        # reward = Avg.LEO user capacity - weight* Avg.GEO interference
        # weight1 = 1
        # weight2 = -1e13
        # reward = weight1*sum(self.leo_user_capacity)/(len(self.leo_user_capacity))\
        #          + weight2*sum(self.leo_to_geo_user_interference)/len(self.leo_to_geo_user_interference)

        # ---- reward terms ----
        mean_leo_capacity = (
            float(sum(self.leo_user_capacity) / len(self.leo_user_capacity))
            if len(self.leo_user_capacity) > 0 else 0.0
        )

        mean_geo_capacity = (
            float(sum(self.geo_user_capacity) / len(self.geo_user_capacity))
            if len(self.geo_user_capacity) > 0 else 0.0
        )

        mean_leo_to_geo_interference = (
            float(sum(self.leo_to_geo_user_interference) / len(self.leo_to_geo_user_interference))
            if len(self.leo_to_geo_user_interference) > 0 else 0.0
        )

        # ---- reward weights ----
        alpha = 1.0
        gamma = 0.1
        beta = 1e13

        reward = (
                alpha * mean_leo_capacity
                + gamma * mean_geo_capacity
                - beta * mean_leo_to_geo_interference
        )

        # info = {'avg_leo_user_capacity': self.leo_user_capacity,
        #         'avg_geo_user_capacity': sum(self.geo_user_capacity)/(len(self.geo_user_capacity)),
        #         'leo_to_geo_interference': self.leo_to_geo_user_interference,
        #         'leo_user_interference': self.leo_user_interference,
        #         'leo1_per_beam_capacity': new_leo1_per_beam_capacity,
        #         }

        info = {
            'avg_leo_user_capacity': self.leo_user_capacity,
            'avg_geo_user_capacity': mean_geo_capacity,
            'leo_to_geo_interference': self.leo_to_geo_user_interference,
            'leo_user_interference': self.leo_user_interference,
            'leo1_per_beam_capacity': new_leo1_per_beam_capacity,
            'reward_terms': {
                'mean_leo_capacity': mean_leo_capacity,
                'mean_geo_capacity': mean_geo_capacity,
                'mean_leo_to_geo_interference': mean_leo_to_geo_interference,
                'alpha_mean_leo': alpha * mean_leo_capacity,
                'gamma_mean_geo': gamma * mean_geo_capacity,
                'minus_beta_interference': -beta * mean_leo_to_geo_interference,
            }
        }

        truncated = False
        return observation, reward, self.terminated, truncated, info

    def render(self):
        # Implement viz
        pass



