import base64
import glob
import io
import os
import re

import numpy as np
import pandas as pd
import random
from collections import defaultdict

from IPython.display import HTML
from IPython import display
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium import spaces

import minigrid
import minigrid.envs
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

max_env_steps = 50

class FlatObsWrapper(gym.core.ObservationWrapper):
    """Fully observable gridworld returning a flat grid encoding."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=((self.unwrapped.width-2) * (self.unwrapped.height-2) * 3,),
            dtype='uint8'
        )
        self.unwrapped.max_steps = max_env_steps

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        full_grid = full_grid[1:-1, 1:-1]
        
        flattened_grid = full_grid.ravel()
        return flattened_grid
    
    def render(self, *args, **kwargs):
        #kwargs['highlight'] = False  #설치한 gymnasim과 minigrid 버전에서는 highlight 인자 존재하지 않음
        return self.unwrapped.render(*args, **kwargs)

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[-1]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
        
def wrap_env(env):
    env = RecordVideo(env, './video')
    return env

def gen_wrapped_env(env_name):
    env = gym.make(env_name, render_mode='rgb_array') #설치한 환경에서는 RecordVideo가 저장하기 위해 rgb_array로 설정
    return wrap_env(FlatObsWrapper(env))