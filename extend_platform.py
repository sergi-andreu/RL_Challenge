# RL challenge in the Platform Environment
# Code author: Sergi Andreu

# Here we modify the platform environment to create better gifs
# (platform env render only shows the player on the platform)

# Import all required data from the platform environment
import gym_platform.envs.platform_env
from gym_platform.envs.platform_env import PlatformEnv
from gym_platform.envs.platform_env import Constants
from gym_platform.envs.platform_env import ACTION_LOOKUP
from gym_platform.envs.platform_env import RUN, HOP, LEAP, JUMP

# import basic libraries
import numpy as np

class PlatformEnvExtended(PlatformEnv):

    def step(self, action, return_every=10):
        """
        Take a full, stabilised update. Returns all intermediate states (not only on platform)
        Parameters
        ----------
        action (ndarray) :
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            terminal (bool) :
            info (dict) : contains the intermediate states as rgb arrays
        """

        # Initialize values

        terminal = False
        running = True
        act_index = action[0] 
        act = ACTION_LOOKUP[act_index]
        param = action[1][act_index][0]
        param = np.clip(param, Constants.PARAMETERS_MIN[act_index], Constants.PARAMETERS_MAX[act_index])

        steps = 0
        difft = 1.0
        reward = 0.
        self.xpos = self.player.position[0]

        # initialize array containing the intermediate states
        all_interm_states = []
        # initialize array containing whether the states are on-platform
        on_platform_array = []

        # update until back on platform
        while running:
            reward, terminal = self._update(act, param)

            if steps%return_every == 0:
                self._initialse_window()
                self._draw_render_states("rgb_array")
                all_interm_states.append(self._get_image()) # append intermediate states

            if act == RUN:
                difft -= Constants.DT
                running = difft > 0
            elif act in [JUMP, HOP, LEAP]:
                running = not self._on_platforms()

            if terminal:
                on_platform_array.append(1) # if terminal (on platform) append 1
                running = False
            else:
                on_platform_array.append(0) # else append 0
            steps += 1

        state = self.get_state()
        obs = (state, steps)

        # create an info dict containing the intermediate states
        # (on_platform_array is not used here. Not needed. It should be deleted)
        info = {"interm_states" : all_interm_states}
        return obs, reward, terminal, info