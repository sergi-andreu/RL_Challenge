# RL challenge in the Platform Environment
# Code author: Sergi Andreu

# import basic libraries
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import shutil
import numpy as np
import torch

# import the platform environment and its utilities
import gym
import gym_platform
from gym_platform.envs.platform_env import PlatformEnv
from extend_platform import PlatformEnvExtended
from gym_platform.envs.platform_env import Constants

PARAMETERS_MAX = Constants.PARAMETERS_MAX # Max parameters for each action

# Register the platform environment and initialize it
env_cur = "Platform-v0"
register_env(env_cur, lambda config: PlatformEnv())
env = gym.make('Platform-v0')

class preprocess_observations():
    """
    Class for preprocessing the observations
    """
    def __init__(self, no_steps=False):
        self.no_steps = no_steps
        self.max_steps = 200

    def transform(self, observation):
        if self.no_steps:
            return torch.from_numpy(observation[0]).type(torch.float32)
        else:
            observation = np.append(observation[0], observation[1]/self.max_steps)
            return torch.from_numpy(observation).type(torch.float32)

class preprocess_actions():
    """
    Class for preprocessing the actions
    """
    def __init__(self, n_discretization=3):
        self.n_discretization = n_discretization
        self.parameters_max = PARAMETERS_MAX

    def transform(self, action, one_hot=False):
        # Action to int / one-hot
        chosen_action = action[0]
        chosen_value = action[1][chosen_action][0]
        max_value = self.parameters_max[chosen_action]

        reshaped_action = np.zeros((3*self.n_discretization), dtype=np.int16)
        action_bin = (self.n_discretization -1)* chosen_value / max_value
        action_bin = round(action_bin)
        action_idx = self.n_discretization*chosen_action + action_bin
        reshaped_action[action_idx] = 1

        if one_hot: return torch.from_numpy(reshaped_action).type(torch.float32)
        else: return torch.tensor([action_idx])

    def reverse_transform(self, prep_action, one_hot=False):
        # one_hot / int to environment-friendly action
        try: prep_action = prep_action.numpy()
        except: pass

        if one_hot: prep_action = np.argmax(prep_action)
        chosen_action = int(prep_action / self.n_discretization)
        action_intensity = prep_action % self.n_discretization

        max_value = self.parameters_max[chosen_action]

        chosen_value = max_value * (action_intensity / (self.n_discretization -1 ))

        act = tuple([chosen_value] if i==chosen_action else [0.0] for i in range(3))
        return (chosen_action, act)


if __name__ == "__main__":

    env = gym.make(env_cur)
    env.reset()

    try_action = ((1), (([20], [10], [10])))

    act_pre = preprocess_actions(n_discretization=5)
    transformed_act = act_pre.transform(try_action)

    act = 1
    rev_act = act_pre.reverse_transform(act)

    print(rev_act)

    #env.step(try_action)
    #print(env.get_state())





