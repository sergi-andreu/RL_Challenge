# RL challenge in the Platform Environment
# Code author: Sergi Andreu

# Load basic packages
import numpy as np
import torch

# Load constants (fixed parameters) of the platform environment
from gym_platform.envs.platform_env import Constants

# Load gym spaces
from gym import spaces

# Set device (cuda if possible; else cpu)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#dev = torch.device("cpu") 

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of possible actions
                            in the platform case, it is 3 (run, hop, leap) 
                            times the discretization parameter (possible intensities)

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''

        # compute random action (uniformly random integer)
        act = np.random.randint(0, self.n_actions)

        self.last_action = act
        return self.last_action


class EGreedyAgent(Agent):
    ''' Agent taking actions epsilon-greedy, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(EGreedyAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray, Q_network, epsilon) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices with probability epsilon. 

            Compute an action greedy based on Q_network values 
            with probability 1-epsilon

            Returns:
                action (int): the random action
        '''
        if np.random.rand() < epsilon:
            # with probability epsilon, chose a random action
            act = np.random.randint(0, self.n_actions)
            self.last_action = act

            return self.last_action
            
        # make the state a tensor and set it to the proper device
        state_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32, device=dev)

        # Compute Forward output of the network
        q_values = Q_network(state_tensor)
        _, action = torch.max(q_values, axis=0)
        # the chosen action is the one with the maximum q value
        self.last_action = action.item()

        return self.last_action
