# RL challenge in the Platform Environment
# Code author: Sergi Andreu

# Load packages
import numpy as np
import gym
from gym_platform.envs.platform_env import PlatformEnv
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, EGreedyAgent
from Networks import TestNetwork, BigNetwork, BiggerNetwork, DuelingBigNetwork
from Experience import Experience, ExperienceReplayBuffer

from preprocessors import preprocess_actions, preprocess_observations

from extend_platform import PlatformEnvExtended

from plotters import training_plotter, view_agent, play_agent, create_gif

import json # for loading the config dictionary

from utils import linear_epsilon, exponential_epsilon, running_average

class DQNTrainer():

    def __init__(self, config, env):
        # Load the parameters from a configuration dictionary file
        parameters = config["Parameters"]
        hyperparameters = config["Hyperparameters"]
        disp_parameters = config["Display parameters"]

        self.N_episodes = parameters["N_episodes"] # Number of episodes to load per iteration
        self.discount_factor = parameters["discount_factor"]
        self.n_ep_running_average = parameters["n_ep_running_average"]
        self.n_discretization = parameters["n_discretization"]
        self.no_steps = parameters["no_steps"]
        self.dim_state = parameters["dim_state"]

        self.eval_episodes = parameters["eval_episodes"]
        self.eval_every = parameters["eval_every"]

        self.n_actions = 3*self.n_discretization
        if not self.no_steps: 
            self.dim_state += 1

        self.hidden_size_1 = parameters["hidden_size_1"]
        self.hidden_size_2 = parameters["hidden_size_2"]

        self.N = hyperparameters["N"]
        self.C = hyperparameters["C"]
        self.N_buffer = hyperparameters["N_buffer"]
        self.lr = hyperparameters["lr"]
        self.e_min = hyperparameters["e_min"]
        self.e_max = hyperparameters["e_max"]
        self.CER = hyperparameters["CER"]
        self.exponential_decay = hyperparameters["exponential_decay"]
        self.experimental_power_reward = hyperparameters["experimental_power_reward"]

        self.show_each = disp_parameters["show_each"]

        self.env = env

        self.network = DuelingBigNetwork(self.dim_state, self.n_actions, hidden_size_1=self.hidden_size_1, hidden_size_2=self.hidden_size_2)
        self.target_network = copy.deepcopy(self.network)

        self.random_agent = RandomAgent(self.n_actions)
        self.greedy_agent = EGreedyAgent(self.n_actions)

        self.buffer = ExperienceReplayBuffer(self.N_buffer)

        self.prep_action = preprocess_actions(n_discretization=self.n_discretization)
        self.prep_obs = preprocess_observations(no_steps=self.no_steps)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        self.save_mode = False
        self.save_every = 100
        self.save_location = "trained_models"

        self.logger = {"episode_reward_list" : [], "episode_number_of_steps" : [], "config" : config}

        self.val_logger = {"train iteration": [], "episode reward" : [], "episode n steps" : [], 
                            "n wins": [], "config" : config}
        self.n_train_iter = 0

        self.wandb_mode = False


    def fill_buffer(self):

        state = self.env.reset()

        for i in range(self.N_buffer):

            action = self.random_agent.forward(state)
            action = self.prep_action.reverse_transform(action)

            # Get next state and reward.
            next_state, reward, done, _ = self.env.step(action)

            if self.experimental_power_reward:
                reward = reward**3

            action = self.prep_action.transform(action)

            # Add the experience to the buffer
            exp = Experience(self.prep_obs.transform(state), action, reward, self.prep_obs.transform(next_state), done)
            self.buffer.append(exp)

            if done:
                state = self.env.reset()
            else:
                state = next_state

    def train(self, reset_buffer = True):

        if reset_buffer: self.fill_buffer()

        EPISODES = trange(self.N_episodes + 1, desc='Episode: ', position=0, leave=True)
        t_c = 0
        for i in EPISODES:
            self.n_train_iter +=1
            # Reset enviroment data and initialize variables
            done = False
            state = self.env.reset()
            #state = prep_obs.transform(state)

            if self.exponential_decay:
                epsilon = exponential_epsilon(self.e_min, self.e_max, i+1, self.N_episodes)
            else:
                epsilon = linear_epsilon(self.e_min, self.e_max, i+1, self.N_episodes)

            total_episode_reward = 0.
            t = 0

            while not done:

                action = self.greedy_agent.forward(self.prep_obs.transform(state), self.network, epsilon)
                action = self.prep_action.reverse_transform(action)

                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise
                next_state, reward, done, _ = self.env.step(action)

                if self.experimental_power_reward:
                    reward = reward**3

                action = self.prep_action.transform(action)

                # Add the experience to the buffer
                exp = Experience(self.prep_obs.transform(state), action, reward, self.prep_obs.transform(next_state), done)
                self.buffer.append(exp)

                ### TRAINING ###
                
                # Randomly pick N samples from the buffer of experiences 
                s_batch, a_batch, r_batch, s_next_batch, d_batch = self.buffer.sample_batch(n=self.N, CER=self.CER)

                # Compute the target output for the experiences
                #s_next_tensor = torch.tensor(s_next_batch, requires_grad=False, dtype=torch.float32)
                s_next_tensor = torch.stack(s_next_batch)
                s_next_tensor.requires_grad = False
                target_tensor, _ = torch.max(self.target_network(s_next_tensor), axis=1)

                # Compute the target
                dones_tensor = torch.tensor(d_batch, requires_grad=False, dtype=torch.float32)
                rewards_tensor = torch.tensor(r_batch, requires_grad=False, dtype = torch.float32)
                y_target = rewards_tensor + self.discount_factor*(1-dones_tensor)*target_tensor

                # Set gradients to 0
                self.optimizer.zero_grad()

                # Compute the real output for the experience
                #s_tensor = torch.tensor(s_batch, requires_grad=True, dtype=torch.float32)
                s_tensor = torch.stack(s_batch)
                s_tensor.requires_grad = False

                #a_tensor = torch.tensor(a_batch, dtype=torch.float32)

                a_tensor = torch.stack(a_batch)
                a_tensor = a_tensor.type(torch.int64)
                a_tensor.requires_grad = False

                q_values = self.network(s_tensor)

                #q_val = q_values.gather(1, a_tensor.unsqueeze(1)).squeeze(-1)
                q_val = torch.gather(q_values, 1, a_tensor.view(-1,1)).view(-1)

                # Compute loss function
                loss = torch.nn.functional.mse_loss(q_val, y_target)

                # Compute gradient
                loss.backward()

                # Clip gradient to 1
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)

                # Perform backward pass (backpropagation)
                self.optimizer.step()

                # Update episode reward
                total_episode_reward += reward

                # Update state for next iteration
                state = next_state
                t += 1
                t_c += 1

                # If C steps have passed, set the target network equal to the main network
                if t_c % self.C == 0:
                    self.target_network = copy.deepcopy(self.network)

                if self.save_mode:
                    if i % self.save_every == 0 and t == 1:
                        # Storing the model
                        torch.save(self.network, f"{self.save_location}/neural-network-1-{str(i).zfill(4)}.pth")

            # Append episode reward and total number of steps
            self.logger["episode_reward_list"].append(total_episode_reward)
            self.logger["episode_number_of_steps"].append(t)

            # Close environment
            #self.env.close()

            # Updates the tqdm update bar with fresh information
            # (episode number, total reward of the last episode, total number of Steps
            # of the last episode, average reward, average number of steps)
            EPISODES.set_description(
                "Episode {} - Total episode reward: {} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward,
                running_average(self.logger["episode_reward_list"], self.n_ep_running_average)[-1],
                running_average(self.logger["episode_number_of_steps"], self.n_ep_running_average)[-1]))

            if i%self.eval_every == 0:
                self.evaluate()

    def evaluate(self):

        total_reward = 0.
        t = 0
        n_wins = 0

        for i in range(self.eval_episodes):
            total_episode_reward = 0.
            # Reset enviroment data and initialize variables
            done = False
            state = self.env.reset()
            #state = prep_obs.transform(state)

            t_temp = 0
            
            while not done and t_temp < 50:

                action = self.greedy_agent.forward(self.prep_obs.transform(state), self.network, 0)
                action = self.prep_action.reverse_transform(action)

                next_state, reward, done, _ = self.env.step(action)
                action = self.prep_action.transform(action)

                # Update episode reward
                total_episode_reward += reward

                # Update state for next iteration
                state = next_state
                t += 1
                t_temp += 1

            #self.env.close()

            if total_episode_reward >  1 - 1e-3:
                n_wins += 1
            
            total_reward += total_episode_reward

        mean_wins = n_wins / self.eval_episodes
        mean_reward = total_reward / self.eval_episodes
        mean_t = t / self.eval_episodes

        # Append episode reward and total number of steps
        self.val_logger["train iteration"].append(self.n_train_iter)
        self.val_logger["episode reward"].append(mean_reward)
        self.val_logger["episode n steps"].append(mean_t)
        self.val_logger["n wins"].append(mean_wins)

        if self.wandb_mode:
            wandb.log({"train iteration" : self.n_train_iter, "episode reward": mean_reward,
                    "episode n steps" : mean_t, "n wins" : mean_wins
            })


import random
def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import warnings
warnings.filterwarnings("ignore")
            
if __name__ == "__main__":

    f = open('config.json')
    config = json.load(f)

    modes = config["Modes"]

    wandb_mode = modes["wandb_mode"]
    save_mode =  modes["save_mode"]
    plot_mode = modes["plot_mode"]
    observation_mode = modes["observation_mode"]
    play_mode = modes["play_mode"]
    seed_mode = modes["seed_mode"]

    if seed_mode:
        set_all_seeds(0)

    if wandb_mode: 
        import wandb
        wandb.init(project="RLChallenge", entity="sergi-andreu")

        save_mode = False
        observation_mode = False
        play_mode = False
        plot_mode = False

        """
        wandb.config = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 128
        }
        """

        config["Hyperparameters"]["N"] = wandb.config["N"]
        config["Hyperparameters"]["C"] = wandb.config["C"]
        config["Hyperparameters"]["N_buffer"] = wandb.config["N_buffer"]
        config["Hyperparameters"]["lr"] = wandb.config["lr"]
        config["Hyperparameters"]["exponential_decay"] = wandb.config["exponential_decay"]
        config["Hyperparameters"]["e_max"] = wandb.config["e_max"]

        config["Parameters"]["hidden_size_1"] = wandb.config["hidden_size_1"]
        config["Parameters"]["hidden_size_2"] = wandb.config["hidden_size_2"]

    # Import and initialize the Platform Environment
    env = gym.make("Platform-v0", disable_env_checker = True)

    if seed_mode: env.seed(0)

    env.reset()

    trainer = DQNTrainer(config, env)
    trainer.wandb_mode = wandb_mode
    trainer.save_mode = save_mode

    """
    print("N", trainer.N)
    print("C", trainer.C)
    print("N_buffer", trainer.N_buffer)
    print("Exponential decay", trainer.exponential_decay)
    """

    if not observation_mode:
        ### Training process
        trainer.train()

        if plot_mode:
            training_plotter(trainer.logger)
            training_plotter(trainer.val_logger)

    if observation_mode:
        env = PlatformEnvExtended()
        sav_file = "trained_models/neural-network-1-1500.pth"
        #view_agent(env, sav_file, n_discretization=trainer.n_discretization, no_steps=trainer.no_steps)
        create_gif(sav_file,  n_discretization=trainer.n_discretization, no_steps=trainer.no_steps, explainable=True,
                    only_failed=True, n_iterations=500)

    if play_mode:
        play_agent(env, n_discretization=5, no_steps=True)
