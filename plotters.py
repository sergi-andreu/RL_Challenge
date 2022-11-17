# RL challenge in the Platform Environment
# Code author: Sergi Andreu

# import basic libraries
import pygame
import torch
import time
import numpy as np

# import matplotlib
import matplotlib.pyplot as plt

# import utils from matplotlib (to create "advanced" subplots, animations, ...)
from matplotlib import animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

# import utils and preprocessors
from utils import running_average
from preprocessors import preprocess_actions, preprocess_observations

# import extended platform to ease plotting
from extend_platform import PlatformEnvExtended

# import library to create .mp4 video files from an animated gif
import ffmpy

def training_plotter(dict, with_config_dict = True, labels=None, save = False, save_name = "figure.png"):
    """
    Function plotting the training evolution

    Args:
        - dict: contains the logger dictionary (data to plot)
        - with_config_dict: boolean indicating whether the config parameters are in the dict
        - labels: labels to use
        - save: boolean used to save or not the figure
        - save_name: if save, filename

    Returns:
        - None
        displays a matplotlib plot 
    """

    keys = list(dict.keys())
    if with_config_dict: keys.remove("config") # we do not want to plot the configuration parameters
    n_plots = len(keys)

    if with_config_dict: n_ep_running_average = dict["config"]["Parameters"]["n_ep_running_average"]
    else: n_ep_running_average = 50

    fig, ax = plt.subplots(nrows=1, ncols=n_plots, figsize=(16, 9))

    for i, key in enumerate(keys):
        assert type(dict[key]) is list, "The given dictionary for the plotter is not appropriate (values should be arrays)"
        N_episodes = len(dict[key])

        ax[i].plot([i for i in range(1, N_episodes+1)], dict[key], label=key)
        ax[i].plot([i for i in range(1, N_episodes+1)], running_average(
            dict[key], n_ep_running_average), label='Avg. {key}')

        ax[i].set_xlabel('Episodes')
        ax[i].set_ylabel(key)
        ax[i].set_title(f'{key} vs Episodes')
        ax[i].legend()
        ax[i].grid(alpha=0.3)

    if save: plt.savefig(save_name)
    plt.show()


def get_saliency(model, input):

    '''
    Returns the 'saliency map 
        Parameters:
            - model: agent model
            - input: observation
        Returns:
            - saliency map (gradients of the input wrt the chosen action)
    '''

    # Unsqueeze the vector before feedforwarding-it with the model
    try: input_tensor = input.unsqueeze(0)
    except: input_tensor = torch.from_numpy(input).unsqueeze(0)

    input_tensor.requires_grad_() # Set require_grad to True

    model.eval() # use evaluation mode (probably not required here)
    output = model(input_tensor)
    output_max = output[0, output.argmax()] # get maximum action
    output_max.backward() # perform backward pass

    saliency = input_tensor.grad.data.abs() # get the absolute gradients

    return saliency


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', fps=10, explainable = False,
                        action_frames=None, saliency_frames=None):

    '''
    Saves the input frames as a gif
        Parameters:
            - frames: frames of the game environment
            - path: path where to save the gif
            - filename: name of the saved gif
            - fps: frames per second of the gif
            - explainable: boolean indicating whether to plot q_values and saliency, or only the game
        Returns:
            - None
            saves a gif
    '''

    if explainable:
        # Plot q values and saliency 

        fig = plt.figure(figsize=(frames[0].shape[1] / 72.0, 2*frames[0].shape[0] / 72.0), dpi=72,
                        constrained_layout=True)
        spec = gridspec.GridSpec(2, 2, figure=fig)

        # Create subplots
        ax0 = fig.add_subplot(spec[0,:])
        ax0.invert_xaxis()
        ax0.axis("off")
        ax10 = fig.add_subplot(spec[1,0])
        ax11 = fig.add_subplot(spec[1,1])


        n_actions = np.shape(action_frames)[1]

        par1 = ax10.secondary_xaxis("top")
        par1.set_xticks([n_actions/6 + i*n_actions/3 for i in range(3)])
        par1.set_xticklabels(["Run", "Hop", "Leap"])

        action_xaxis = [i for i in range(n_actions)]
        saliency_xaxis = [i for i in range(9)]

        action_xaxis_labels = [i%int(n_actions/3) for i in range(n_actions)]
        colors_dict = {0 : "r", 1 : "g", 2 : "b"}
        action_colors = [colors_dict[int(3*i/n_actions)] for i in range(n_actions)]

        saliency_xaxis_labels = ["Player position", "Player velocity", "Enemy position", "Enemy dx",
                                 "Platform wd1", "Platform wd2", "Platform gap", "Platform pos", "Platform diff"]
        saliency_colors = ["orange", "orange", "yellow", "yellow", "purple", "purple", "purple", "purple", "purple"]

        ax0.imshow(frames[0])
        ax10.bar(action_xaxis, action_frames[0])
        ax11.bar(saliency_xaxis, saliency_frames[0])

        axs = [ax0, ax10, ax11, par1]

        def run(i):
            axs[0].clear()
            axs[0].imshow(frames[i])
            axs[0].invert_xaxis()
            axs[0].axis("off")
            axs[1].clear()
            axs[1].bar(action_xaxis, action_frames[i], tick_label=action_xaxis_labels, color=action_colors)
            axs[1].set_yticks([])
            axs[1].set_ylabel("Q values")
            ymin = np.min(action_frames[i])
            ymax = np.max(action_frames[i])
            axs[1].set_ylim(ymin, 1.2*ymax)
            axs[1].set_xlabel("Intensity")

            axs[2].clear()
            axs[2].bar(saliency_xaxis, saliency_frames[i], tick_label=saliency_xaxis_labels, color=saliency_colors)
            axs[2].set_yticks([])
            axs[2].set_ylabel("Saliency")
            #axs[2].yaxis.set_label_position("right")
            axs[2].tick_params(labelrotation=45)
            ymin = np.min(saliency_frames[i])
            ymax = np.max(saliency_frames[i])
            axs[2].set_ylim(ymin, 1.2*ymax)

            for ax in [axs[1], axs[2], axs[3]]:
                for spine in ax.spines.values():
                    spine.set_color("white")

            axs[3] = axs[1].secondary_xaxis("top")

            axs[3].set_xticks([n_actions/6 + i*n_actions/3 for i in range(3)])
            axs[3].set_xticklabels(["Run", "Hop", "Leap"])
            axs[3].tick_params(direction="in", pad = -20, length=0, labelsize=15)

            axs[1].spines["top"].set_color("white")

            return axs

        anim = animation.FuncAnimation(fig, run, frames = len(frames))
        print("Before anim.save")
        anim.save(path + filename, fps=fps)

    else:
        # only plot game
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.gca().invert_xaxis()
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames))
        print("Before anim.save")
        anim.save(path + filename, fps=fps)

def view_agent(env, sav_file, n_discretization=3, no_steps=True):

    '''
    Display the trained agent
        Parameters:
            - env: initialized gym environment
            - sav_file: checkpoint file saved from training
            - n_discretization: discretization number used when training the agent
            - no_steps: whether the agent has been trained with the steps in the obs
        Returns:
            - None
            displays the game with the actions given by the agent
    '''

    prep_action = preprocess_actions(n_discretization=n_discretization)
    prep_obs = preprocess_observations(no_steps=no_steps)

    parameters_max = prep_action.parameters_max

    model = torch.load(sav_file)
    env.metadata["render_modes"] = "human"

    obs = env.reset()
    ttl_reward = 0
    n_step = 20

    for step in range(n_step):
        q_values = model(prep_obs.transform(obs))
        _, action = torch.max(q_values, axis=0)

        saliency = get_saliency(model, prep_obs.transform(obs))

        obs, reward, done, info = env.step(prep_action.reverse_transform(action))
        ttl_reward += reward

        env.render()
        time.sleep(2) # make pygame slower 

        if done:
            # reward at the end of episode
            print("reward", ttl_reward)
            obs = env.reset()
            ttl_reward = 0
            
    env.close() 

def create_gif(sav_file, n_discretization=3, no_steps=True, only_failed=False, n_iterations=10, explainable=False, save_video=True):

    '''
    Creates a gif to visualize the agent
        Parameters:
            - sav_file: checkpoint file saved from training
            - n_discretization: discretization number used with the agent
            - no_steps:
            - only_failed: whether to only show "failed" episodes
            - n_iterations: number of iterations (actions) to perform
            - explainable: whether to plot q values and saliency 
            - save_video: whether to convert the gif into a mp4 video file
        Returns:
            None
    '''
    # initialize the preprocessors
    prep_action = preprocess_actions(n_discretization=n_discretization)
    prep_obs = preprocess_observations(no_steps=no_steps)

    # load the specified checkpoint
    model = torch.load(sav_file)

    # load the extended version of the platform environment
    # containing the intermediate rgb arrays in the info dict
    env = PlatformEnvExtended()
    env.metadata["render_modes"] = []
    obs = env.reset()

    ttl_reward = 0

    frames = []

    if explainable:
        frames_actions = []
        frames_saliency = []
    else:
        frames_actions = None
        frames_saliency = None

    if only_failed:
        idx_good = 0
        idx_temp = 0

    for step in range(n_iterations):
        q_values = model(prep_obs.transform(obs))
        _, action = torch.max(q_values, axis=0)

        saliency = get_saliency(model, prep_obs.transform(obs))

        obs, reward, done, info = env.step(prep_action.reverse_transform(action))

        ttl_reward += reward

        interm_states = info["interm_states"]
        length = len(interm_states)
        for inter in interm_states:
            frames.append(inter)
            if only_failed: idx_temp+=1
            if explainable:
                frames_actions.append(q_values.detach().numpy())
                frames_saliency.append(saliency.detach().numpy()[0])

        if done:
            # reward at the end of episode
            print("reward", ttl_reward)
            obs = env.reset()

            if only_failed:
                if ttl_reward > 1 -1e-3:
                    idx_good += idx_temp
                    idx_temp = 0
                else:
                    print(f"Idx good {idx_good}")
                    print(f"Len frames {len(frames)}")
                    frames = frames[idx_good:] 
                    print(f"New length frames {len(frames)}")
                    env.close()
                    break
            ttl_reward = 0
    env.close()
    pygame.quit()

    save_frames_as_gif(frames, fps=5, explainable=explainable, action_frames=frames_actions, saliency_frames=frames_saliency)
    
    if save_video:
        ff = ffmpy.FFmpeg(
            inputs = {"gym_animation.gif" : None},
            outputs = {"gym_animation.mp4" : None}
        )
        ff.run()

def actions_to_names(action, n_discretization, max_values):
    '''
    Returns a 'name' for an action given as an int
        Parameters:
            - action: action (as int)
            - n_discretization
            - max_values: max intensity values of the platform env
        Returns:
            - action name
    '''
    try: action = action.numpy()
    except: pass

    action_dict = {0: "run", 1: "hop", 2: "leap"}

    action_type = int(action / n_discretization)
    action_intensity = action % n_discretization

    action_string = f"{action_dict[action_type]} with {(action_intensity/(n_discretization-1))*max_values[int(action / n_discretization)]} intensity"

    return action_string

def play_agent(env, n_discretization=3, no_steps=True):

    '''
    Play with the actions choosen by the player (inputting them as ints in the console)
        Parameters:
            env:
            n_discretization:
            no_steps:
        Returns:
            None
    '''

    prep_action = preprocess_actions(n_discretization=n_discretization)
    prep_obs = preprocess_observations(no_steps=no_steps)

    parameters_max = prep_action.parameters_max

    env.metadata["render_modes"] = "human"

    obs = env.reset()
    ttl_reward = 0
    n_step = 20

    for step in range(n_step):
        action = int(input("Choose the action (as integer): "))

        obs, reward, done, info = env.step(prep_action.reverse_transform(action))
        ttl_reward += reward

        print(f"Action without processing: {action}")
        print(f"Action with processing {prep_action.reverse_transform(action)}")
        print(f"Action name {actions_to_names(action, n_discretization, parameters_max)}")

        env.render()
        time.sleep(1)

        if done:
            # reward at the end of episode
            print("reward", ttl_reward)
            obs = env.reset()
            ttl_reward = 0
            
    env.close() 

def view_agent_ppo(env_cur, agent, sav_file):

    '''
    Display the PPO trained agent. 
        Parameters:
            env_cur: gym environment
            agent: trained RL agent
            sav_file: checkpoint file saved from training
        Returns:
            None
    '''

    # run the policy
    agent.restore(sav_file)
    env = gym.make(env_cur)
    env.metadata["render_modes"] = "human"
    env.metadata["render_fps"] = 10

    obs = env.reset()
    ttl_reward = 0
    n_step = 200

    for step in range(n_step):
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        ttl_reward += reward

        env.render()
        time.sleep(0.001)
        if done:
            # reward at the end of episode
            print("reward", ttl_reward)
            obs = env.reset()
            ttl_reward = 0
            
    env.close()
