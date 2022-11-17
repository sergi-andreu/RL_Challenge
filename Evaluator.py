# RL challenge in the Platform Environment
# Code author: Sergi Andreus

import torch
import gym
import gym_platform
import matplotlib.pyplot as plt

from preprocessors import preprocess_actions, preprocess_observations

from plotters import get_saliency


def ModelEvaluator(env, model, eval_episodes=10, n_discretization=5, no_steps=True):

    # Instanciate the prepocessors:
    prep_action = preprocess_actions(n_discretization=n_discretization)
    prep_obs = preprocess_observations(no_steps=no_steps)


    total_reward = 0.
    t = 0
    t_wins = 0
    n_wins = 0

    confidence_action = 0.
    confidence_saliency = 0.

    for i in range(eval_episodes):
        total_episode_reward = 0.
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()

        t_temp = 0

        c_act_temp = 0.
        c_sal_temp = 0.
        
        while not done and t_temp < 50:

            q_values = model(prep_obs.transform(state))
            q_val_max, action = torch.max(q_values, axis=0)

            con_q = q_val_max / q_values.sum()
            c_act_temp += con_q.detach().numpy()

            saliency = get_saliency(model, prep_obs.transform(state))[0]
            sal_val_max, sal_max = torch.max(saliency, axis=0)

            con_sal = sal_val_max / saliency.sum()
            c_sal_temp += con_sal.detach().numpy()

            action = prep_action.reverse_transform(action)

            next_state, reward, done, _ = env.step(action)
            action = prep_action.transform(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1
            t_temp += 1

        if total_episode_reward >  1 - 1e-3:
            t_wins += t_temp
            n_wins += 1
        
        total_reward += total_episode_reward
        confidence_action += c_act_temp / t_temp
        confidence_saliency += c_sal_temp / t_temp

    mean_wins = n_wins / eval_episodes
    mean_reward = total_reward / eval_episodes
    mean_t = t / eval_episodes

    if n_wins > 0: mean_t_wins = t_wins / n_wins
    else: mean_t_wins = 0

    mean_con_act = confidence_action / eval_episodes
    mean_con_sal = confidence_saliency / eval_episodes

    return mean_wins, mean_reward, mean_t, mean_t_wins, mean_con_act, mean_con_sal

if __name__ == "__main__":
    env = gym.make("Platform-v0", disable_env_checker = True)
    checkpoints = [f"trained_models/neural-network-1-{str(i*100).zfill(4)}.pth" for i in range(16)]

    WINS, REWARDS, TS, TWINS, CONACT, CONSAL = [], [], [], [], [], []

    for cp in checkpoints:

        model = torch.load(cp)
        wins, rewards, ts, twins, conact, consal = ModelEvaluator(env, model, eval_episodes=100)

        WINS.append(wins)
        REWARDS.append(rewards)
        TS.append(ts)
        TWINS.append(twins)
        CONACT.append(conact)
        CONSAL.append(consal)

    xaxis = [i*100 for i in range(16)]

    fig, axes = plt.subplots(nrows=3, ncols=2)

    axes[0,0].plot(xaxis, WINS)
    axes[0,0].set_title("Probability of win")

    axes[0,1].plot(xaxis, REWARDS)
    axes[0,1].set_title("Mean total reward")

    axes[1,0].plot(xaxis, TS)
    axes[1,0].set_title("Mean time")

    axes[1,1].plot(xaxis, TWINS)
    axes[1,1].set_title("Mean winning time")

    axes[2,0].plot(xaxis, CONACT)
    axes[2,0].set_xlabel("Training iteration")
    axes[2,0].set_title("Confidence on action")

    axes[2,1].plot(xaxis, CONSAL)
    axes[2,1].set_xlabel("Training iteration")
    axes[2,1].set_title("Confidence on features")

    plt.show()