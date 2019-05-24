
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_results, load_params
import pandas as pd


def get_data(env_name, rollouts):

    filename = 'results/BehavioralCloning-' + \
        env_name + '-' + str(rollouts[0]) + 'rollouts.pkl'

    data = load_results(filename)
    params = data["hyperparameters"]
    # typo workaround
    data["r_expert_stds"] = data["r_expert_tds"]
    del data["hyperparameters"]
    del data["r_expert_tds"]

    for rollout in rollouts[1:]:
        filename = 'results/BehavioralCloning-' + \
            env_name + '-' + str(rollout) + 'rollouts.pkl'
        res = load_results(filename)
        data["r_agent_means_" + str(rollout)] = res["r_agent_means"]
        data["r_agent_stds_"+str(rollout)] = res["r_agent_stds"]

    n = len(data["r_agent_means"])
    epochs = list(range(len(data["r_agent_stds"])))

    df = pd.DataFrame(data)
    params = res["hyperparameters"]

    return df, epochs, params


def plot_data(env_names, rollouts):
    """
    Creates 6 subplots for each environment
    - Blue - agent trained on 200 rollouts
    - Green - agent trained on 20 rollouts
    - Red - expert agent
    """

    subplot_str = '23'
    sns.set()
    for i, env_name in enumerate(env_names):

        df, epochs, params = get_data(env_name, rollouts)

        plt.subplot(subplot_str + str(i))

        agent1 = sns.tsplot(
            time=epochs, data=df["r_agent_means"], color='#5DCDBD', linestyle='-')
        agent2 = sns.tsplot(
            time=epochs, data=df["r_agent_means_200"], color='#000c2f', linestyle='-')
        expert = sns.tsplot(
            time=epochs, data=df["r_expert_means"], color='#F65B4B', linestyle='--')
        plt.ylabel("Return")

        plt.xlabel("Epoch")
        plt.title(env_name)
    plt.subplots_adjust(wspace=0.5, hspace=.5)
    plt.show()


if __name__ == '__main__':
    filename = 'results/BehavioralCloning-Hopper-v2-20rollouts.pkl'
    # plot_data(filename)
    params = load_params('sweep_config.json')
    env_names = params.env_names
    rollouts = [20, 200]
    plot_data(env_names, rollouts)
