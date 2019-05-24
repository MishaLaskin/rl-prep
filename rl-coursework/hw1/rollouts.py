import gym
import pickle
import torch
import tf_util
import os
import numpy as np
from load_policy import load_policy
import tensorflow as tf


def run_agent(policy, env_name, n_rollouts):

    env = gym.make(env_name)
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    for _ in range(n_rollouts):
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            obs_tensor = torch.tensor(obs[None, :]).float()
            action = policy(obs_tensor).reshape(-1).detach().numpy()
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if steps >= max_steps:
                break
        returns.append(totalr)
    observations = np.array(observations).astype(np.float32)
    actions = np.array(actions).astype(np.float32)
    r_mean = np.mean(returns)
    r_std = np.std(returns)
    return observations, actions, r_mean, r_std


def run_expert(params, act_dim, save=False, verbose=False, evaluation=False):
    """
    Generates rollouts from an expert policy saved as a .pkl file in a given directory.

    args:
    - params:namedtuple - hyperparameters
    - act_dim:int - action space dimension
    - save:bool - save model
    - verbose:bool - print rollout values
    """
    # read in parameters from json file
    env_name = params.env_name
    filename = params.expert_path + '/' + env_name + '.pkl'

    print('loading and building policy:',
          './experts/' + env_name + '.pkl')
    policy_fn = load_policy(filename)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()
        env = gym.make(env_name)
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        n_rollouts = params.expert_rollouts if not evaluation else params.eval_rollouts
        for i in range(n_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if params.render:
                    env.render()
                if done or steps >= max_steps:
                    break
            returns.append(totalr)
            if verbose:
                print('iter', i, 'return', round(totalr, 2))

        if verbose:
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

        if save:
            expert_data = {'observations': np.array(observations).astype(np.float32),
                           'actions': np.array(actions).astype(np.float32)}

            file = os.path.join('expert_data', env_name +
                                '-' + str(params.expert_rollouts) + '.pkl')
            with open(file, 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
    return expert_data['observations'], expert_data['actions'].reshape(-1, act_dim), np.mean(returns), np.std(returns)
