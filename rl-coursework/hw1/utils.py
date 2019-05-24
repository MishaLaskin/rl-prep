from collections import namedtuple
import json
import gym
import numpy as np
import pickle
import os
from load_policy import load_policy


def dict2tuple(dictionary):
    return namedtuple('Hyperparameters', dictionary.keys())(**dictionary)


def load_params(path, as_dict=False):
    try:
        with open(path) as f:
            params = json.load(f) if as_dict else dict2tuple(json.load(f))
    except:
        raise Exception(path + " not found")
    return params


def init_env(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return env, obs_dim, act_dim


def load_data(env_name, act_dim, n_rollouts):
    """ 
    loads data from .pkl file 
    """
    filename = './expert_data/'+env_name+'-'+str(n_rollouts)+'.pkl'
    try:

        with open(filename, 'rb') as f:
            data = pickle.load(f)
        X_train = data['observations']
        y_train = data['actions'].reshape(-1, act_dim)

        return (X_train, y_train)

    except:
        print('Data not found for:')
        print(filename)
        print('Generating data with expert rollout instead')
        return False


def load_dagger_data(act_dim):
    """ 
    loads data from .pkl file 
    """
    params = load_params('config.json')
    filename = './dagger_data/' + params.env_name + '-' + \
        str(params.dagger_iters) + 'iters-' + \
        str(params.dagger_rollouts) + 'rollouts.pkl'
    try:

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        X_train = data['observations']
        y_train = data['actions'].reshape(-1, act_dim)

        return (X_train, y_train)

    except:
        raise Exception(
            "DAgger data not found, must be a bug. Looking for: "+filename)


def print_title_and_parameters(title, params):
    m = 7
    n = len(title)+2*m
    print('='*n)
    print('|' + ' ' * (m - 1) + title + ' ' * (m - 1) + '|')
    print('=' * n)
    for k, v in zip(params._fields, params):
        key_val = k + ' : ' + str(v)
        j = n - len(key_val)-3
        print('| '+key_val + ' '*j+'|')
    print('='*n)


def load_results(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(data, directory, filename):

    file = os.path.join(directory, filename)
    with open(file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def get_expert_policy():
    # read in parameters from json file
    params = load_params('config.json')
    env_name = params.env_name
    filename = params.expert_path + '/' + env_name + '.pkl'

    print('loading and building expert policy:',
          './experts/' + env_name + '.pkl')
    policy_fn = load_policy(filename)
    print('expert policy loaded and built')
    return policy_fn


if __name__ == "__main__":
    # test that load_results works
    filename = 'results/BehavioralCloning-HalfCheetah-v2-20rollouts.pkl'
    data = load_results(filename)
