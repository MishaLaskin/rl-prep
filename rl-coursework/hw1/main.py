import torch
from torch.utils.data import TensorDataset
import json
import os
import pickle
from utils import load_params, init_env, load_data, print_title_and_parameters
from utils import dict2tuple, save_data, load_dagger_data, get_expert_policy
from rollouts import run_expert, run_agent
from train import fit_data, ClonePolicy
import tensorflow as tf
import numpy as np


class ImitationLearning:
    """
    Imitation Learning class inputs parameters from 
    - config.json - for a single set of parameters
    - sweep_config.json - for a parameter sweep

    ImitationLearning contains the following methods:
    - runBehavioralCloning(self) - runs behavioral cloning algorithm
    - runDAgger(self) - runs DAgger algorithm
    """

    def __init__(self, use_sweep=False):
        # load hyper parameters
        self.use_sweep = use_sweep
        self.params = load_params('config.json')
        self.env, self.obs_dim, self.act_dim = init_env(self.params.env_name)

    def run(self):
        """
        Either runs multiple times by sweeping through parameters in sweep_config.json
        or runs once by reading parameters from config.json
        """
        if self.use_sweep:
            param_sweep = load_params('sweep_config.json', as_dict=True)

            # iterate over env names
            total_runs = len(param_sweep["env_names"])
            current_run = 1
            for env_name in param_sweep["env_names"]:
                params = param_sweep.copy()
                del params["env_names"]
                params["env_name"] = env_name
                self.params = dict2tuple(params)
                self.env, self.obs_dim, self.act_dim = init_env(
                    self.params.env_name)
                print('STARTING RUN', current_run, '/', total_runs)

                self.one_run()
                current_run += 1
        else:
            self.one_run()

    def one_run(self):
        """
        One run of imitation learning for a given hyperparameter configuration
        """
        if self.params.algorithm == 'BehavioralCloning':
            title = 'Launching Behavioral Cloning'
            print_title_and_parameters(title, self.params)
        elif self.params.algorithm == 'DAgger':
            title = 'Launching DAgger'
            print_title_and_parameters(title, self.params)
        else:
            raise ValueError('algorithm must be BehavioralCloning or DAgger')

        # get results
        results = self.runBehavioralCloningOrDAgger()
        # save results from the run
        results['hyperparameters'] = self.params._asdict()
        filename = self.params.algorithm + '-' + self.params.env_name + '-' + \
            str(self.params.expert_rollouts) + 'rollouts.pkl'

        save_data(results, 'results', filename)

    def runBehavioralCloningOrDAgger(self):
        """
        If algorithm=="BehavioralCloning":
            Runs a vanilla behavioral cloning algorithm
            1. Collect expert data of form (a_i,o_i) 
            2. Train policy pi_theta (a|o) on expert data of form (a_i,o_i) for i=0,...,N

        If algorithm=="DAgger":
            Run the DAgger strategy for augmenting behavioral cloning
            1. Collect expert data of form (a_i,o_i) 
            2. Train policy pi_theta (a|o) on expert and aggregated data of form (a_i,o_i) for i=0,...,N
            3. For i in dagger_iters:
                4. Run the agent pi_theta and generates new experiences (o_i)
                5. Use expert policy pi_expert to label generated (o_i) with (a_i)
        """

        # load hyper parameters
        params = self.params
        obs_dim = self.obs_dim
        act_dim = self.act_dim

        clone_policy = ClonePolicy(obs_dim, params.h_dim, act_dim)
        # check if expert data is already in DB
        data = load_data(params.env_name, act_dim, params.expert_rollouts)

        if data:
            expert_obs, expert_actions = data
        else:
            expert_obs, expert_actions, _, _ = run_expert(
                params, act_dim, save=True, verbose=True)

        """ Behavioral Cloning """

        expert_obs = torch.tensor(expert_obs)
        expert_actions = torch.tensor(expert_actions)
        data = TensorDataset(expert_obs, expert_actions)

        r_means, r_stds, loss_means = fit_data(params, clone_policy, data)

        """ DAgger """

        if params.algorithm == "DAgger":
            print('Starting DAgger')
            # create Dagger data file
            filename = self.params.env_name + '-' + \
                str(self.params.dagger_iters) + 'iters-' + \
                str(self.params.dagger_rollouts) + 'rollouts.pkl'
            print('Saving data in: ./dagger_data/' + filename)

            """ data loading / saving for dagger """
            data = load_data(params.env_name, act_dim, params.expert_rollouts)

            def save_dagger_data(data):
                data_dict = {}
                data_dict['observations'] = data[0]
                data_dict['actions'] = data[1]
                save_data(data_dict, 'dagger_data', filename)

            save_dagger_data(data)
            """ load expert policy """
            expert_policy = get_expert_policy()

            for i in range(1, params.dagger_iters + 1):
                # rollout
                print('DAgger iteration', i, '/', params.dagger_iters)
                observations, actions, _, _ = run_agent(
                    clone_policy, params.env_name, params.dagger_rollouts)

                # overwrite actions with expert policy

                with tf.Session():
                    actions = expert_policy(observations)
                # get old data
                data = load_dagger_data(act_dim)
                expert_obs, expert_actions = data
                # aggregate data
                dagger_obs = np.vstack((expert_obs, observations))
                dagger_actions = np.vstack((expert_actions, actions))
                save_dagger_data([dagger_obs, dagger_actions])

                # load aggregated data
                print('Aggregated data. Total data is now at:',
                      dagger_obs.shape[0])

                # fit the mdoel again
                dagger_obs = torch.tensor(dagger_obs)
                dagger_actions = torch.tensor(dagger_actions)
                data = TensorDataset(dagger_obs, dagger_actions)

                r_means, r_stds, loss_means = fit_data(
                    params, clone_policy, data)

        """ Final evaluation """

        _, _, r_expert_means, r_expert_stds = run_expert(
            params, act_dim, save=True, verbose=True, evaluation=True)
        results = {}
        results['r_agent_means'] = r_means
        results['r_agent_stds'] = r_stds
        results['r_expert_means'] = r_expert_means
        results['r_expert_stds'] = r_expert_stds
        results['loss_means'] = loss_means

        return results


if __name__ == "__main__":
    use_sweep = False
    imitation_learning = ImitationLearning(use_sweep)
    imitation_learning.run()
