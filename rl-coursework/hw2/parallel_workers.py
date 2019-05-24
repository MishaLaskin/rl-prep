import multiprocessing as mp
import random
import string
import numpy as np
import gym
import time

random.seed(123)

out = mp.Queue()


def one_rollout(output=False):
    env = gym.make('CartPole-v0')
    done = False
    all_obs, all_acts, all_rews = [], [], []

    obs = env.reset()

    def sample_act():
        return env.action_space.sample()

    def update_path(o, a, r):
        all_obs.append(o)
        all_acts.append(a)
        all_rews.append(r)
    print('Starting rollout')
    while not done:
        act = sample_act()
        obs2, rew, done, _ = env.step(act)
        update_path(obs, act, rew)
        obs2 = obs
        if done:
            break

    path = {"observations": all_obs, "actions": all_acts, "rewards": all_rews}
    if output:
        output.put(path)


def serial_rollout(n):
    paths = []
    for _ in range(n):
        path = one_rollout()
        paths.append(path)
    return paths


processes = [mp.Process(target=one_rollout, args=(out,)) for _ in range(4)]

if __name__ == "__main__":

    start = time.time()
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    res = [out.get() for p in processes]
    end = time.time()

    print('Async time', end - start)

    start = time.time()
    paths = serial_rollout(4)
    end = time.time()
    print('Sync time', end - start)
