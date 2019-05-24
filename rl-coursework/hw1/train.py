import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from rollouts import run_agent


class ClonePolicy(nn.Module):
    def __init__(self, obs_dim, h_dim, act_dim):
        super(ClonePolicy, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, act_dim)
        )

    def forward(self, x):
        return self.layers(x)


def fit_data(params, policy, data):
    env_name = params.env_name

    optimizer = optim.Adam(policy.parameters(),
                           lr=params.learning_rate, weight_decay=params.L2)
    loss_fn = nn.MSELoss()
    loader = DataLoader(data, batch_size=params.batch_size, shuffle=True)

    batch_losses = []
    r_epoch_means = []
    r_epoch_stds = []
    loss_epoch_means = []

    for epoch in range(1, params.epochs+1):
        for batch in loader:
            obs_train, act_train = batch
            act_pred = policy(obs_train)
            loss = loss_fn(act_pred, act_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        loss_mean = np.mean(batch_losses)
        _, _, r_mean, r_std = run_agent(
            policy, env_name, params.eval_rollouts)

        r_epoch_means.append(r_mean)
        r_epoch_stds.append(r_std)
        loss_epoch_means.append(loss_mean)

        print('epoch {} loss: {:.4f}  r_mean: {:.2f}  r_std: {:.2f}'.format(
            epoch, loss_mean, r_mean, r_std))

    prefix = "DAgger-" if params.algorithm == "DAgger" else ""
    save_path = params.model_path + '/' + prefix + env_name + \
        '-' + str(params.expert_rollouts) + '.pkl'

    torch.save(policy.state_dict(), save_path)

    return r_epoch_means, r_epoch_stds, loss_epoch_means
