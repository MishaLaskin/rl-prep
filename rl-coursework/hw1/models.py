import torch
from torch import nn


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
