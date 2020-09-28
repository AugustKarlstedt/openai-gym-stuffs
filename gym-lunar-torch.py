import seaborn as sns
import pandas as pd
import numpy as np
import itertools

import os
import pdb

import torch
import torch.tensor
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset

import gym

env = gym.make('LunarLander-v2')

class SimplePolicyGradient(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lin_1 = nn.Linear(8, 64)
        self.lin_2 = nn.Linear(64, 4)

    def forward(self, xb):
        xb = torch.tanh(self.lin_1(xb))
        xb = self.lin_2(xb)
        return xb

n = SimplePolicyGradient()

try:
    model = torch.load('gym-lunar-torch-model.pt')()
    n.load_state_dict(model)
except FileNotFoundError:
    pass

lr = 1e-2
opt = optim.Adam(n.parameters(), lr=lr)
batch_size = 4096
epochs = 128
render = True

for i in range(epochs):
    print(f"Epoch {i} of {epochs}: {i/epochs*100:.2f}% ", end="")

    batch_observations = []
    batch_actions = []
    batch_weights = []
    batch_episode_returns = []
    batch_episode_lengths = []

    episode_rewards = []
    observation = env.reset()

    while True:

        if render:
            env.render()

        batch_observations.append(observation)

        n.eval()
        with torch.no_grad():
            logits = n(torch.from_numpy(observation).type(torch.FloatTensor))
        d = dist.Categorical(logits=logits)
        action = d.sample()

        observation, reward, done, info = env.step(action.item())

        batch_actions.append(action.item())
        episode_rewards.append(reward)

        if done:
            episode_return = sum(episode_rewards)
            episode_length = len(episode_rewards)

            batch_episode_returns.append(episode_return)
            batch_episode_lengths.append(episode_length)

            batch_weights += [episode_return] * episode_length

            observation = env.reset()
            done = False
            episode_rewards = []

            if len(batch_observations) > batch_size:
                break
            
    n.train()
    opt.zero_grad()

    logits = n(torch.as_tensor(batch_observations).type(torch.FloatTensor))
    loss = -torch.mean(torch.FloatTensor(batch_weights) * dist.Categorical(logits=logits).log_prob(torch.IntTensor(batch_actions)))
    print(f" loss: {loss:.2f} return: {np.mean(batch_episode_returns):.2f} episode length: {np.mean(batch_episode_lengths):.2f}")
    loss.backward()

    opt.step()

torch.save(n.state_dict, './gym-lunar-torch-model.pt')