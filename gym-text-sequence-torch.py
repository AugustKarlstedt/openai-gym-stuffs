import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

import os
import pdb
import time
import math

import torch
import torch.tensor
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

import gym

base = 4
env = gym.make('Copy-v0', base=base)
charmap = env.charmap

class ValueFunctionNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.h_0 = torch.zeros(1, 16)
        self.c_0 = torch.zeros(1, 16)

        self.r_cell_1 = nn.LSTMCell(input_size=1, hidden_size=16)

        self.linear_1 = nn.Linear(16, 16)

        self.output_1 = nn.Linear(16, 1)

    def reset_hidden_state(self):
        self.h_0 = torch.zeros(1, 16)
        self.c_0 = torch.zeros(1, 16)

    def forward(self, *input):
        h_1, c_1 = self.r_cell_1(input[0], (self.h_0, self.c_0))

        self.h_0 = h_1
        self.c_0 = c_1

        return self.output_1(torch.tanh(self.linear_1(h_1)))

v = ValueFunctionNetwork()

class SimplePolicyGradient(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.h_0 = torch.zeros(1, 256)
        self.c_0 = torch.zeros(1, 256)

        self.r_cell_1 = nn.LSTMCell(input_size=1, hidden_size=256)

        self.linear_1 = nn.Linear(256, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.linear_3 = nn.Linear(256, 256)

        self.output_1 = nn.Linear(256, 2)
        self.output_2 = nn.Linear(256, 2)
        self.output_3 = nn.Linear(256, base)

    def reset_hidden_state(self):
        self.h_0 = torch.zeros(1, 256)
        self.c_0 = torch.zeros(1, 256)

    def forward(self, *input):
        h_1, c_1 = self.r_cell_1(input[0], (self.h_0, self.c_0))

        self.h_0 = h_1
        self.c_0 = c_1

        return self.output_1(torch.tanh(self.linear_1(h_1))), self.output_2(torch.tanh(self.linear_2(h_1))), self.output_3(torch.tanh(self.linear_3(h_1)))

n = SimplePolicyGradient()

try:
    model = torch.load('gym-copy-torch-model.pt')()
    n.load_state_dict(model)
except FileNotFoundError:
    pass

try:
    model = torch.load('gym-copy-torch-v-model.pt')()
    v.load_state_dict(model)
except FileNotFoundError:
    pass

lr = 1e-4
opt = optim.Adam(n.parameters(), lr=lr)
batch_size = 32
epochs = 4096
render = False

v_opt = optim.Adam(v.parameters(), lr=lr)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

ENCODE_DICT = {}
DECODE_DICT = {}

for i, c in enumerate(charmap):
    if c == ' ':
        continue

    vec = np.zeros(base)
    vec[i] = 1
    ENCODE_DICT[i] = vec
    DECODE_DICT[i] = c

mean_returns = []
prev_mean = 0.0
skip = 8

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(left=0, right=epochs+skip)

for i in range(epochs):
    if i % skip == 0 and len(mean_returns) > 0:
        print('*'*16, f'Mean Returns: {np.mean(mean_returns)}', '*'*16)
        line1, = ax.plot([max(i-skip, 0), i], [prev_mean, np.mean(mean_returns)], 'bo-', linewidth=2, markersize=2)
        fig.canvas.draw()
        prev_mean = np.mean(mean_returns)
        mean_returns = []

    print(f"Epoch {i} of {epochs}: {i/epochs*100:.2f}% ", end="")

    # """Sets the learning rate to the initial LR decayed by 10 every 2048 epochs"""
    # lr = lr * (0.1 ** (i // 1024))
    # for param_group in opt.param_groups:
    #     param_group['lr'] = lr

    batch_observations = []
    batch_actions = []
    batch_log_probs_inp = []
    batch_log_probs_out = []
    batch_log_probs_pred = []
    batch_targets = []
    batch_weights = []
    batch_episode_returns = []
    batch_episode_lengths = []
    batch_return_pred = []

    episode_rewards = []
    observation = env.reset()

    while True:

        if render:
            env.render()

        n.eval()
        observation = torch.FloatTensor([observation]).view(-1, 1)
        batch_observations.append(observation)
        
        inp_logits, out_logits, pred_logits = n(observation)

        inp_dist = dist.Multinomial(logits=inp_logits)
        out_dist = dist.Multinomial(logits=out_logits)
        pred_dist = dist.Multinomial(logits=pred_logits)

        inp_action = inp_dist.sample()
        out_action = out_dist.sample()
        pred_action = pred_dist.sample()

        batch_log_probs_inp.append(inp_dist.log_prob(inp_action))
        batch_log_probs_out.append(out_dist.log_prob(out_action))
        batch_log_probs_pred.append(pred_dist.log_prob(pred_action))

        inp = np.argmax(inp_action).item()
        out = np.argmax(out_action).item()
        pred = np.argmax(pred_action).item()

        v.eval()
        pred_avg_return = v(observation)
        batch_return_pred.append(pred_avg_return)

        observation, reward, done, info = env.step((inp, out, pred))

        episode_rewards.append(reward)

        if done:
            episode_return = sum(episode_rewards)
            episode_length = len(episode_rewards)

            batch_episode_returns.append(episode_return)
            batch_episode_lengths.append(episode_length)

            # batch_weights += [episode_return] * episode_length
            batch_weights += list(reward_to_go(episode_rewards))

            observation = env.reset()
            done = False
            episode_rewards = []
            n.reset_hidden_state()
            v.reset_hidden_state()

            if len(batch_observations) > batch_size:
                break
            
    v.train()
    v_opt.zero_grad()

    batch_weights = torch.tensor(batch_weights)
    batch_return_pred = torch.tensor(batch_return_pred, requires_grad=True)

    v_loss = torch.sqrt(nn.functional.mse_loss(batch_return_pred, batch_weights))
    v_loss.backward()
    v_opt.step()

    n.train()
    opt.zero_grad()

    baseline = torch.mean(batch_return_pred)

    inp_batch_loss = torch.cat(batch_log_probs_inp) * batch_weights
    inp_policy_loss = -inp_batch_loss.mean()# - baseline

    out_batch_loss = torch.cat(batch_log_probs_out) * batch_weights
    out_policy_loss = -out_batch_loss.mean()# - baseline

    pred_batch_loss = torch.cat(batch_log_probs_pred) * batch_weights
    pred_policy_loss = -pred_batch_loss.mean()# - baseline

    loss = inp_policy_loss + out_policy_loss + pred_policy_loss
    loss.backward()

    opt.step()
    
    batch_return_pred = []

    mean_return = np.mean(batch_episode_returns)
    mean_returns.append(mean_return)
    print(f"v function loss: {v_loss:.2f} total loss: {loss:.2f} return: {mean_return:.2f} episode length: {np.mean(batch_episode_lengths):.2f}")




torch.save(n.state_dict, './gym-copy-torch-model.pt')

plt.show(block=True)