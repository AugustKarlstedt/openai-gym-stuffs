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

    def forward(self, *input):
        return input

v = ValueFunctionNetwork()

class SimplePolicyGradient(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.h_0 = torch.zeros(1, 256)
        self.c_0 = torch.zeros(1, 256)

        self.r_cell_1 = nn.LSTMCell(input_size=1, hidden_size=256)
        self.linear_1 = nn.Linear(256, 256)

        # The reasoning for base*2*2 is as follows
        # we have base equal to the number of characters in our problem, e.g.
        # base=4 means our input space is A B C D
        # we need two additional outputs for WRITE or NOT WRITE
        # so we have, e.g., A+WRITE
        # and we also now need the direction to traverse the tape
        # so LEFT or RIGHT
        # thus, for example A+WRITE+LEFT
        # or C+NOT WRITE+RIGHT
        # which gives us base*2*2 possibilities for outputs
        # e.g. in the 4*2*2 in the above example = 16
        # A WRITE LEFT, A WRITE RIGHT, A NOT WRITE LEFT, A NOT WRITE RIGHT
        # B WRITE LEFT, B WRITE RIGHT, B NOT WRITE LEFT, B NOT WRITE RIGHT
        # C WRITE LEFT, C WRITE RIGHT, C NOT WRITE LEFT, C NOT WRITE RIGHT
        # D WRITE LEFT, D WRITE RIGHT, D NOT WRITE LEFT, D NOT WRITE RIGHT

        self.output_1 = nn.Linear(256, base*2*2)

    def reset_hidden_state(self):
        self.h_0 = torch.zeros(1, 256)
        self.c_0 = torch.zeros(1, 256)

    def forward(self, *input):
        h_1, c_1 = self.r_cell_1(input[0], (self.h_0, self.c_0))

        self.h_0 = h_1
        self.c_0 = c_1

        return self.output_1(torch.tanh(self.linear_1(h_1)))

n = SimplePolicyGradient()

try:
    model = torch.load('gym-copy-torch-model.pt')()
    n.load_state_dict(model)
except FileNotFoundError:
    pass

lr = 1e-4
opt = optim.Adam(n.parameters(), lr=lr)
batch_size = 32
epochs = 4096
render = False

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def encode(letter, write, direction):
    return (letter * 4) + write + (direction * 2) - 1

NOT_WRITE = 1
WRITE = 2

LEFT = 0
RIGHT = 1

ENCODE_DICT = {}
DECODE_DICT = {}

for i, c in enumerate(charmap):
    if c == ' ':
        continue

    vec = np.zeros(base*2*2)
    idx = encode(i, NOT_WRITE, LEFT)
    vec[idx] = 1
    ENCODE_DICT[idx] = vec
    DECODE_DICT[idx] = (i, NOT_WRITE, LEFT)

    vec = np.zeros(base*2*2)
    idx = encode(i, WRITE, LEFT)
    vec[idx] = 1
    ENCODE_DICT[idx] = vec
    DECODE_DICT[idx] = (i, WRITE, LEFT)

    vec = np.zeros(base*2*2)
    idx = encode(i, NOT_WRITE, RIGHT)
    vec[idx] = 1
    ENCODE_DICT[idx] = vec
    DECODE_DICT[idx] = (i, NOT_WRITE, RIGHT)

    vec = np.zeros(base*2*2)
    idx = encode(i, WRITE, RIGHT)
    vec[idx] = 1
    ENCODE_DICT[idx] = vec
    DECODE_DICT[idx] = (i, WRITE, RIGHT)

for i in range(epochs):
    print(f"Epoch {i} of {epochs}: {i/epochs*100:.2f}% ", end="")

    batch_observations = []
    batch_actions = []
    batch_log_probs = []
    batch_targets = []
    batch_weights = []
    batch_episode_returns = []
    batch_episode_lengths = []

    episode_rewards = []
    observation = env.reset()

    while True:

        if render or i % 128 == 0:
            env.render()

        n.eval()
        observation = torch.FloatTensor([observation]).view(-1, 1)
        batch_observations.append(observation)
        
        logits = n(observation)
        d = dist.Multinomial(logits=logits)
        action = d.sample()

        batch_log_probs.append(d.log_prob(action))

        pred, out_act, inp_act = DECODE_DICT[action.argmax().item()]
        out_act -= 1

        observation, reward, done, info = env.step((inp_act, out_act, pred))

        if render or i % 128 == 0:
            print(f'Prediction: {charmap[pred]} Observation: {charmap[observation]}')

        batch_actions.append(action)
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

            if len(batch_observations) > batch_size:
                break
            
    n.train()
    opt.zero_grad()

    batch_weights = torch.Tensor(batch_weights)

    batch_loss = torch.cat(batch_log_probs) * batch_weights
    policy_loss = -batch_loss.mean()

    # prediction_ce_loss = 0 # -nn.functional.cross_entropy(batch_actions_predictions, batch_actions_targets, reduction='mean')
    # print(f"policy loss: {policy_loss:.2f} prediction ce loss: {prediction_ce_loss:.2f} ", end='')

    loss = policy_loss# + prediction_ce_loss
    loss.backward()
    print(f"total loss: {loss:.2f} return: {np.mean(batch_episode_returns):.2f} episode length: {np.mean(batch_episode_lengths):.2f}")

    opt.step()

torch.save(n.state_dict, './gym-copy-torch-model.pt')
