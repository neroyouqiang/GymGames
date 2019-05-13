import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class ActorModel(nn.Module):
    def __init__(self, i, o, h, out_range=None, device=torch.device('cpu')):
        super().__init__()

        self.device = device
        self.out_range = out_range

        self.linear1 = nn.Linear(i, h).to(device=self.device)
        self.linear2 = nn.Linear(h, o).to(device=self.device)

        # self.linear1.weight.data.normal_(0, 0.1)
        # self.linear2.weight.data.normal_(0, 0.1)

    def forward(self, s):
        y = F.relu(self.linear1(s))
        y = self.linear2(y)

        if self.out_range is not None:
            y = torch.tanh(y) * self.out_range

        return y


class CriticModel(nn.Module):
    def __init__(self, i, o, h, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.linear_s = nn.Linear(i, h).to(device=self.device)
        self.linear_a = nn.Linear(o, h).to(device=self.device)
        self.linear2 = nn.Linear(h, 1).to(device=self.device)

        # self.linear_s.weight.data.normal_(0, 0.1)
        # self.linear_a.weight.data.normal_(0, 0.1)
        # self.linear2.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        y = F.relu(self.linear_s(s) + self.linear_a(a))
        y = self.linear2(y)

        return y


class DPG:
    """
    Deterministic Policy Gradient for Continuous actions.
    """
    def __init__(self, i, o, ha, hc, hyperparam={}, isinit=False, save_file=None):
        # init variables
        self.save_file = save_file
        self.input_dim = i
        self.output_dim = o

        hyperparam.setdefault('lr_critic', 1e-3)
        hyperparam.setdefault('lr_actor', 1e-3)
        hyperparam.setdefault('wd_critic', 1e-5)
        hyperparam.setdefault('wd_actor', 1e-5)
        hyperparam.setdefault('reward_decay', 0.1)
        hyperparam.setdefault('max_mem', 10000)
        hyperparam.setdefault('batch_size', 32)
        hyperparam.setdefault('tau', 1.0) # speed to synchronize the target and main model
        hyperparam.setdefault('out_range', 2.0) # the range of output values

        self.lr_critic = hyperparam['lr_critic']
        self.lr_actor = hyperparam['lr_actor']
        self.wd_critic = hyperparam['wd_critic']
        self.wd_actor = hyperparam['wd_actor']
        self.reward_decay = hyperparam['reward_decay']
        self.tau = hyperparam['tau']
        self.max_mem = hyperparam['max_mem']
        self.batch_size = hyperparam['batch_size']
        self.out_range = hyperparam['out_range']

        self.actor = ActorModel(i, o, h=ha, out_range=self.out_range)
        self.critic = CriticModel(i, o, h=hc)

        self.target_actor = ActorModel(i, o, h=ha, out_range=self.out_range)
        self.target_critic = CriticModel(i, o, h=hc)

        self.optimer_critic = optim.Adam(self.critic.parameters(),
                                         lr=self.lr_critic,
                                         weight_decay=self.wd_critic)

        self.optimer_actor = optim.Adam(self.actor.parameters(),
                                        lr=self.lr_actor,
                                        weight_decay=self.wd_actor)

        self.memory = np.zeros([self.max_mem, self.input_dim + self.output_dim + 1 + self.input_dim])
        self.mem_idx = 0

        if not isinit and os.path.exists(self.save_file):
            self.load_models()
        else:
            self.save_models()

    def action(self, s, sch):
        mean = self.actor(torch.FloatTensor([s]))
        std = torch.FloatTensor([sch])
        a = Normal(mean, std).sample()

        # print(mean, std, a, self.critic(torch.FloatTensor([s]), mean))

        return a[0].numpy()

    def learn(self):
        # sample memory
        idxs = np.random.choice(self.memory.shape[0], self.batch_size)
        data = self.memory[idxs, :]

        # memory replay
        s = torch.FloatTensor(data[:, 0: self.input_dim])
        a = torch.FloatTensor(data[:, self.input_dim: self.input_dim + self.output_dim])
        r = torch.FloatTensor(data[:, self.input_dim + self.output_dim: self.input_dim + self.output_dim + 1])
        s1 = torch.FloatTensor(data[:, self.input_dim + self.output_dim + 1: self.input_dim + self.output_dim + 1 + self.input_dim])

        # modify critic
        self.optimer_critic.zero_grad()

        # set the value of middle state
        target = r + (1 - self.reward_decay) * self.target_critic(s1, self.target_actor(s1))
        # target = r + (1 - self.reward_decay) * self.critic(s1, self.actor(s1))

        # set the value of end state to reward
        # idxs = (s == 0).all(dim=1)
        # target[idxs] = r[idxs]

        v = self.critic(s, a)
        loss = (v - target.detach()) ** 2 / 2
        loss = loss.mean()

        loss.backward()

        self.optimer_critic.step()

        # modify actor
        self.optimer_actor.zero_grad()

        loss = -self.critic(s, self.actor(s))
        loss = loss.mean()

        loss.backward()

        self.optimer_actor.step()

        # update target model parameters
        for key in self.actor.state_dict().keys():
            exec('self.target_actor.' + key + '.data = self.tau * self.actor.' + key + '.data + (1 - self.tau) * self.target_actor.' + key + '.data')

        for key in self.critic.state_dict().keys():
            exec('self.target_critic.' + key + '.data = self.tau * self.critic.' + key + '.data + (1 - self.tau) * self.target_critic.' + key + '.data')

        # return info
        return {}

    def store_transition(self, s, a, r, s1):
        # if s1 is None:
        #     s1 = np.zeros(s.shape)

        idx = self.mem_idx % self.max_mem
        self.memory[idx, :] = np.hstack([s, a, [r], s1])

        self.mem_idx += 1

    def save_models(self):
        if self.save_file is not None:
            models = [self.actor.state_dict(), self.critic.state_dict()]
            torch.save(models, self.save_file)

    def load_models(self):
        if self.save_file is not None:
            models = torch.load(self.save_file)
            self.actor.load_state_dict(models[0])
            self.critic.load_state_dict(models[1])