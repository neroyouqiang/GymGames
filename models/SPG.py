import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical


class ActorModel(nn.Module):
    def __init__(self, i, o, h, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.linear1 = nn.Linear(i, h).to(device=self.device)
        self.linear2 = nn.Linear(h, o).to(device=self.device)
        self.softmax = nn.Softmax(dim=1).to(device=self.device)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return self.softmax(y)


class CriticModel(nn.Module):
    def __init__(self, i, h, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.linear1 = nn.Linear(i, h).to(device=self.device)
        self.linear2 = nn.Linear(h, 1).to(device=self.device)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)


class SPG:
    """
    Stochastic Policy Gradient for Discrete actions.
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
        hyperparam.setdefault('beta', 0.0) # strength of entropy regularization

        self.lr_critic = hyperparam['lr_critic']
        self.lr_actor = hyperparam['lr_actor']
        self.wd_critic = hyperparam['wd_critic']
        self.wd_actor = hyperparam['wd_actor']
        self.reward_decay = hyperparam['reward_decay']
        self.beta = hyperparam['beta']

        self.actor = ActorModel(i, o, h=ha)
        self.critic = CriticModel(i, h=hc)

        self.optimer_critic = optim.SGD(self.critic.parameters(),
                                        lr=self.lr_critic,
                                        weight_decay=self.wd_critic)

        self.optimer_actor = optim.SGD(self.actor.parameters(),
                                       lr=self.lr_actor,
                                       weight_decay=self.wd_actor)

        if not isinit and os.path.exists(self.save_file):
            self.load_models()
        else:
            self.save_models()

    def action(self, s):
        ps = self.actor(torch.FloatTensor([s]))
        a = Categorical(ps).sample()

        # print(ps, a)

        return a[0].numpy()

    def learn(self, a, s, s1, r):
        # data to tensors
        r = torch.FloatTensor([r])
        s = torch.FloatTensor([s])
        if s1 is not None:
            s1 = torch.FloatTensor([s1])

        # modify critic
        self.optimer_critic.zero_grad()

        if s1 is None:
            target = r
        else:
            v1 = self.critic(s1)[:, 0]
            target = r + (1 - self.reward_decay) * v1

        v = self.critic(s)[:, 0]
        loss = (v - target.detach()) ** 2 / 2
        loss = loss.mean()

        loss.backward()

        self.optimer_critic.step()

        # modify actor
        self.optimer_actor.zero_grad()

        pas = self.actor(s)
        pa = pas[:, a]
        loss = -(target - v).detach() * torch.log(pa) + self.beta * (pas * torch.log(pas)).mean(dim=1)
        loss = loss.mean()

        loss.backward()

        self.optimer_actor.step()

        # return info
        return {}

    def save_models(self):
        if self.save_file is not None:
            models = [self.actor.state_dict(), self.critic.state_dict()]
            torch.save(models, self.save_file)

    def load_models(self):
        if self.save_file is not None:
            models = torch.load(self.save_file)
            self.actor.load_state_dict(models[0])
            self.critic.load_state_dict(models[1])