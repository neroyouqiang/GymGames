import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self, i, o, h1, h2, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.linear1 = nn.Linear(i, h1).to(device=self.device)
        self.linear2 = nn.Linear(h1, h2).to(device=self.device)
        self.linear3 = nn.Linear(h2, o).to(device=self.device)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(y))
        return self.linear3(y)


class QFunc:
    def __init__(self, h1, h2, hyperparam={}, isinit=False):
        # init variables
        self.save_file = 'params/QModels.pkl'

        hyperparam.setdefault('learn_rate', 1e-3)
        hyperparam.setdefault('weight_decay', 1e-5)
        hyperparam.setdefault('reward_decay', 0.1)

        self.learn_rate = hyperparam['learn_rate']
        self.weight_decay = hyperparam['weight_decay']
        self.reward_decay = hyperparam['reward_decay']

        self.model = MyModel(2, 3, h1=h1, h2=h2)
        self.optimer = optim.SGD(self.model.parameters(),
                                 lr=self.learn_rate,
                                 weight_decay=self.weight_decay)

        if not isinit and os.path.exists(self.save_file):
            # load params
            self.load_models()
        else:
            # save params
            self.save_models()

    def action(self, s, e=0.01):
        # e-greedy
        if random.random() <= e:
            # select a random action
            return random.randint(0, 2)
        else:
            # select the best action
            qs = self.model(torch.FloatTensor([s]))
            a = qs.argmax(dim=1)[0].numpy()

            # print(qs, a)

            return a

    def learn(self, a, s, s1, r):
        r_tns = torch.FloatTensor([r])
        s_tns = torch.FloatTensor([s])
        if s1 is None:
            s1_tns = None
        else:
            s1_tns = torch.FloatTensor([s1])

        # backpropagate
        self.optimer.zero_grad()

        qs = self.model(s_tns)
        if s1_tns is None:
            tf = qs[:, a] - r_tns.detach()
        else:
            q1s = self.model(s1_tns)
            q1_max, _ = q1s.max(dim=1)
            tf = qs[:, a] - (r_tns + (1 - self.reward_decay) * q1_max).detach()

        loss = (tf ** 2 / 2).mean()
        loss.backward()

        # modify parameters
        self.optimer.step()

        # save params
        # self.save_models()

    def save_models(self):
        print('Save model to', self.save_file)
        torch.save(self.model, self.save_file)
        print(self.model.linear1.bias)

    def load_models(self):
        print('Load model from', self.save_file)
        self.model = torch.load(self.save_file)
        print(self.model.linear1.bias)