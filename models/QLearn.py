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

    """
    def predict(self, s, a=None):
        if a is None:
            q, _ = self.model(s).max(dim=1)
        else:
            q = self.model(s)[:, a]

        return q
    """


class QLearn:
    """
    Q learning by neural network
    """
    def __init__(self, i, o, h1, h2, hyperparam={}, isinit=False, save_file=None):
        # init variables
        self.save_file = save_file
        self.input_dim = i
        self.output_dim = o

        hyperparam.setdefault('learn_rate', 1e-3)
        hyperparam.setdefault('weight_decay', 1e-5)
        hyperparam.setdefault('reward_decay', 0.1)

        self.learn_rate = hyperparam['learn_rate']
        self.weight_decay = hyperparam['weight_decay']
        self.reward_decay = hyperparam['reward_decay']

        self.model = MyModel(i, o, h1=h1, h2=h2)
        self.optimer = optim.SGD(self.model.parameters(),
                                 lr=self.learn_rate,
                                 weight_decay=self.weight_decay)

        if not isinit and os.path.exists(self.save_file):
            self.load_models()
        else:
            self.save_models()

    def action(self, s, e=0.1):
        # e-greedy
        if random.random() <= e:
            # select a random action
            return random.randint(0, self.output_dim - 1)
        else:
            # select the best action
            qs = self.model(torch.FloatTensor([s]))
            a = qs.argmax(dim=1).numpy()

            print(qs, a)

            return a[0]

    def learn(self, a, s, s1, r):
        # data to tensors
        r = torch.FloatTensor([r])
        s = torch.FloatTensor([s])
        if s1 is not None:
            s1 = torch.FloatTensor([s1])

        # backpropagate
        self.optimer.zero_grad()

        if s1 is None:
            target = r
        else:
            q1, _ = self.model(s1).max(dim=1)
            target =  r + (1 - self.reward_decay) * q1

        q = self.model(s)[:, a]
        loss = ((q - target.detach()) ** 2 / 2).mean()

        loss.backward()

        # modify parameters
        self.optimer.step()

        # save params
        # self.save_models()

    def save_models(self):
        if self.save_file is not None:
            torch.save(self.model.state_dict(), self.save_file)

    def load_models(self):
        if self.save_file is not None:
            self.model.load_state_dict(torch.load(self.save_file))