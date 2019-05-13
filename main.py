import gym
from gym import spaces
import time
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


if __name__ == '__main__':
    # for game in gym.envs.registry.all():
    #     print(game)

    # env = gym.make('SpaceInvaders-v0')
    # CartPole-v0
    # MountainCar_v0
    # MountainCarContinuous-v0
    # Pendulum-v0
    # Arobot-v1
   env = gym.make('Pendulum-v0').unwrapped
   env.reset()
   print(env.action_space)
   print(env.observation_space)
   while True:
        # print(env.step([1]))
       print(env.step([3]))
       env.render()

"""
    x = Variable(torch.Tensor([1, 1]), requires_grad=True)
    y = Variable(torch.Tensor([2, 2]), requires_grad=True)
    
    a = x + 1
    b = y + 1
    
    c = a + b
#    c = nn.MSELoss()(x, y.detach())
    
    c.backward(torch.Tensor([[1, 1], [1, 1], [2, 2]]))
    
    print(x.grad, y.grad, a.grad)
"""
