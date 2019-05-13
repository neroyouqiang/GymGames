import gym
from gym import spaces
import time
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == '__main__':
    """
    with open('params/QParams.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data[0].reshape(-1).min())
    print(data[0].reshape(-1).max())
    print(data[1].reshape(-1).min())
    print(data[1].reshape(-1).max())

    tp = (0, 1, 2, 3)

    plt.figure()
    plt.plot(data[0].transpose(tp).mean(axis=1).mean(axis=1).mean(axis=1).reshape(-1))
    plt.plot(data[1].transpose(tp).mean(axis=1).mean(axis=1).mean(axis=1).reshape(-1))

    # plt.figure()
    # plt.plot(data[0].transpose(tp).reshape(-1) - data[1].transpose(tp).reshape(-1))
    # plt.show()
    """

    """
    # scores
    with open('saves/scores.pkl', 'rb') as f:
        scores = pickle.load(f)

    print(len(scores), scores)

    scores_show = []
    for ii in range(50, len(scores)):
        scores_show.append(np.mean(scores[ii - 50: ii]))

    plt.figure()
    plt.plot(scores_show)
    plt.show()
    """

    model = torch.load('params/QModels.pkl')
    print(model.linear1.bias)
