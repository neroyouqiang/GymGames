import gym
from gym import spaces
import time
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # scores
    with open('scores_spg.pkl', 'rb') as f:
        scores_spg = pickle.load(f)

    with open('scores_ql.pkl', 'rb') as f:
        scores_ql = pickle.load(f)

    # print(len(scores), scores)

    scores_spg_show = []
    for ii in range(50, len(scores_spg)):
        scores_spg_show.append(np.mean(scores_spg[ii - 50: ii]))

    scores_ql_show = []
    for ii in range(50, len(scores_ql)):
        scores_ql_show.append(np.mean(scores_ql[ii - 50: ii]))

    plt.figure()
    plt.plot(scores_ql_show)
    plt.plot(scores_spg_show)
    plt.show()
