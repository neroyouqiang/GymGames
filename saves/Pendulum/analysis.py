import gym
from gym import spaces
import time
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # scores
    name_list = ['rewards_ha_64_hc_64_tau_0.1.pkl',
                 'rewards_ha_32_hc_32_tau_0.1.pkl',
                 'rewards_ha_32_hc_32_tau_1.0.pkl',
                 'rewards_ha_256_hc_256_tau_1.0.pkl']
    scores_list = []
    for name in name_list:
        with open(name, 'rb') as f:
            scores = pickle.load(f)
            scores_list.append(scores)

    # print(len(scores), scores)
    plt.figure()

    for scores in scores_list:
        scores_show = []
        for ii in range(50, len(scores)):
            scores_show.append(np.mean(scores[ii - 50: ii]))

        plt.plot(scores_show)

    plt.legend(['ha_64_hc_64_tau_0',
                'ha_32_hc_32_tau_0.1',
                'ha_32_hc_32_tau_1.0',
                'ha_256_hc_256_tau_1.0'])
    plt.show()
