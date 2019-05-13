import gym
import time
import pickle
from MountainCar_v0.QFunc import QFunc


if __name__ == '__main__':
    env = gym.make('MountainCar-v0').unwrapped

    # learn_rate=1e-3 for training
    Q = QFunc(h1=512, h2=128, isinit=False, hyperparam={'learn_rate': 1e-3,
                                                        'weight_decay': 1e-5,
                                                        'reward_decay': 0.1})
    data1 = []
    data2 = []
    scores = []
    for ii in range(50000):
        state1 = env.reset()

        for turn in range(1000000):
            env.render()

            action = Q.action(state1, e=0.5)
            state2, reward, done, info = env.step(action)

            if done:
                Q.learn(action, state1, None, 100)
            else:
                if state2[0] > 0.2:
                    Q.learn(action, state1, state2, state2[0])
                else:
                    Q.learn(action, state1, state2, -1)

            if reward > -1:
                print(state2, reward)

            state1 = state2

            if done:
                print("Episode", ii, " finished after {} timesteps.".format(turn + 1), 'Reward is', reward)
                scores.append(turn + 1)
                with open('saves/scores.pkl', 'wb') as f:
                    pickle.dump(scores, f)
                time.sleep(1)
                Q.save_models()
                break