import pickle
import time
import gym

from models.QLearn import QLearn

if __name__ == '__main__':
    env = gym.make('Acrobot-v1').unwrapped

    RL = QLearn(i=6, o=3, h1=512, h2=128,
                isinit=False, save_file='saves/Acrobot/params/ql.pkl',
                hyperparam={'learn_rate': 1e-3, # lr=1e-3 for training
                            'weight_decay': 1e-5,
                            'reward_decay': 0.01})

    scores = []
    for ii in range(1000000):
        state1 = env.reset()

        for turn in range(1000000):
            env.render()

            action = RL.action(state1, e=0.1) #  e=0.3 for training
            state2, reward, done, info = env.step(action)

            if done:
                RL.learn(action, state1, None, 1)
            else:
                RL.learn(action, state1, state2, -1)

            state1 = state2

            if done:
                print("Episode", ii, " finished after {} timesteps".format(turn + 1))
                RL.save_models()
                scores.append(turn + 1)
                with open('saves/Acrobot/scores.pkl', 'wb') as f:
                    pickle.dump(scores, f)
                time.sleep(1)
                break