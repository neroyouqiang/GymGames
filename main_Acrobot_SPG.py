import pickle
import time
import gym

from models.SPG import SPG

if __name__ == '__main__':
    env = gym.make('Acrobot-v1').unwrapped

    RL = SPG(i=6, o=3, ha=512, hc=512,
             isinit=False, save_file='saves/Acrobot/params/spgd.pkl',
             hyperparam={'lr_critic': 1e-3,
                             'lr_actor': 1e-4,
                             'wd_critic': 1e-5,
                             'wd_actor': 1e-5,
                             'reward_decay': 0.01,
                             'beta': 1.0,})

    scores = []
    for ii in range(1000000):
        state1 = env.reset()

        for turn in range(1000000):
            env.render()

            action = RL.action(state1)
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