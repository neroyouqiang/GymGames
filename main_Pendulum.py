import pickle
import time
import gym

from models.DPG import DPG

if __name__ == '__main__':
    env = gym.make('Pendulum-v0').unwrapped

    RL = DPG(i=3, o=1, ha=64, hc=64,
             isinit=True, save_file='saves/Pendulum/params/spgd.pkl',
             hyperparam={'lr_critic': 2e-3,
                         'lr_actor': 1e-3,
                         'wd_critic': 1e-5,
                         'wd_actor': 1e-5,
                         'reward_decay': 0.1,
                         'max_mem': 1000,
                         'batch_size': 64,
                         'tau': 1.0,})

    state1 = env.reset()

    rewards = []
    var = 1.0
    for ii in range(50000):
        state1 = env.reset()

        reward_sum = 0
        for turn in range(200):

            action = RL.action(state1, var)
            state2, reward, done, info = env.step(action)
            RL.store_transition(state1, action, reward,  state2)

            if RL.mem_idx > RL.max_mem:
                env.render()
                RL.learn()

            state1 = state2

            reward_sum += reward

        if RL.mem_idx > RL.max_mem:
            var *= 0.995

        rewards.append(reward_sum)

        print("At episode {}, reward is {}, variance is {}".format(ii + 1, reward_sum, var))
        RL.save_models()
        with open('saves/Pendulum/rewards.pkl', 'wb') as f:
            pickle.dump(rewards, f)

        time.sleep(1)
