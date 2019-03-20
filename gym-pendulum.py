import seaborn as sns
import pandas as pd
import numpy as np
import itertools

import gym
env = gym.make('Pendulum-v0')

learning_rate = 0.1
discount_factor = 0.9

try:
        df = pd.read_csv('qtable-pendulum.csv', index_col=0)
except FileNotFoundError:
        df = pd.DataFrame([])

if df.empty:
        cos_theta = [np.round(x, decimals=1) for x in np.arange(start=-0.1, stop=0.2, step=0.1)]
        sin_theta = [np.round(x, decimals=1) for x in np.arange(start=-0.1, stop=0.2, step=0.1)]
        theta_dot = [np.round(x, decimals=1) for x in np.arange(start=-0.1, stop=0.2, step=0.1)]

        p = list(map(list, itertools.product(cos_theta, sin_theta, theta_dot)))                
        df = pd.DataFrame(p, columns=['cos_theta', 'sin_theta', 'theta_dot'])
        df['action_torque_left'] = np.random.uniform(low=-0.5, high=0.5, size=len(p))
        df['action_torque_right'] = np.random.uniform(low=-0.5, high=0.5, size=len(p))

        print(df)

for i in range(1000):

        print(f"{i/1000*100:.2f}% ", end="")

        env.reset()
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())

        while not done:

                cos_theta = max(min(float("{0:.1f}".format(observation[0])), 0.1), -0.1)
                sin_theta = max(min(float("{0:.1f}".format(observation[1])), 0.1), -0.1)
                theta_dot = max(min(float("{0:.1f}".format(observation[2])), 0.1), -0.1)

                actions = df[(df['cos_theta']==cos_theta) & (df['sin_theta']==sin_theta) & (df['theta_dot']==theta_dot)]                

                action_torque_left = actions['action_torque_left'].item()
                action_torque_right = actions['action_torque_right'].item()

                
                if action_torque_left > action_torque_right:
                        action = [-2.0]
                elif action_torque_left < action_torque_right:
                        action = [2.0]
                else:
                        action = [0.0]

                env.render()
                observation, reward, done, info = env.step(action)

                if done:
                        print('!!! DONE !!!')
                        print(df[(df['action_torque_left']!=0.0) | (df['action_torque_right']!=0.0)])
                        print()

                cos_theta = max(min(float("{0:.1f}".format(observation[0])), 0.1), -0.1)
                sin_theta = max(min(float("{0:.1f}".format(observation[1])), 0.1), -0.1)
                theta_dot = max(min(float("{0:.1f}".format(observation[2])), 0.1), -0.1)

                values = df[(df['cos_theta']==cos_theta) & (df['sin_theta']==sin_theta) & (df['theta_dot']==theta_dot)]                

                if action == [-2.0]:
                        actions['action_torque_left'] = actions['action_torque_left'].item() + learning_rate * (reward + discount_factor * values['action_torque_left'].item() - actions['action_torque_left'].item())
                elif action == [2.0]:
                        actions['action_torque_right'] = actions['action_torque_right'].item() + learning_rate * (reward + discount_factor * values['action_torque_right'].item() - actions['action_torque_right'].item())
                else:
                        actions['action_torque_left'] = actions['action_torque_left'].item() + learning_rate * (reward + discount_factor * values['action_torque_left'].item() - actions['action_torque_left'].item())
                        actions['action_torque_right'] = actions['action_torque_right'].item() + learning_rate * (reward + discount_factor * values['action_torque_right'].item() - actions['action_torque_right'].item())


                df.update(actions)
                
df.to_csv('qtable-pendulum.csv')
