import seaborn as sns
import pandas as pd
import numpy as np
import itertools

import gym
env = gym.make('CartPole-v1')

learning_rate = 0.33
discount_factor = 0.66

try:
        df = pd.read_csv('qtable.csv', index_col=0)
except FileNotFoundError:
        df = pd.DataFrame([])

if df.empty:
        #cart_positions = [np.round(x, decimals=2) for x in np.arange(start=-0.25, stop=0.26, step=0.01)]
        #cart_velocities = [np.round(x, decimals=1) for x in np.arange(start=-3.0, stop=3.1, step=0.1)]
        pole_angles = [np.round(x, decimals=2) for x in np.arange(start=-0.05, stop=0.06, step=0.01)]
        pole_velocities = [np.round(x, decimals=2) for x in np.arange(start=-0.05, stop=0.06, step=0.01)]

        #p = list(map(list, itertools.product(cart_positions, cart_velocities, pole_angles, pole_velocities)))
        #p = list(map(list, itertools.product(cart_positions, pole_angles)))        
        p = list(map(list, itertools.product(pole_angles, pole_velocities)))                
        #df = pd.DataFrame(p, columns=['state_cart_position', 'state_cart_velocity', 'state_pole_angle', 'state_pole_velocity'])
        #df = pd.DataFrame(p, columns=['state_cart_position', 'state_pole_angle'])
        df = pd.DataFrame(p, columns=['state_pole_angle', 'state_pole_velocity'])
        df['action_move_left'] = 0.0
        df['action_move_right'] = 0.0

        print(df)

for i in range(1000):

        print(f"{i/1000*100:.2f}% ", end="")

        env.reset()
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())

        episode_length = 1
        while not done:
                #cart_position = max(min(float("{0:.2f}".format(observation[0])), 0.25), -0.25)
                #cart_velocity = max(min(float("{0:.1f}".format(observation[1])), 3.0), -3.0)
                pole_angle = max(min(float("{0:.2f}".format(observation[2])), 0.05), -0.05)
                pole_velocity_at_tip = max(min(float("{0:.2f}".format(observation[3])), 0.05), -0.05)

                #actions = df[(df['state_cart_position']==cart_position) & (df['state_cart_velocity']==cart_velocity) & (df['state_pole_angle']==pole_angle) & (df['state_pole_velocity']==pole_velocity_at_tip)]                
                #actions = df[(df['state_cart_position']==cart_position) &(df['state_pole_angle']==pole_angle)]                
                actions = df[(df['state_pole_angle']==pole_angle) & (df['state_pole_velocity']==pole_velocity_at_tip)]                

                move_left = actions['action_move_left'].item()
                move_right = actions['action_move_right'].item()

                if np.random.rand() <= 0.01:
                        action = env.action_space.sample()
                else:
                        action = 0 if move_left > move_right else 1

                env.render()
                observation, reward, done, info = env.step(action)

                if done:
                        print('Done! Episode length:', episode_length)

                        if episode_length < 200:
                                reward = -2

                #cart_position = max(min(float("{0:.2f}".format(observation[0])), 0.25), -0.25)
                #cart_velocity = max(min(float("{0:.1f}".format(observation[1])), 3.0), -3.0)
                pole_angle = max(min(float("{0:.2f}".format(observation[2])), 0.05), -0.05)
                pole_velocity_at_tip = max(min(float("{0:.2f}".format(observation[3])), 0.05), -0.05)

                #values = df[(df['state_cart_position']==cart_position) & (df['state_cart_velocity']==cart_velocity) & (df['state_pole_angle']==pole_angle) & (df['state_pole_velocity']==pole_velocity_at_tip)]                
                #values = df[(df['state_cart_position']==cart_position) & (df['state_pole_angle']==pole_angle)]                
                values = df[(df['state_pole_angle']==pole_angle) & (df['state_pole_velocity']==pole_velocity_at_tip)]                

                if action == 0:
                        actions['action_move_left'] = actions['action_move_left'].item() + learning_rate * (reward + discount_factor * values['action_move_left'].item() - actions['action_move_left'].item())
                else:
                        actions['action_move_right'] = actions['action_move_right'].item() + learning_rate * (reward + discount_factor * values['action_move_right'].item() - actions['action_move_right'].item())

                df.update(actions)
                
                episode_length += 1
                
df.to_csv('qtable.csv')
