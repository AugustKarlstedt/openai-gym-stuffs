import seaborn as sns
import pandas as pd
import numpy as np
import itertools

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

import gym

env = gym.make('CartPole-v1')

learning_rate = 0.001
discount_factor = 0.9

a = Input(shape=(4, ))
b = Dense(16, activation='relu')(a)
c = Dense(16, activation='relu')(b)
d = Dense(2, activation='linear')(c)
model = Model(inputs=[a], outputs=[d])
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate))

for i in range(1000):
        print(f"{i/1000*100:.2f}% ", end="")

        observation = env.reset()
        done = False

        explore_rate =  max(np.exp(-i*0.01), 0.01)
        print(explore_rate)

        episode_length = 1
        while not done:
                env.render()

                x = np.reshape(observation, (1, 4))
                Q = model.predict(x)
                
                if np.random.rand() <= explore_rate:
                        action = env.action_space.sample()
                else:
                        action = np.argmax(Q)

                observation, reward, done, info = env.step(action)

                if not done:
                        x_next = np.reshape(observation, (1, 4))
                        Q_next = model.predict(x=x_next)
                        target = reward + discount_factor * np.amax(Q_next)
                else:
                        print('Done! Episode length:', episode_length)
                        target = reward
                        
                Q[0, action] = target

                model.fit(x=x, y=Q, epochs=1, verbose=0)

                episode_length += 1
