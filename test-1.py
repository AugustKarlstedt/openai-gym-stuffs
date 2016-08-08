import gym
from gym import envs
from gym import spaces

#print(envs.registry.all())

env = gym.make('CartPole-v0')

print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

print(env.action_space)
action_space_len = 2

for i_episode in range(20):
    observation = env.reset()

    for t in range(100):
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())

        for i_action in range(action_space_len):
            print(env.action_space.sample())

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
