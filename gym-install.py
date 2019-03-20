import gym
env = gym.make('LunarLander-v2')
env.reset()

for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action

    cart_position = observation[0]
    cart_velocity = observation[1]
    pole_angle = observation[2]
    pole_velocity_at_tip = observation[3]

    print(f"{observation} {reward} {done} {info}")

    if done:
        break
