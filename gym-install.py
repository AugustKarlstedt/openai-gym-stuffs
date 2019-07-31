import gym

base = 5
env = gym.make('Copy-v0', base=base)
env.reset()

for _ in range(1000):
    env.render()

    action = env.action_space.sample() # take a random action

    # 1. Move read head left or right (or up/down)
    # 2. Write or not
    # 3. Which character to write. (Ignored if should_write=0)

    input_action, output_action, prediction = action

    observation, reward, done, info = env.step(action)

    print(f"{observation} {reward} {done} {info}")

    if done:
        break
