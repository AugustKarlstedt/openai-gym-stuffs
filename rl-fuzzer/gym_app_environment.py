import gym
from gym import error, spaces, utils
from gym.utils import seeding
from subprocess import Popen, PIPE

class AppEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.running = False
        self.process = None

    def step(self, action):
        if not self.running:
            self.process = Popen(["targets/buffer_overflow.exe"], stdin=PIPE, bufsize=0, text=True)
            self.running = True

        if action == '<SUBMIT>':
            self.process.stdin.write('\n')
        else:
            self.process.stdin.write(action)

        self.process.poll()

        obs = 
        done = self.process.returncode is not None
        info = {}

        return obs, reward, done, info
            

    def reset(self):
        self.running = False
        self.process.terminate()
        self.process = None

    def render(self, mode='human'):
        pass

    def close(self):
        self.running = False
        self.process.terminate()
        self.process = None
