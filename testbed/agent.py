from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def process(self, observation):
        raise NotImplementedError()

    @abstractmethod
    def act(self):
        raise NotImplementedError()