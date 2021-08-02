from abc import ABC, abstractmethod


class NeuralNetwork(ABC):

    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def build(self, *args):
        pass

    @abstractmethod
    def fit(self, *args):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def predict(self, *args):
        pass


