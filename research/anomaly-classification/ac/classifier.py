from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def predict(self, examples):
        raise NotImplementedError
    
    @abstractmethod
    def load(**kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(**kwargs):
        raise NotImplementedError

