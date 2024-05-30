from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    @abstractmethod
    def add_noise(self):
        pass

    @abstractmethod
    def __len__(self):
        pass
