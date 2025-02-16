from abc import ABC, abstractmethod


class PipelineStep(ABC):
    @abstractmethod
    def run(self, data):
        pass

    @abstractmethod
    def save(self):
        pass
