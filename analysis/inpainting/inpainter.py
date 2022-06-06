from abc import ABC, abstractmethod


class Inpainter(ABC):
    @abstractmethod
    def inpaint(self, image, mask):
        pass
