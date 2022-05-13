from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Classifier(ABC):

    @classmethod
    def from_pretrained(cls, path: str) -> Classifier:
        """load classifier with pretrained weight file

        Args:
            path (str): the path of the pretrained model 
        """
        raise NotImplementedError
    
    def predict(self, image_path: str) -> List[float]:
        """predict based on the image path

        Args:
            image_path (str): the path of the image file
        """
        raise NotImplementedError
