from typing import List
from time import sleep
from random import random

from lib.model import InferenceModel, ModelException

class SimpleModel(InferenceModel):

    def __init__(self) -> None:      
        super().__init__() 

    @InferenceModel.task()
    def predict(self, texts: List[str]):
        sleep(0.1 + 0.005 * len(texts))
        return [[random() * 50] * 768] * len(texts)

    @InferenceModel.task()
    def simulate_unknown_error(self, data):
        raise ValueError("Value small")
    
    @InferenceModel.task()
    def simulate_known_error(self, data):
        raise ModelException("Model hiccup")

