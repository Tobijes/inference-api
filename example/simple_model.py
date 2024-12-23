import sys, os
sys.path.append(os.path.abspath(".."))

from typing import List
from time import sleep
from random import random

from lib.model import InferenceModel, ModelError

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
        raise ModelError("Model hiccup")
