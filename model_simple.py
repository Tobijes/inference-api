from typing import List
from time import sleep
from random import random

from lib.model import InferenceModel, TaskType

class SimpleTaskType(TaskType):
    PREDICT = "PREDICT"

class SimpleModel(InferenceModel):
    task_type: SimpleTaskType


    def __init__(self) -> None:      
        super().__init__() 

    @InferenceModel.task(SimpleTaskType.PREDICT)
    def handle_passage(self, texts: List[str]):
        sleep(0.1 + 0.005 * len(texts))
        # if random() < 0.2:
        #     raise ValueError("Value small")
        return [[random() * 50] * 768] * len(texts)
