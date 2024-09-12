from typing import List
from time import sleep
from random import random

from lib.model import InferenceModel, TaskType

class ModelTaskType(TaskType):
    PREDICT = "PREDICT"

class SimpleModel(InferenceModel):
    task_type: ModelTaskType


    def __init__(self) -> None:      
        super().__init__() 

    @InferenceModel.task(ModelTaskType.PREDICT)
    def handle_passage(self, texts: List[str]):
        sleep(0.1 + 0.005 * len(texts))
        return [[random() * 50] * 768] * len(texts)
