from time import perf_counter
from typing import List, Any
from model import TaskType
from dataclasses import dataclass

########################################################
### Functions that will be run in the worker process ###
########################################################
def worker_create_model(model_type):
    global model
    model = model_type()
 
 
def worker_model_predict(task_type: TaskType, data: List[Any]):
    start_time = perf_counter()
    result = model.task(task_type, data) 
    inference_time = int((perf_counter() - start_time) * 1000)
    print(f"Batch size: {len(data)} | {inference_time}ms")
    return result
########################################################