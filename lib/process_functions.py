from time import perf_counter
from typing import List, Any
from model import InferenceModel, TaskType
from dataclasses import dataclass

########################################################
### Functions that will be run in the worker process ###
########################################################
model: InferenceModel

def worker_create_model(model_type):
    global model
    model = model_type()
 
 
def worker_model_predict(task_type: TaskType, data: List[Any]):
    start_time = perf_counter()
    result = model.run_task(task_type, data) 
    inference_time = int((perf_counter() - start_time) * 1000)
    print(f"Batch size: {len(data)} | {inference_time}ms | Task: {task_type}")
    return result

def worker_prepare_model():
    pass
########################################################