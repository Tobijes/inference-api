from time import perf_counter
from typing import List, Any
from dataclasses import dataclass

from .model import InferenceModel, ModelException

@dataclass
class TaskResult:
    inference_time: int
    result: Any = None
    error: Exception = None

########################################################
### Functions that will be run in the worker process ###
########################################################
model: InferenceModel

def worker_create_model(model_type):
    global model
    model = model_type()
 
 
def worker_model_predict(task_name: str, data: List[Any]) -> TaskResult:
    start_time = perf_counter()
    result = None
    error = None
    try:
        result = model.run_task(task_name, data) 
    except Exception as e:
        error = e
    inference_time = int((perf_counter() - start_time) * 1000)
    return TaskResult(
        inference_time = inference_time,
        result = result,
        error = error
    )

def worker_prepare_model():
    pass
########################################################