from time import perf_counter
from typing import List, Any
from .model import InferenceModel, ModelError

########################################################
### Functions that will be run in the worker process ###
########################################################
model: InferenceModel

def worker_create_model(model_type):
    global model
    model = model_type()
 
 
def worker_model_predict(task_name: str, data: List[Any]):
    start_time = perf_counter()
    try:
        result = model.run_task(task_name, data) 
    except Exception as e:
        result = ModelError(e)
    inference_time = int((perf_counter() - start_time) * 1000)
    return inference_time, result

def worker_prepare_model():
    pass
########################################################