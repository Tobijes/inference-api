from time import perf_counter
import logging
from typing import List, Any
from dataclasses import dataclass

from .model import InferenceModel
from .model import ModelError

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
    except ModelError as me:
        logging.getLogger('uvicorn.error').error("Model Error: %s", me.message)
        error = me
    except Exception as e:
        message = f"{type(e).__name__}: {str(e)}"
        logging.getLogger('uvicorn.error').error(message)
        error = ModelError(message=message, http_status_code=400)
    inference_time = int((perf_counter() - start_time) * 1000)
    return TaskResult(
        inference_time = inference_time,
        result = result,
        error = error
    )

def worker_model_prepare():
    return True
########################################################

 
# def worker_model_predict(args):
#     start_time = perf_counter()
#     try:
#         result = model.predict(*args)
#     except ModelError as me:
#         logging.getLogger('uvicorn.error').error("Model Error: %s", me.message)
#         result = me
#     except Exception as e:
#         message = f"{type(e).__name__}: {str(e)}"
#         logging.getLogger('uvicorn.error').error(message)
#         result = ModelError(message=message, http_status_code=400)

#     inference_time = perf_counter() - start_time
#     return inference_time, result
