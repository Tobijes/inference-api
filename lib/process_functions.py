from time import perf_counter_ns
import logging
from typing import Any
from dataclasses import dataclass

from setproctitle import setproctitle

from .model import InferenceModel
from .exceptions import ModelError

@dataclass
class TaskResult:
    process_id: int
    inference_time: int
    result: Any = None
    error: Exception = None

########################################################
### Functions that will be run in the worker process ###
########################################################
model: InferenceModel

def worker_create_model(model_type: type[InferenceModel]):
    setproctitle(f"ModelWorker/{model_type.__name__}")
    global model
    model = model_type()
 
 
def worker_model_predict(data: list[Any], **kwargs) -> TaskResult:
    start_time = perf_counter_ns()
    result = None
    error = None
    try:
        result = model.infer(data, **kwargs) 
    except ModelError as me:
        logging.getLogger('uvicorn.error').error("Model Error: %s", me.message)
        error = me
    except Exception as e:
        message = f"{type(e).__name__}: {str(e)}"
        logging.getLogger('uvicorn.error').error(message)
        error = ModelError(message=message, http_status_code=400)
    inference_time = int((perf_counter_ns() - start_time) / 10**6)
    return TaskResult(
        process_id = model.process_id,
        inference_time = inference_time,
        result = result,
        error = error
    )

def worker_model_prepare():
    return True
########################################################
