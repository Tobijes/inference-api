import logging
from typing import List, Any, Dict, Callable, Optional, get_type_hints
from dataclasses import dataclass

from lib.settings import BaseSettings, SettingsLoader
from lib.utils import is_cuda_available

@dataclass(frozen=True)
class TaskKey:
    model_name: str
    task_name: str

class ModelError(Exception):
    message: str 
    http_status_code: int

    def __init__(self, message = "Error in model inference", http_status_code = 400):
        self.message = message
        self.http_status_code = http_status_code

class InferenceModel:
    _task_registry: Dict[TaskKey, Callable] = {}

    model_metrics_timing_buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 90.0, 120.0, 200.0]

    settings: BaseSettings
    model_name = "Default Model"
    device: str = "cpu"
    
    def __init__(self) -> None:        
        tasks = ", ".join(self.get_task_names())
        print(f"Model '{self.__class__.__name__}' initiating with following tasks defined: {tasks}")
        self.logger = logging.getLogger('uvicorn.error')
        # Load settings
        settings_type = get_type_hints(type(self))["settings"]
        self.settings = SettingsLoader.load(settings_type)
        # Set device CPU/CUDA
        self.device = "cuda" if (is_cuda_available() and self.settings.USE_GPU) else "cpu"
        
    def run_task(self, task_name: str, data: List[Any]):
        task_key = TaskKey(self.__class__.__name__, task_name)
        handler = self._task_registry.get(task_key, self._default_handler)
        return handler(self, data)
            
    def _default_handler(self, data):
        return data

    @classmethod
    def get_task_names(cls) -> List[str]:
        return [x.task_name for x in cls._task_registry if x.model_name == cls.__name__]
    
    @classmethod
    def get_task_key(cls, func: Callable) -> TaskKey:
        qual_parts = func.__qualname__.split(".")
        model_name = qual_parts[0]
        task_name = "".join(qual_parts[1:])
        return TaskKey(model_name, task_name)

    @classmethod
    def task(cls):
        # This decorator will store the task_name and function to be registered later
        def decorator(func: Callable):
            task_key = cls.get_task_key(func)

            if task_key in cls._task_registry:
                raise Exception("Duplicate task types defined across InferenceModel classes. Please define unique task types")
            cls._task_registry[task_key] = func
            return func
        return decorator
