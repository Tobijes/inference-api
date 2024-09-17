from typing import List, Any
from enum import Enum
from dataclasses import dataclass

@dataclass
class ModelError(Exception):
    exception: Exception

class TaskType(str, Enum):
    pass

class InferenceModel():
    task_type: TaskType
    _task_registry = {}
    
    def __init__(self) -> None:        
        tasks = ", ".join(map(lambda x: x.name, self._task_registry))
        print(f"Model initating with following tasks defined: {tasks}")

    def run_task(self, task_type: TaskType, data: List[Any]):
        handler = self._task_registry.get(task_type, self._default_handler)
        return handler(self, data)
            
    def _default_handler(self, data):
        return data

    @classmethod
    def task(cls, task_type: TaskType):
        # This decorator will store the task_type and function to be registered later
        def decorator(func):
            if task_type in cls._task_registry:
                raise Exception("Duplicate task types defined across InferenceModel classes. Please define unique task types")
            cls._task_registry[task_type] = func
            return func
        return decorator
