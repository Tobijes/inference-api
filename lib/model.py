from typing import List, Any, Dict, Callable
from dataclasses import dataclass

@dataclass(frozen=True)
class TaskKey:
    model_name: str
    task_name: str

@dataclass
class ModelError(Exception):
    exception: Exception

class InferenceModel:
    _task_registry: Dict[TaskKey, Callable] = {}
    
    def __init__(self) -> None:        
        tasks = ", ".join(self.get_task_names())
        print(f"Model '{self.__class__.__name__}' initiating with following tasks defined: {tasks}")
        
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
