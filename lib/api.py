from typing import List, Type, Any, Dict

from fastapi import FastAPI
from contextlib import asynccontextmanager

from .model import InferenceModel, TaskType
from .scheduler import BatchScheduler

class InferenceAPI(FastAPI):
    _schedulers: Dict[Type, BatchScheduler]

    def __init__(self, model_type: Type | List[Type]):
        super().__init__(lifespan=self.lifespan)
        self._schedulers = {}
        if isinstance(model_type, List):
            for mt in model_type:
                self._schedulers[mt] = BatchScheduler(mt)
        else:
            self._schedulers[model_type] = BatchScheduler(model_type)
        
        self.add_api_route("/health", self.health, methods=["GET"])
        self.add_api_route("/queue", self.get_queue_size, methods=["GET"])

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Load the ML model
        for (_, scheduler) in self._schedulers.items():
            await scheduler.start()
        yield
        # Clean up the ML models and release the resources
        for (_, scheduler) in self._schedulers.items():
            scheduler.stop()
    
    async def health(self):
        return "OK"

    async def get_queue_size(self):
        return self._schedulers.get_queue_sizes()
    
    async def submit_task(self, model_type: Type, task_type: TaskType, data: List[Any]):
        result = await self._schedulers[model_type].submit_task(task_type=task_type, data=data)
        return result