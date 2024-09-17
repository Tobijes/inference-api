from typing import List, Type, Any, Dict

from fastapi import FastAPI
from contextlib import asynccontextmanager

from .model import InferenceModel
from .scheduler import BatchScheduler

class InferenceAPI(FastAPI):
    _schedulers: Dict[str, BatchScheduler] = {}

    def __init__(self, model_type: Type[InferenceModel] | List[Type[InferenceModel]]):
        super().__init__(lifespan=self.lifespan)

        # Create schedulers for model types
        if isinstance(model_type, List):
            for mt in model_type:
                self.add_scheduler(mt)
        else:
            self.add_scheduler(model_type)
        
        # Add standard API routes
        self.add_api_route("/health", self.health, methods=["GET"])
        self.add_api_route("/queue", self.get_queue_sizes, methods=["GET"])

    def add_scheduler(self, model_type: Type[InferenceModel]):
        self._schedulers[model_type.__name__] = BatchScheduler(model_type)

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

    async def get_queue_sizes(self):
        return {model_type: scheduler.get_queue_size() for (model_type, scheduler) in self._schedulers.items()}
    
    async def submit_task(self, task_signature, data: List[Any]):
        task_key = InferenceModel.get_task_key(task_signature)
        result = await self._schedulers[task_key.model_name].submit_task(task_name=task_key.task_name, data=data)
        return result