from typing import List, Type, Any, Dict, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .model import InferenceModel, ModelException
from .scheduler import BatchScheduler



class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

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
        self.add_exception_handler(ModelException, self.model_exception_handler)

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

    async def model_exception_handler(self, request: Request, exc: ModelException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": str(exc)},
        )

    async def health(self):
        return "OK"

    async def get_queue_sizes(self):
        return {model_type: scheduler.get_queue_size() for (model_type, scheduler) in self._schedulers.items()}
    
    async def submit_task(self, task_signature, data: Any):
        task_key = InferenceModel.get_task_key(task_signature)
        result = await self._schedulers[task_key.model_name].submit_tasks(task_name=task_key.task_name, data=[data])
        return result[0]

    async def submit_tasks(self, task_signature, data: Iterable[Any]):
        task_key = InferenceModel.get_task_key(task_signature)
        result = await self._schedulers[task_key.model_name].submit_tasks(task_name=task_key.task_name, data=data)
        return result