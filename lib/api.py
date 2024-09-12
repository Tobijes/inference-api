from typing import List

from fastapi import FastAPI
from contextlib import asynccontextmanager

from model import InferenceModel
from lib.scheduler import BatchScheduler

class InferenceAPI(FastAPI):
    scheduler: BatchScheduler

    def __init__(self, model_type: InferenceModel):
        super().__init__(lifespan=self.lifespan)
        self.scheduler = BatchScheduler(model_type)
        self.add_api_route("/health", self.health, methods=["GET"])
        self.add_api_route("/queue", self.get_queue_size, methods=["GET"])

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Load the ML model
        await self.scheduler.start()
        yield
        # Clean up the ML models and release the resources
        self.scheduler.stop()
    
    async def health(self):
        return "OK"

    async def get_queue_size(self):
        return self.scheduler.get_queue_sizes()