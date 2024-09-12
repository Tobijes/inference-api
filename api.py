from fastapi import FastAPI
from typing import List
import asyncio
from model import Model, ModelTaskType, SimpleModel
from scheduler import BatchScheduler
from contextlib import asynccontextmanager
scheduler = BatchScheduler(Model)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    await scheduler.start()
    yield
    # Clean up the ML models and release the resources
    scheduler.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/queue")
async def get_queue_size():
    return scheduler.get_queue_sizes()

@app.post("/passage")
async def predict(data: List[str]) -> List[List[float]]:
    result = await scheduler.submit_task(ModelTaskType.PASSAGE, data)
    return result

@app.post("/query")
async def predict(data: List[str]) -> List[List[float]]:
    result = await scheduler.submit_task(ModelTaskType.QUERY, data)
    return result

@app.get("/health")
async def health():
    return "OK"