from fastapi import FastAPI
from typing import List

from model import Model, ModelTaskType
from scheduler import BatchScheduler

app = FastAPI()
scheduler = BatchScheduler(Model)

@app.post("/passage")
async def predict(data: List[str]) -> List[List[float]]:
    result = await scheduler.submit_task(data, ModelTaskType.PASSAGE)
    return result

@app.get("/health")
async def health():
    return "OK"