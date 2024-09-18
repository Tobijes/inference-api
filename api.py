from typing import List
from fastapi import HTTPException

from lib.api import InferenceAPI
from model import Model
from model_simple import SimpleModel

app = InferenceAPI(model_type=[SimpleModel, Model])
# app = InferenceAPI(model_type=[SimpleModel])

@app.post("/passage")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_tasks(Model.passage, data)
    return result

@app.post("/query")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_tasks(Model.query, data)
    return result

@app.post("/batch")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_tasks(SimpleModel.predict, data)
    return result

@app.post("/predict")
async def predict(data: str) -> List[float]:
    result = await app.submit_task(SimpleModel.predict, data)
    return result

@app.post("/unknown")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_task(SimpleModel.simulate_unknown_error, data)

@app.post("/known")
async def predict(data: List[str]) -> List[List[float]]:
    await app.submit_task(SimpleModel.simulate_known_error, data)
    