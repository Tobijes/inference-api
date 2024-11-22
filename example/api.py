from typing import List
from fastapi import HTTPException

from lib.api import InferenceAPI, OPENAPI_TAGS_MODEL
from .model import Model
from .simple_model import SimpleModel

app = InferenceAPI(model_type=Model)
# app = InferenceAPI(model_type=SimpleModel)

@app.post("/passage", tags=OPENAPI_TAGS_MODEL)
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_tasks(Model.passage, data)
    return result

@app.post("/query", tags=OPENAPI_TAGS_MODEL)
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_tasks(Model.query, data)
    return result

@app.post("/batch", tags=OPENAPI_TAGS_MODEL)
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_tasks(SimpleModel.predict, data)
    return result

@app.post("/predict", tags=OPENAPI_TAGS_MODEL)
async def predict(data: str) -> List[float]:
    result = await app.submit_task(SimpleModel.predict, data)
    return result

@app.post("/unknown", tags=OPENAPI_TAGS_MODEL)
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_task(SimpleModel.simulate_unknown_error, data)

@app.post("/known", tags=OPENAPI_TAGS_MODEL)
async def predict(data: List[str]) -> List[List[float]]:
    await app.submit_task(SimpleModel.simulate_known_error, data)
    