from typing import List
from lib.api import InferenceAPI
from model import Model
from model_simple import SimpleModel

app = InferenceAPI(model_type=[SimpleModel, Model])

@app.post("/passage")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_task(Model.query, data)
    return result

@app.post("/query")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_task(Model.query, data)
    return result

@app.post("/predict")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_task(SimpleModel.predict, data)
    return result