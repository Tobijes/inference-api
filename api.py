from typing import List
from lib.api import InferenceAPI
from model import Model, ModelTaskType
from model_simple import SimpleModel, SimpleTaskType

app = InferenceAPI(model_type=[SimpleModel])

@app.post("/passage")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_task(SimpleModel, SimpleTaskType.PREDICT, data)
    return result

@app.post("/query")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_task(SimpleModel, SimpleTaskType.PREDICT, data)
    return result
