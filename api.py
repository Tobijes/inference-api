from typing import List
from lib.api import InferenceAPI
from model import Model, ModelTaskType
# from model_simple import SimpleModel

app = InferenceAPI(model_type=Model)

@app.post("/passage")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.scheduler.submit_task(ModelTaskType.PASSAGE, data)
    return result

@app.post("/query")
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.scheduler.submit_task(ModelTaskType.QUERY, data)
    return result
