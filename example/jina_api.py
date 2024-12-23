# Run from current directory with 'uvicorn jina_api:app'
import sys, os
import aiofiles
import tempfile

from fastapi import UploadFile
sys.path.append(os.path.abspath(".."))

from typing import List

from lib.api import InferenceAPI, OPENAPI_TAGS_MODEL
from jina_clip import JinaClip

app = InferenceAPI(model_type=JinaClip)

@app.post("/texts", tags=OPENAPI_TAGS_MODEL)
async def predict(data: List[str]) -> List[List[float]]:
    result = await app.submit_tasks(JinaClip.texts, data)
    return result

@app.post("/images", tags=OPENAPI_TAGS_MODEL)
async def predict(data: List[UploadFile]) -> List[List[float]]:

    local_files = []
    
    # Process each uploaded file
    for file in data:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        local_files.append(temp_file.name)  # Keep track of temporary files

        async with aiofiles.open(temp_file.name, mode="wb") as f:
            while chunk := await file.read(1024 * 1024):  # Read in chunks of 1MB
                await f.write(chunk)

    result = await app.submit_tasks(JinaClip.images, local_files)
    return result
