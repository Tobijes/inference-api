# Mess with path to get example to import package
import sys, os
sys.path.append(os.path.abspath(".."))

# Third-party
from pydantic import BaseModel, Field
from fastapi import File, UploadFile, Request

# Imports
from inference_api import InferenceAPI, OPENAPI_TAGS_MODEL
from simple_model import SimpleModel

Vector = list[float]
#
# API defintion
#
title = "My Test Model Inference API"
description = """
### Description
This a description of the **Test Model**
"""

app = InferenceAPI(SimpleModel, title=title, description=description)

#
# Endpoints for testing basic functionality
#
@app.post("/ping", tags=OPENAPI_TAGS_MODEL, summary="Primary model prediction endpoint")
async def ping(data: list[str]) -> list[Vector]:
    return [[0.0, 0.0, 0.0]]

@app.post("/known_error", tags=OPENAPI_TAGS_MODEL, summary="Endpoint for testing known error handling")
async def known_error(request: Request):
    result = await app.submit(request, None, task="known_error")
    return result

@app.post("/unknown_error", tags=OPENAPI_TAGS_MODEL, summary="Endpoint for testing unknown error handling")
async def unknown_error(request: Request):
    result = await app.submit(request, None, task="unknown_error")
    return result

#
# Endpoint for testing single element inference
#
class PredictInputRequest(BaseModel):
    text: str = Field(example="My String", min_length=1)

@app.post("/predict", tags=OPENAPI_TAGS_MODEL, summary="Primary model prediction endpoint")
async def predict(request: Request, data: PredictInputRequest) -> Vector:
    """
    ## Endpoint Description
    This is a short summary of what the endpoints does
    - We can even use **Markdown**
    """
    result = await app.submit(request, data.text, task="texts")
    return result


#
# Endpoint for testing batch of elements inference
#
class PredictBatchInputRequest(BaseModel):
    texts: list[str] = Field(example=["My String"], min_length=1)

@app.post("/batch", tags=OPENAPI_TAGS_MODEL, summary="Primary model prediction endpoint")
async def predict_batch(request: Request, data: PredictBatchInputRequest) -> list[Vector]:
    """
    ## Endpoint Description
    This is a short summary of what the endpoints does
    - We can even use **Markdown**
    """
    result = await app.submit(request, data.texts, task="texts")
    return result

#
# Endpoint for testing inference of file-like elements
#
@app.post('/files', tags=OPENAPI_TAGS_MODEL)
async def predict_files(request: Request, files: list[UploadFile] = File(...)) -> list[list[float]]:
    # Save files to disk, to be loaded by other process
    file_paths = await app.storage.save_temporary_media(files)

    try:
        # Do inference on file list
        feature_vectors = await app.submit(request, file_paths, task="files")
    finally:
        # Cleanup stored files
        app.storage.delete_temporary_media(file_paths)

    return feature_vectors