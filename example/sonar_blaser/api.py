# Mess with path to get example to import package
import sys, os
sys.path.append(os.path.abspath("../.."))

import asyncio

# Third-party
from pydantic import BaseModel, Field
from fastapi import Request

# Imports
from inference_api import InferenceAPI, OPENAPI_TAGS_MODEL
from model import SonarBlaserText

Vector = list[float]
#
# API defintion
#
title = "Translation Evaluation Inference API"
description = """
### Description
Using Facebook AI Research's **SONAR** for multilingual embedding and **BLASER** model for evaluation of translations
"""

app = InferenceAPI(SonarBlaserText, title=title, description=description)

#
# Endpoint for testing batch of elements inference
#
class SonarEmbeddingsInputRequest(BaseModel):
    texts: list[str] = Field(example=["My String"], min_length=1)
    src_lang: str = Field(example="eng_Latn", min_length=8, max_length=8)

@app.post("/sonar_embeddings", tags=OPENAPI_TAGS_MODEL, summary="Primary model prediction endpoint")
async def predict_sonar_embeddings(request: Request, body: SonarEmbeddingsInputRequest) -> list[Vector]:
    result = await app.submit(request, body.texts, src_lang=body.src_lang, task="SONAR")
    return result

class ScoreInputRequest(BaseModel):
    original_texts: list[str] = Field(example=[
        "Hej, husk at tænde ovnen!", 
        "Den skal nemlig slukkes."
        ], min_length=1)
    translated_texts: list[list[str]] = Field(example=[
            ["Hey, remember to turn on the oven!", "It needs to be turned off."],
            ["Hey, don't forget to light the oven!", "It must be extinguished"]
        ], min_length=1)
    src_lang: str = Field(example="dan_Latn", min_length=8, max_length=8)
    tgt_lang: str = Field(example="eng_Latn", min_length=8, max_length=8)

@app.post("/score", tags=OPENAPI_TAGS_MODEL, summary="Primary model prediction endpoint")
async def predict_translation_score(request: Request, body: ScoreInputRequest) -> list[float]:
    # Create list of original_text embeddings and translated text embeddings [org, mt, ..., mt]
    embeddings_future = [
        app.submit(request, body.original_texts, src_lang=body.src_lang, task="SONAR")
    ]

    for translated_texts in body.translated_texts:
        embeddings_future.append(
            app.submit(request, translated_texts, src_lang=body.tgt_lang, task="SONAR")
        )

    original_text_embeddings, *translated_text_embeddings = await asyncio.gather(*embeddings_future)
    print(original_text_embeddings)
    # Measure each translated text embedding against original text
    score_futures = []
    for original_text_embedding, translated_text_embedding in zip([original_text_embeddings] * len(translated_text_embeddings), translated_text_embeddings):
        score_futures.append(app.submit(request, (original_text_embedding, translated_text_embedding), task="BLASER"))

    scores = await asyncio.gather(*score_futures)

    return scores

