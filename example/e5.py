import sys, os
sys.path.append(os.path.abspath(".."))

from typing import List
from time import sleep
from random import random

from sentence_transformers import SentenceTransformer
from lib.model import InferenceModel, ModelError
from settings import ModelSettings
    
class E5LargeModel(InferenceModel):
    model_metrics_timing_buckets = [10, 50, 100, 250, 500, 1000, 2500, 5000]
    settings: ModelSettings

    def __init__(self) -> None:      
        super().__init__() 
        self.logger.info("Loading model...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        self.logger.info("Model initiated on %s", self.model.device)

    @InferenceModel.task()
    def passage(self, texts: List[str]):
        embeddings = self.model.encode(texts, prompt="passage: ", normalize_embeddings=True)
        return embeddings
    
    @InferenceModel.task()
    def query(self, texts: List[str]):
        embeddings = self.model.encode(texts, prompt="query: ", normalize_embeddings=True)
        return embeddings