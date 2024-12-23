import sys, os
sys.path.append(os.path.abspath(".."))

from typing import List
from time import perf_counter
from random import random

from sentence_transformers import SentenceTransformer
from lib.model import InferenceModel, ModelError
from settings import ModelSettings

from PIL import Image
    
class JinaClip(InferenceModel):
    model_metrics_timing_buckets = [10, 50, 100, 250, 500, 1000, 2500, 5000]
    settings: ModelSettings

    def __init__(self) -> None:      
        super().__init__() 
        self.logger.info("Loading model...")
        self.model = SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True, device=self.device)
        self.logger.info("Model initiated on %s", self.model.device)

    @InferenceModel.task()
    def texts(self, texts: List[str]):
        embeddings = self.model.encode(texts, task="text-matching", normalize_embeddings=True)
        return embeddings
    
    @InferenceModel.task()
    def images(self, image_paths: List[str]):
        print(perf_counter())
        images = [Image.open(image_path) for image_path in image_paths]
        print(perf_counter())
        embeddings = self.model.encode(images, normalize_embeddings=True)
        print(perf_counter())
        return embeddings