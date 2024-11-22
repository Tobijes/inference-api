from typing import List
from time import sleep
from random import random

from sentence_transformers import SentenceTransformer
from lib.model import InferenceModel, ModelError
    
class Model(InferenceModel):

    def __init__(self) -> None:      
        super().__init__() 
        print("Loading model...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        print("Model initiated on", self.model.device)

    @InferenceModel.task()
    def passage(self, texts: List[str]):
        embeddings = self.model.encode(texts, prompt="passage: ", normalize_embeddings=True)
        return embeddings
    
    @InferenceModel.task()
    def query(self, texts: List[str]):
        embeddings = self.model.encode(texts, prompt="query: ", normalize_embeddings=True)
        return embeddings
    
    @InferenceModel.task()
    def predict(self, texts: List[str]):
        sleep(0.1 + 0.005 * len(texts))
        return [[random() * 50] * 768] * len(texts)

    @InferenceModel.task()
    def simulate_unknown_error(self, data):
        raise ValueError("Value small")
    
    @InferenceModel.task()
    def simulate_known_error(self, data):
        raise ModelError("Model hiccup")