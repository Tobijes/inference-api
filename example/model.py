from typing import List
from time import sleep
from random import random

from sentence_transformers import SentenceTransformer
from lib.model import InferenceModel, ModelError
    
class Model(InferenceModel):
    model_name = "ExampleModel"
    
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