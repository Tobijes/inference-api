from typing import List
from sentence_transformers import SentenceTransformer

from lib.model import InferenceModel, TaskType
    
class ModelTaskType(TaskType):
    PASSAGE = "PASSAGE"
    QUERY = "QUERY"

class Model(InferenceModel):
    task_type: ModelTaskType

    def __init__(self) -> None:      
        super().__init__() 
        print("Loading model...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        print("Model initiated on", self.model.device)

    @InferenceModel.task(ModelTaskType.PASSAGE)
    def handle_passage(self, texts: List[str]):
        embeddings = self.model.encode(texts, prompt="passage: ", normalize_embeddings=True)
        return embeddings
    
    @InferenceModel.task(ModelTaskType.QUERY)
    def handle_query(self, texts: List[str]):
        embeddings = self.model.encode(texts, prompt="query: ", normalize_embeddings=True)
        return embeddings

