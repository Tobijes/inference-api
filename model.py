from typing import List, Any, Callable
from enum import Enum
from sentence_transformers import SentenceTransformer

class TaskType(str, Enum):
    pass

class InferenceModel():
    task_type: TaskType

    def __init__(self) -> None:        
        # Internal
        self._tasks = {}

    def task(self, task_type: TaskType, data: List[Any]):
        handler = self._tasks.get(task_type, self._default_handler)
        return handler(data)
            
    def _default_handler(self, data):
        return data
    
    def register_task_handler(self, task_type: TaskType, handler: Callable):
        self._tasks[task_type] = handler


### DEVELOPER ###
class ModelTaskType(TaskType):
    PASSAGE = "PASSAGE"
    QUERY = "QUERY"

class Model(InferenceModel):
    task_type: ModelTaskType

    def __init__(self) -> None:      
        super().__init__() 

        self.model = SentenceTransformer('intfloat/multilingual-e5-large')

        self.register_task_handler(ModelTaskType.PASSAGE, self.handle_passage)
        self.register_task_handler(ModelTaskType.QUERY, self.handle_query)

    def handle_passage(self, texts: List[str]):
        embeddings = self.model.encode(texts, prompt="passage: ", normalize_embeddings=True)
        return embeddings
    
    def handle_query(self, texts: List[str]):
        embeddings = self.model.encode(texts, prompt="query: ", normalize_embeddings=True)
        return embeddings
