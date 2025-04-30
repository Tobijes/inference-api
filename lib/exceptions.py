class ModelError(Exception):
    message: str 
    http_status_code: int

    def __init__(self, message = "Error in model inference", http_status_code = 400):
        self.message = message
        self.http_status_code = http_status_code

class BatchSizeExceededError(Exception):
    def __init__(self, supplied_batch_size: int, max_batch_size: int) -> None:
        self.supplied_batch_size = supplied_batch_size
        self.max_batch_size = max_batch_size
        super().__init__(f"Batch size {supplied_batch_size} exceeds the allowed limit of {max_batch_size}.")

    def __str__(self) -> str:
        return f"BatchSizeExceededError: Supplied batch size ({self.supplied_batch_size}) exceeds max ({self.max_batch_size})."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(supplied_batch_size={self.supplied_batch_size}, max_batch_size={self.max_batch_size})"

class UnknownTaskError(Exception):

    def __init__(self, model_name: str, task_name: str) -> None:
        self.model_name = model_name
        self.task_name = task_name
        super().__init__(f"In model '{self.model_name}' a task named '{self.task_name}' was not found.")

    def __str__(self) -> str:
        return f"UnknownTaskError: In model '{self.model_name}' a task named '{self.task_name}' was not found.)."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name}, task_name={self.task_name})"
    
class TaskCancelledError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__("Inference task was cancelled")