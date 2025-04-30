class APIHandledError(Exception):
    http_status_code: int 
    message: str

class ModelError(APIHandledError):

    def __init__(self, message = "Error in model inference", http_status_code = 400):
        self.message = message
        self.http_status_code = http_status_code
    
class TaskCancelledError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__("Inference task was cancelled")