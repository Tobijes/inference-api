import typed_settings as ts
from typing import TypeVar, Type, get_type_hints
from dataclasses import dataclass

T = TypeVar('T')
APP_NAME = "INFERENCE"

# Environment variables are prefixed with 'INFERENCE_', example usage 'INFERENCE_USE_GPU=True'
@dataclass
class BaseSettings:
    POOL_WORKERS: int = 1 # Number of instances of the model to handle inference
    USE_GPU: bool = True # Use GPU if CUDA is available
    MAX_BATCH_SIZE: int = 32 # Max size of batch
    MAX_BATCH_WAIT_MS: int = 50 # Max milliseconds to wait for filling up a batch 

class SettingsLoader:

    @staticmethod
    def load(config_type: Type[T]) -> T:
        return ts.load(config_type, appname=APP_NAME)
    
    @staticmethod
    def load_from_model(model_type: Type[T]) -> BaseSettings:
        config_type = get_type_hints(model_type)["settings"]
        return ts.load(config_type, appname=APP_NAME)