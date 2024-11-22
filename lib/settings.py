import typed_settings as ts
from typing import TypeVar, Type
from dataclasses import dataclass
from pathlib import Path

T = TypeVar('T')
APP_NAME = "INFERENCE"

# Environment variables are prefixed with 'INFERENCE_', example usage 'INFERENCE_USE_GPU=True'
@dataclass
class BaseSettings:
    POOL_WORKERS: int = 1
    USE_GPU: bool = True
    WARMUP: bool = True
    MAX_BATCH_SIZE = 32 # Max size of batch
    MAX_BATCH_WAIT_TIME = 0.05 # Max milliseconds to wait for filling up a batch 
    FILL_QUEUE_SIZE_THRESHOLD = 3 # Set queue size threshold for ignoring MAX_BATCH_WAIT_TIME

class SettingsLoader:

    @staticmethod
    def load(config_type: Type[T]) -> T:
        return ts.load(config_type, appname=APP_NAME)