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
    MODEL_CACHE: str = str(Path.home()) + "/.cache/inference_api_models"

class SettingsLoader:

    @staticmethod
    def load(config_type: Type[T]) -> T:
        return ts.load(config_type, appname=APP_NAME)