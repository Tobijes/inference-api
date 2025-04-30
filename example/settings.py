# Run using: INFERENCE_MAX_BATCH_SIZE=32 uvicorn api:app
from dataclasses import dataclass
from lib import  BaseSettings
#
# To use model-specific settings, add a child class of BaseSettings (remember @dataclass)
# Notes:
# - Remember to add @dataclass
# - Environment variables are prefixed with 'INFERENCE_', example usage 'INFERENCE_USE_GPU=True'
#
@dataclass
class ModelSettings(BaseSettings):
    MAX_BATCH_WAIT_MS: int = 50 # Example of overiding base settings
    MAX_BATCH_SIZE: int = 32
    MY_CUSTOM_SETTING: str = "MySetting" # Example of model specific setting