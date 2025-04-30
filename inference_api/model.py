import os
import logging
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass

from inference_api.settings import BaseSettings, SettingsLoader
from inference_api.utils import is_cuda_available
from inference_api.logging import WorkerAdapter

@dataclass(frozen=True)
class TaskKey:
    model_name: str
    task_name: str


class InferenceModel:
    _task_registry: dict[TaskKey, Callable] = {}

    model_metrics_timing_buckets = [50, 100, 500, 1000, 5000, 10000]

    settings: BaseSettings
    device: str = "cpu"
    process_id: int
    
    def __init__(self) -> None:       
        self.process_id = os.getpid() 
        self.logger = WorkerAdapter(logging.getLogger('uvicorn.error'), None, worker_id=self.process_id)
        self.logger.info("Model '%s' initiating", self.__class__.__name__)
        # Load settings
        self.settings = SettingsLoader.load_from_model(type(self))
        # Set device CPU/CUDA
        self.device = "cuda" if (is_cuda_available() and self.settings.USE_GPU) else "cpu"
        
    def infer(self, data: list[Any], **kwargs):
        raise NotImplementedError