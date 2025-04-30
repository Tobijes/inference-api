# Mess with path to get example to import package
import sys, os
sys.path.append(os.path.abspath(".."))

from time import sleep
from random import random
from pathlib import Path

from lib.model import InferenceModel
from lib.exceptions import ModelError
from settings import ModelSettings

class SimpleModel(InferenceModel):

    # Override/extend settings
    settings: ModelSettings
    # Override inference timing buckets
    model_metrics_timing_buckets = [10, 50, 100, 500]

    def __init__(self) -> None:      
        super().__init__() 
        self.logger.info("Custom setting %s", self.settings.MY_CUSTOM_SETTING)

    def infer(self, data: list[str], task=None, **kwargs):
        match task:
            case "texts":
                return self.texts(data)
            case "images":
                return self.images(data)
            case "unknown_error": 
                self.simulate_unknown_error()
            case "known_error":
                self.simulate_known_error()

    def texts(self, texts: list[str]):
        sleep(0.1 + 0.005 * len(texts))
        return [[random() * 50] * 768] * len(texts)
    
    def images(self, image_paths: list[Path]):
        sleep(0.1 + 0.005 * len(image_paths))
        return [[random() * 50] * 768] * len(image_paths)

    def simulate_unknown_error(self):
        raise ValueError("Value small")    

    def simulate_known_error(self):
        try:
            raise ValueError("Value small")
        except Exception:
            raise ModelError("Simulated handled known error", http_status_code=415)