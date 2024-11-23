import math
from multiprocessing.managers import DictProxy
from time import perf_counter
from random import choices
import multiprocessing
from typing import Any, Dict, List, Type
import json
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np

import pynvml

from lib.model import InferenceModel

def print_result(result):
    data_points = []
    for key in filter(lambda x: "delta" in x, result.keys()):
        value = result[key]
        if "time" in key:
            value = str(math.ceil(value * 1000)) + "ms"
        if "memory" in key:
            value = str(math.ceil(value / 1024**2)) + " MiB"
        data_points.append(f"{key}: {value}")

    print(f"Batch Size: {result['batch_size']}, Max Length: {result['max_length']}, Sample: {result['sample']} | " + "\t".join(data_points))


def generate_length(s, length):
    if len(s) == length:
        return s
    
    if len(s) > length:
        return s[:length]
    
    rounds = math.ceil(length / len(s))
    s = " ".join([s] * rounds)
    return s[:length]


def run_single_task(model_type: Type[InferenceModel], task_name: str, batch: List[Any], result):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    model_before_time = perf_counter()
    model_before_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    model = model_type()
    model_after_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    model_after_time = perf_counter()

    model_delta_memory = model_after_memory - model_before_memory
    model_delta_time = model_after_time - model_before_time

    model.settings.MAX_BATCH_SIZE = len(batch)

    inference_before_time = perf_counter()
    inference_before_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    model.run_task(task_name, batch)
    inference_after_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    inference_after_time = perf_counter()

    inference_delta_memory = inference_after_memory - inference_before_memory
    inference_delta_time = inference_after_time - inference_before_time

    total_delta_memory = inference_after_memory - model_before_memory
    total_delta_time = inference_after_time - model_before_time

    result.update({
        "model_delta_memory": model_delta_memory,
        "model_delta_time": model_delta_time,
        "inference_delta_memory": inference_delta_memory,
        "inference_delta_time": inference_delta_time,
        "total_delta_memory": total_delta_memory,
        "total_delta_time": total_delta_time
    })

    pynvml.nvmlShutdown()


class BatchMemoryLimiter():
    model_type: Type[InferenceModel]
    config_path: Path
    corpus_path: Path
    memory_models: Dict[str, Pipeline] = {}

    def __init__(self, model_type: Type[InferenceModel], config_directory=Path("./evaluation")):
        self.model_type = model_type
        self.config_path = config_directory / f"{self.model_type.model_name}.memorymodel.json"
        self.corpus_path = Path(__file__).parent / "static" / "corpus.txt"

    def read_config(self):
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            return config
            
    def update_config(self, data):
        config = self.read_config()
        config = config | data
        # Save as JSON
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    # Extracting components
    def extract_memory_model(self, model):
        model_data = {
            'scaler_params':  {
                'mean': model.named_steps['scaler'].mean_.tolist(),
                'scale': model.named_steps['scaler'].scale_.tolist()
            },
            'poly_params': {
                'degree': model.named_steps['poly'].degree,
                'n_input_features': model.named_steps['poly'].n_features_in_
            },
            'linear_params': {
                'coef': model.named_steps['linear'].coef_.tolist(),
                'intercept': model.named_steps['linear'].intercept_.tolist()
            }
        }
        return model_data

    def learn(self):
        for task_name in self.model_type.get_task_names():
            data = self.generate_training_data(task_name)
            
            # Save for reproducability
            self.update_config({
                task_name: {
                    "data": data
                }
            })
            
            # Use training data to train regressor
            model = self.train_memory_model(task_name)
            self.memory_models[task_name] = model
            
            # Save for reuse
            model_data = self.extract_memory_model(model)
            self.update_config({
                task_name: {
                    "model": model_data
                }
            })


    def generate_training_data(self, task_name):
        
        batch_sizes = [1]#,2,4,8,16,24,32,40,48,56,64]
        lengths = [50]#, 100, 250, 500]
        n_samples = 1#3

        with open(self.corpus_path, "r") as f:
            corpus = f.readlines()
        
        results = []
        for batch_size in batch_sizes:
            for length in lengths:
                for sample in range(n_samples): 
                    batch = choices(corpus, k = batch_size)
                    batch = list(map(lambda x: generate_length(x, length), batch))
                    
                    sample_data = {
                        "sample": sample,
                        "batch_size" : batch_size,
                        "max_length" : max(map(len, batch)),
                        "task_name" : task_name,
                    }
                    manager = multiprocessing.Manager()
                    return_dict = manager.dict()
                    p = multiprocessing.Process(target=run_single_task, args=(self.model_type, task_name, batch, return_dict))
                    p.start()
                    p.join()

                    result = sample_data | dict(return_dict)
                    print_result(result)
                    results.append(result)
        return results


    def train_memory_model(self, task_name):
        config = self.read_config()
    
        # Extract data fields for numerical values (excluding task_name)
        data_fields = []
        for k,v in config[task_name]["data"][0].items():
            if isinstance(v, int) or isinstance(v, float):
                data_fields.append(k)
        

        data = config[task_name]["data"]
        print(data)
        # Prepare the list of rows for the NumPy array
        X_data = [[entry["batch_size"], entry["max_length"]] for entry in data]
        Y_data = [entry["total_delta_memory"] for entry in data]

        # Convert to NumPy array
        X = np.array(X_data)
        y = np.array(Y_data)
        print(X)
        print(y)

        # Polynomial regression
        polynomial_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])

        return polynomial_model.fit(X, y)