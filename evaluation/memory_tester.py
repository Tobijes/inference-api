import time
from time import perf_counter
import threading
from typing import Callable

from sentence_transformers import SentenceTransformer
import torch
import gc
import math
from pynvml import *


nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
print(f"Device  : {nvmlDeviceGetName(handle)}")

def get_memory(handle, verbose=True):
    nvml_info = nvmlDeviceGetMemoryInfo(handle)
    nvml_total = nvml_info.total / 1024**2
    nvml_free = nvml_info.free / 1024**2
    nvml_used = nvml_info.used / 1024**2

    torch_info = torch.cuda.mem_get_info()
    torch_total = torch_info[1] /  1024**2
    torch_free = torch_info[0] /  1024**2
    torch_used = torch_total - torch_free

    torch_allocated = torch.cuda.memory_allocated() /  1024**2
    torch_max_allocated = torch.cuda.max_memory_allocated() /  1024**2
    if verbose:
        print(f"NVML: {nvml_used:.2f} MiB, Torch: {torch_used:.2f} MiB, Torch Allocated: {torch_allocated:.2f} MiB, Torch Max Allocated: {torch_max_allocated:.2f} MiB")
    return nvml_used


def profile_code(func: Callable, *args):
    before_time = perf_counter()
    before_memory = get_memory(handle, verbose=False)
    returnment = func(*args)
    after_memory = get_memory(handle, verbose=False)
    after_time = perf_counter()

    delta_memory = after_memory - before_memory
    delta_time = math.ceil((after_time - before_time) * 1000)

    return returnment, delta_memory, delta_time

def load_model():
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    return model

def run_model(model, texts):
    embeddings = model.encode(texts, prompt="passage: ", normalize_embeddings=True, batch_size=32)
    return embeddings

with open("./evaluation/corpus2.txt", "r") as f:
    texts = f.readlines()

print("Loading model")
model, delta_memory, delta_time = profile_code(load_model)
print(f"Loading model | dMemory: {delta_memory:.2f} MiB, dTime: {delta_time}ms")

print("Encode")
embeddings, delta_memory, delta_time = profile_code(run_model, model, texts)
print(f"Encode | dMemory: {delta_memory:.2f} MiB, dTime: {delta_time}ms")

print("Done")

nvmlShutdown()


# def loop_get_memory(stop_event):
#     while not stop_event.is_set():
#         get_memory(handle)
#         time.sleep(0.1)

# # get_memory(handle)
# stop_signal = threading.Event()
# t = threading.Thread(target=loop_get_memory, args=(stop_signal,))
# t.start()

# stop_signal.set()
# t.join()