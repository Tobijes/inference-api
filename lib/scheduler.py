from typing import List, Any, Dict, Type
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass 
from collections import deque
import asyncio

from .model import InferenceModel, ModelError
from .process_functions import worker_create_model, worker_model_predict, worker_prepare_model

@dataclass
class TaskElement:
    future: asyncio.Future
    data: Any

@dataclass
class TaskBatch:
    task_name: str
    buffer: List[TaskElement]

class BatchScheduler:
    max_batch_size = 32
    max_wait_time = 0.05
    fill_queue_size_threshold = 3
    num_workers = 1

    def __init__(self, model_type: Type[InferenceModel]):
        self.model_type = model_type
        self.pool = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=worker_create_model,
            initargs=(model_type,)
        )

        self.task_queues: Dict[str, asyncio.Queue[TaskElement]]  = {}
        self.batch_queue: asyncio.Queue[TaskBatch] = asyncio.Queue()
        self.batch_sizes = deque(maxlen=10)

        # Create queues for each task type and startk worker,
        loop = asyncio.get_running_loop()
        for task_name in self.model_type.get_task_names():
            self.task_queues[task_name] = asyncio.Queue()
            loop.create_task(self.task_worker(task_name))

        for _ in range(self.num_workers):
            loop.create_task(self.batch_queue_worker())

    async def start(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.pool, worker_prepare_model)

    def stop(self):
        self.pool.shutdown()

    def get_queue_size(self):
        return {
            "batch_queue_size": self.batch_queue.qsize(),
            "batch_avg_size": sum(self.batch_sizes) / 10
        }

    async def submit_task(self, task_name: str, data: List[Any]):
        queue = self.task_queues[task_name]
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in data]

        for (future, element) in zip(futures, data):
            batch_element = TaskElement(future, element)
            await queue.put(batch_element)

        await asyncio.gather(*futures)

        return [future.result() for future in futures]


    async def task_worker(self, task_name: str):
        queue = self.task_queues[task_name]

        buffer = []
        while True: # Worker loop
            try:
                async with asyncio.timeout(self.max_wait_time):
                    while len(buffer) < self.max_batch_size : # Buffer fill loop
                        element = await queue.get()
                        buffer.append(element)
            except TimeoutError:
                pass
            
            if len(buffer) == 0:
                continue
            
            # If batch_queue is getting buffered, we might as well fill up the batches
            if self.batch_queue.qsize() > self.fill_queue_size_threshold \
            and len(buffer) < self.max_batch_size:
                continue

            # Send batch 
            batch = TaskBatch(task_name=task_name, buffer=buffer)
            await self.batch_queue.put(batch)

            # Clear buffer
            buffer = []

    async def batch_queue_worker(self):
        while True:
            # Get task batch from queue
            task_batch: TaskBatch = await self.batch_queue.get()
            self.batch_sizes.append(len(task_batch.buffer))

            # Split the task batch elements into native list
            futures = list(map(lambda x: x.future, task_batch.buffer))
            data = list(map(lambda x: x.data, task_batch.buffer))

            # Run the model with list of data
            loop = asyncio.get_running_loop()
            
            inference_time, result = await loop.run_in_executor(self.pool, worker_model_predict, task_batch.task_name, data)
            inference_log = f"Batch size: {len(data)} | {inference_time}ms | Model: {self.model_type.__name__} | Task: {task_batch.task_name}" 
            if isinstance(result, ModelError):
                print(inference_log + " | Had error")
                for f in futures:
                    f.set_exception(result.exception)
                continue

            print(inference_log)
            # Set the individual element results
            for (f, r) in zip(futures, result):
                f.set_result(r)


