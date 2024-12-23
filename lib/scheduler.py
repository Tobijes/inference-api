from typing import List, Any, Dict, Type
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass 
from collections import deque
import asyncio

from lib.settings import BaseSettings, SettingsLoader

from .model import InferenceModel
from .process_functions import TaskResult, worker_create_model, worker_model_predict, worker_model_prepare
from .metrics import Metrics

@dataclass
class TaskElement:
    future: asyncio.Future
    data: Any

@dataclass
class TaskBatch:
    task_name: str
    buffer: List[TaskElement]

class Scheduler:
    model_type: Type[InferenceModel]
    metrics: Metrics

    def __init__(self, model_type: Type[InferenceModel]):
        self.model_type = model_type
        self.settings = SettingsLoader.load(BaseSettings)
        self.pool = ProcessPoolExecutor(
            max_workers=self.settings.POOL_WORKERS,
            initializer=worker_create_model,
            initargs=(model_type,)
        )
        # Initiate metrics
        self.metrics = Metrics(self.model_type)

        # Queue for the individual task elements before being batch grouped
        self.task_queues: Dict[str, asyncio.Queue[TaskElement]]  = {}
        # Queue for the batches of elements already batched up
        self.batch_queue: asyncio.Queue[TaskBatch] = asyncio.Queue()

        # Create queues for each task type and startk worker,
        loop = asyncio.get_running_loop()
        for task_name in self.model_type.get_task_names():
            loop.create_task(self.task_batcher_worker(task_name))
            self.task_queues[task_name] = asyncio.Queue()
            # Update metrics
            self.metrics.task_queue_size_gauge.labels(task_name).set(0)

        for _ in range(self.settings.POOL_WORKERS):
            loop.create_task(self.batch_queue_worker())

    async def start(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.pool, worker_model_prepare)

    def stop(self):
        self.pool.shutdown()


    async def submit_tasks(self, task_name: str, data: List[Any]):
        queue = self.task_queues[task_name]
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in data]

        for (future, element) in zip(futures, data):
            batch_element = TaskElement(future, element)
            await queue.put(batch_element)

        await asyncio.gather(*futures)

        return [future.result() for future in futures]


    async def task_batcher_worker(self, task_name: str):
        queue = self.task_queues[task_name]

        buffer = []
        while True: # Worker loop
            try:
                async with asyncio.timeout(self.settings.MAX_BATCH_WAIT_TIME / 1000.0):
                    while len(buffer) < self.settings.MAX_BATCH_SIZE : # Buffer fill loop
                        element = await queue.get()
                        buffer.append(element)
            except TimeoutError:
                pass
            
            if len(buffer) == 0:
                continue
            
            # If batch_queue is getting buffered, we might as well fill up the batches
            if self.batch_queue.qsize() > self.settings.FILL_QUEUE_SIZE_THRESHOLD \
            and len(buffer) < self.settings.MAX_BATCH_SIZE:
                continue

            # Send batch 
            batch = TaskBatch(task_name=task_name, buffer=buffer)
            await self.batch_queue.put(batch)
            # Clear buffer
            buffer = []

            # Update metrics
            self.metrics.task_queue_size_gauge.labels(task_name).set(queue.qsize())

    async def batch_queue_worker(self):
        while True:
            # Get task batch from queue
            task_batch: TaskBatch = await self.batch_queue.get()
            
            # Update metrics
            self.metrics.batch_queue_size_gauge.set(self.batch_queue.qsize())
            self.metrics.batch_size_histogram.observe(len(task_batch.buffer))

            # Split the task batch elements into native list
            futures = list(map(lambda x: x.future, task_batch.buffer))
            data = list(map(lambda x: x.data, task_batch.buffer))

            # Run the model with list of data
            loop = asyncio.get_running_loop()
            task_result = await loop.run_in_executor(self.pool, worker_model_predict, task_batch.task_name, data)

            # Handle error and do logging
            inference_log = f"Batch size: {len(data)} | {task_result.inference_time}ms | Task: {task_batch.task_name}" 
            if task_result.error is not None:
                print(inference_log + " | Had error")
                for f in futures:
                    f.set_exception(task_result.error)
                continue
            print(inference_log)

            # Set the individual element results
            for (f, r) in zip(futures, task_result.result):
                f.set_result(r)

            # Update metrics (only if no error)
            self.metrics.task_inference_time_histogram.labels(task_batch.task_name).observe(task_result.inference_time)


