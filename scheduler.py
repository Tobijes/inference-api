from typing import List, Any, Dict, Type
import asyncio
from model import TaskType
import time 
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from process_functions import worker_create_model, worker_model_predict, worker_prepare_model
from collections import deque

@dataclass
class TaskElement:
    future: asyncio.Future
    data: Any

@dataclass
class TaskBatch:
    task_type: TaskType
    buffer: List[TaskElement]

class BatchScheduler:
    max_batch_size = 32
    max_wait_time = 0.05
    fill_queue_size_threshold = 3
    num_workers = 1

    def __init__(self, model_type: Type):
        task_types = [task_type for task_type in model_type.__annotations__.get('task_type')]
        self.pool = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=worker_create_model,
            initargs=(model_type,)
        )

        self.task_queues: Dict[TaskType, asyncio.Queue[TaskElement]]  = {}
        self.batch_queue: asyncio.Queue[TaskBatch] = asyncio.Queue()
        self.batch_sizes = deque(maxlen=10)

        # Create queues for each task type and startk worker,
        loop = asyncio.get_running_loop()
        for task_type in task_types:
            self.task_queues[task_type] = asyncio.Queue()
            loop.create_task(self.task_worker(task_type))

        for _ in range(self.num_workers):
            loop.create_task(self.batch_queue_worker())

    async def start(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.pool, worker_prepare_model)

    def stop(self):
        self.pool.shutdown()

    def get_queue_sizes(self):
        return {
            "batch_queue_size": self.batch_queue.qsize(),
            "batch_avg_size": sum(self.batch_sizes) / 10
        }

    async def submit_task(self, task_type: TaskType, data: List[Any]):
        queue = self.task_queues[task_type]
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in data]

        for (future, element) in zip(futures, data):
            batch_element = TaskElement(future, element)
            await queue.put(batch_element)

        await asyncio.gather(*futures)

        return [future.result() for future in futures]


    async def task_worker(self, task_type: TaskType):
        queue = self.task_queues[task_type]

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
            batch = TaskBatch(task_type=task_type, buffer=buffer)
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
            result = await loop.run_in_executor(self.pool, worker_model_predict, task_batch.task_type, data)
            # Set the individual element results
            for (f, r) in zip(futures, result):
                f.set_result(r)


