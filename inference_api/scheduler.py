from typing import Any
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
import asyncio
import logging
from time import perf_counter_ns
from functools import partial

from inference_api.settings import  SettingsLoader

from .model import InferenceModel
from .timestamped_queue import TimestampedQueue
from .process_functions import worker_create_model, worker_model_predict, worker_model_prepare
from .metrics import Metrics
from .exceptions import TaskCancelledError

# Meaningful type hints
BatchableData = Any

@dataclass
class Item:
    future: asyncio.Future
    data: BatchableData
    is_cancelled: Callable[[], Coroutine[None, None, bool]]

@dataclass
class Batch:
    buffer: list[Item]
    kwargs: dict

@dataclass
class KwargsTask:
    kwargs: dict
    queue: TimestampedQueue

class Scheduler:
    model_type: type[InferenceModel]
    metrics: Metrics

    kwargs_tasks: dict[str, KwargsTask]

    def __init__(self, model_type: type[InferenceModel]):
        self.model_type = model_type
        self.logger = logging.getLogger('uvicorn.error')
        self.settings = SettingsLoader.load_from_model(model_type)
        self.pool = ProcessPoolExecutor(
            max_workers=self.settings.POOL_WORKERS,
            initializer=worker_create_model,
            initargs=(model_type,)
        )
        # Initiate metrics
        self.metrics = Metrics(self.model_type, self.settings)
        
        # Create queue task management dict
        self.kwargs_tasks = {}
        
        loop = asyncio.get_running_loop()
        loop.create_task(self.create_batches_worker())
        # Queue for the batches of elements already batched up
        self.batch_queue: asyncio.Queue[Batch] = asyncio.Queue(maxsize=self.settings.POOL_WORKERS+1)
        # Start batch queue workers
        for _ in range(self.settings.POOL_WORKERS):
            loop.create_task(self.model_queue_worker())

        # Update metrics
        self.metrics.items_queue_size_gauge.set(0)

    async def start(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.pool, worker_model_prepare)

    def stop(self):
        self.pool.shutdown()

    def queue_key(self, **kwargs):
        return "-".join([f"{k}:{v}" for k,v in kwargs.items()])

    def get_kwargs_task(self, **kwargs):
        key = self.queue_key(**kwargs)

        if key in self.kwargs_tasks:
            return self.kwargs_tasks[key]
        
        self.kwargs_tasks[key] = KwargsTask(
            kwargs=kwargs,
            queue=TimestampedQueue()
        )
        self.logger.info("Creating kwargs task queue with key: %s", key)
        return self.kwargs_tasks[key]

    async def submit(self, is_cancelled: Callable[[], Coroutine[None, None, bool]], data: list[Any], **kwargs):
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in data]

        kwargs_task = self.get_kwargs_task(**kwargs)

        for (future, element) in zip(futures, data):
            await kwargs_task.queue.put(Item(future, element, is_cancelled))

        self.metrics.items_queue_size_gauge.inc(len(data))

        try:
            await asyncio.gather(*futures)
        except asyncio.CancelledError as ce:
            raise TaskCancelledError() from ce

        return [future.result() for future in futures]


    async def create_batches_worker(self):
        while True: 
            # Scan all queues for oldest item. Select this kwargs_task
            
            # Filter empty queues and compute combined queue size:
            keyset = set(self.kwargs_tasks)
            for key in keyset:
                # Check for empty queue and remove them
                # This is safe as no one can be awaiting get() as this function is only 
                #  called from the batcher worker which is the only one getting from the queue
                if self.kwargs_tasks[key].queue.qsize() == 0:
                    self.logger.info("Removing kwargs task queue with key: %s", key)
                    self.kwargs_tasks.pop(key)

            # Check that any queues are active
            if len(self.kwargs_tasks) == 0:
                await asyncio.sleep(0.1)
                continue
            
            # Create iterator and get first item
            iterator = iter(self.kwargs_tasks.values())
            kwargs_task = next(iterator)
            # Continue iterating and check for older heads
            for task in iterator:
                # Check if current_task in iterator is older
                if task.queue.head_time < kwargs_task.queue.head_time:
                    kwargs_task = task


            # When task is found try to fill up batch
            buffer: list[Item] = []
            try:
                async with asyncio.timeout(self.settings.MAX_BATCH_WAIT_MS / 1000.0):
                    while len(buffer) < self.settings.MAX_BATCH_SIZE : # Buffer fill loop
                        # Wait for element in queue and add to buffer
                        element = await kwargs_task.queue.get()
                        buffer.append(element)
            except TimeoutError:
                if len(buffer) == 0:
                    continue
            
            # Send batch 
            batch = Batch(buffer=buffer, kwargs=kwargs_task.kwargs)
            # Notize batch_queue is small (maxsize) meaning worker will await, which allows next batches to be more filled
            await self.batch_queue.put(batch) 

            # Clear buffer
            buffer = []

    async def model_queue_worker(self):
        while True:
            # Get task batch from queue
            batch = await self.batch_queue.get()

            # Update metrics
            self.metrics.items_queue_size_gauge.dec(len(batch.buffer))
            
            # Filter task batch elements for cancelled elements
            is_cancelleds = list(map(lambda x: x.is_cancelled(), batch.buffer))
            is_cancelled_buffer = await asyncio.gather(*is_cancelleds)
            cancel_buffer = [task for task, is_cancelled in zip(batch.buffer, is_cancelled_buffer) if is_cancelled]
            batch_buffer = [task for task, is_cancelled in zip(batch.buffer, is_cancelled_buffer) if not is_cancelled]

            for task in cancel_buffer:
                task.future.cancel()
            
            # Strings for logging
            cancelled_str = f"(Cancelled: {len(cancel_buffer)}) " if len(cancel_buffer) > 0 else ""
            kwargs_str = ",".join([f"'{k} = {v}'" for k,v in batch.kwargs.items()])
                
            if len(batch_buffer) == 0:
                self.logger.info("Batch size: %d {%s}| Kwargs: {%s} | Skipping", len(batch_buffer), cancelled_str, kwargs_str)
                continue

            # Update metrics
            self.metrics.batch_size_histogram.observe(len(batch_buffer))

            # Split the task batch elements into native list
            futures = list(map(lambda x: x.future, batch_buffer))
            data = list(map(lambda x: x.data, batch_buffer))

            # Run the model with list of data
            loop = asyncio.get_running_loop()
            task_result = await loop.run_in_executor(self.pool, partial(worker_model_predict, data, **batch.kwargs))
   
            # Handle error and do logging
            inference_log = f"Worker ID: {task_result.process_id} | Batch size: {len(data)} {cancelled_str}| Time: {task_result.inference_time}ms | Kwargs: {kwargs_str}" 
            if task_result.error is not None:
                self.logger.error("%s | %s", inference_log, task_result.error.message)
                for f in futures:
                    f.set_exception(task_result.error)
                continue
            self.logger.info(inference_log)

            # Set the individual element results
            for (f, r) in zip(futures, task_result.result):
                f.set_result(r)

            # Update metrics (only if no error)
            self.metrics.batch_inference_time_histogram.observe(task_result.inference_time)

