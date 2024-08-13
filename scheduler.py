from typing import List, Any, Dict, Type
import asyncio
from model import TaskType
import time 
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from process_functions import worker_create_model, worker_model_predict

@dataclass
class BatchElement:
    future: asyncio.Future
    data: Any

class BatchScheduler:
    max_batch_size = 32
    max_queue_time_ms = 100

    def __init__(self, model_type: Type):
        task_types = [task_type for task_type in model_type.__annotations__.get('task_type')]
        self.pool = ProcessPoolExecutor(
            max_workers=1,
            initializer=worker_create_model,
            initargs=(model_type,)
        )

        self.task_queues: Dict[TaskType, asyncio.Queue[BatchElement]]  = {}

        # Create queues for each task type and startk worker,
        loop = asyncio.get_running_loop()
        for task_type in task_types:
            self.task_queues[task_type] = asyncio.Queue()
            loop.create_task(self.worker(task_type))


    async def submit_task(self, data: List[Any], task_type: TaskType = None):
        queue = self.task_queues[task_type]
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in data]

        for (future, element) in zip(futures, data):
            batch_element = BatchElement(future, element)
            await queue.put(batch_element)

        await asyncio.gather(*futures)

        return [future.result() for future in futures]


    async def worker(self, task_type: TaskType):
        queue = self.task_queues[task_type]
        while True:
            batch_buffer: List[BatchElement] = []
            start_time = time.time_ns()
            while len(batch_buffer) < self.max_batch_size and (time.time_ns() - start_time) / 1000000 < self.max_queue_time_ms:
                if queue.qsize() > 0:
                    element = await queue.get()
                    batch_buffer.append(element)
                else:
                    await asyncio.sleep(0)   

            b_size = len(batch_buffer)
            if b_size == 0:
                continue
            
            futures = list(map(lambda x: x.future, batch_buffer))
            data = list(map(lambda x: x.data, batch_buffer))

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self.pool, worker_model_predict, task_type, data)

            for (f, r) in zip(futures, result):
                f.set_result(r)

            await asyncio.sleep(0)


