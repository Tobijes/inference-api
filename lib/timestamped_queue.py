from typing import Any
from asyncio import Queue
from time import perf_counter_ns

class TimestampedWrapper:
    """
    A class that wraps an arbitrary object with a timestamp. Mostly for use with TimestampedQueue
    """
    created_time: int
    data: Any

    def __init__(self, data: Any) -> None:
        self.created_time = perf_counter_ns()
        self.data = data

class TimestampedQueue:
    """
    A class that mimics a asyncio.Queue with the exception that we can track put time of the oldest object
    Illustration of the queue
            TAIL         HEAD
    -> [ X , X , X ] -> [ X ] ->
    Items are put into the tail queue and progress along the queue until the item is moved to the head queue
    
    Why a single-element head queue? It allows for easy awaiting in the scenario of a completely empty queue
    """

    _tail: Queue[TimestampedWrapper]
    _head: Queue[TimestampedWrapper]
    head_time: int

    def __init__(self) -> None:
        self._tail = Queue(maxsize=-1) # Infinite queue
        self._head = Queue(maxsize=1) # Single item queue
        self.head_time = None

    async def put(self, data: Any):
        wrapper = TimestampedWrapper(data)

        if self._head.qsize() == 0:
            # Set new head and head_time
            await self._head.put(wrapper)
            self.head_time = wrapper.created_time
        else:
            await self._tail.put(wrapper)
    
    async def get(self) -> Any:
        head = await self._head.get()
        
        # If tail is not empty we should get a new head before returning
        if self._tail.qsize() > 0:
            # Pop tail for new head
            oldest_tail = await self._tail.get()
            # Set new head and head_time
            await self._head.put(oldest_tail)
            self.head_time = oldest_tail.created_time
        else:
            self.head_time = None
        return head.data
    
    def qsize(self) -> int:
        return self._tail.qsize() + self._head.qsize()