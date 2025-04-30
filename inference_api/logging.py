import logging
from typing import Any

class EndpointFilter(logging.Filter):
    def __init__(
        self,
        path: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._path = path

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self._path) == -1


class WorkerAdapter(logging.LoggerAdapter):
    worker_id: int

    def __init__(self, logger, extra, worker_id):
        super().__init__(logger, extra)
        self.worker_id = worker_id 

    def process(self, msg, kwargs):
        return f"[Worker {self.worker_id}] {msg}", kwargs