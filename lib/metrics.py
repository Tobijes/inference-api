from typing import Dict, Type
from prometheus_client import Gauge, Histogram

from lib.model import InferenceModel

class Metrics:
    batch_queue_size_gauge = Gauge("batch_queue_size", documentation="Queue size for batch queue")
    batch_size_histogram = Histogram("batch_sizes", documentation="Batch sizes used", buckets=[1,2,4,6,8,16,32,64])
    task_inference_time_histogram: Histogram
    task_queue_size_gauge: Gauge

    def __init__(self, model_type: Type[InferenceModel]):
        self.task_inference_time_histogram = Histogram("task_inference_time", documentation=f"Inference time for task", labelnames=["task_name"], buckets=model_type.model_metrics_timing_buckets)
        self.task_queue_size_gauge = Gauge("task_queue_size", documentation=f"Queue size for task", labelnames=["task_name"])

    def get_instrumentations(self):
        return [
            self.batch_queue_size_gauge,
            self.batch_size_histogram,
            self.task_inference_time_histogram,
            self.task_queue_size_gauge
        ]
    