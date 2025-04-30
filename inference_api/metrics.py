from prometheus_client import Gauge, Histogram

from inference_api.model import InferenceModel
from inference_api.settings import BaseSettings

def compute_batch_size_buckets(max_size):
    buckets = []
    k = 1
    while k < max_size:
        buckets.append(k)
        k *= 2
    buckets.append(max_size)
    return buckets

class Metrics:
    items_queue_size_gauge = Gauge("items_queue_size", documentation="Queue size for submitted items")
    batch_size_histogram = Histogram
    batch_inference_time_histogram =  Histogram

    def __init__(self, model_type: type[InferenceModel], settings: BaseSettings):

        self.batch_size_histogram = Histogram(
            name="batch_size",
            documentation="Histogram for batch sized used",
            buckets=compute_batch_size_buckets(settings.MAX_BATCH_SIZE)
        )
        
        self.batch_inference_time_histogram = Histogram(
            name="batch_inference_time",
            documentation="Queue size for task",
            buckets=model_type.model_metrics_timing_buckets
        )

    def get_instrumentations(self):
        return [
            self.items_queue_size_gauge,
            self.batch_size_histogram,
            self.batch_inference_time_histogram
        ]
    
