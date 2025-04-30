from locust import HttpUser, task, events
import random
from pathlib import Path
import gevent.lock

SAMPLE_TEXT = "Hello, world."

success_counter = 0
counter_lock = gevent.lock.Semaphore()

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print(f"Running total of processed items: {success_counter}")

class ApiUser(HttpUser):
    abstract = True
    endpoint = "/"

    def request(self, request_body, n_items):
        with self.client.post(url=self.endpoint, json=request_body, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed with status {response.status_code}. Response: {response.text}")
            
            # Calculate processing time per item
            total_time = response.elapsed.total_seconds()
            avg_time_per_item = total_time / n_items
            
            # Record custom metrics
            self.environment.events.request.fire(
                request_type='custom',
                name='avg_time_per_text',
                response_time=avg_time_per_item * 1000,  # Convert to ms
                response_length=len(response.content),
                exception=None
            )

            global success_counter
            with counter_lock:
                success_counter += n_items

class SingleTest(ApiUser):
    endpoint = "/predict" 

    @task
    def test(self):

        self.request(
            request_body={
            "text": SAMPLE_TEXT, 
            },
            n_items=1
        )
        

class BatchText(ApiUser):
    endpoint = "/batch" 

    @task
    def test(self):
        """Test different batch sizes"""
        sample_text = "Hello, world."
        batch_size = random.randint(1,20)
        texts = [sample_text] * batch_size

        self.request(
            request_body={
            "texts": texts, 
            },
            n_items=batch_size
        )
        