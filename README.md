**Note:** Requires Python 3.11

# Inference API
This repo is a reusable Python module to create consistent APIs for synchronous ML model inference workloads.
The module has the benefits:
- Easy to API'rize a model. 
- Handles process pooling ensuring that the event-loop of FastAPI is not blocked. Enables health checks, documentation requests and metrics to be accesible even when the model is inferencing and using most of the CPU.
- Creating this package allows for new learnings to be easily reused across APIs/models.

## Included in the package
The main class of the package derives from the usual `FastAPI` object class, but adds a lot of default things on top. This includes:
- Creates the `ProcessPool` and initates the defined Model class.
- Loads environment variable settings. These can be extended by providing a class extending `BaseSettings` (see example linked below)
- Sets up a logger and defines routes to be ignored, like `/health` and `/metrics` to avoid overblown logging file.
- Handles dynamic batching of multiple requests batched into a single inference pass
- Automatically handles errors and times the inference
- Sets up automatic Swagger documentation by adding required static files and `/docs` route
- Sets up instrumentation for Prometheus metrics collection. Default includes inference time histogram metric in high resolution and all API endpoints in low resolution.

## Mental model of the package
The goal of the package is to seperate the CPU/GPU intensive part of ML inference from the async web API non-blocking nature. Easy to use "Dynamic Batching" is also a huge part of it. Inference requests are received in the `API` using the FastAPI framework and then forwarded to the `Scheduler`. The `Scheduler` forwards batches to a `Model`. Batches are made from queues group by keyword arguments in the submission call. The `Model` receives batches along with keyword arguments in the `infer()` function. It is up the `Model` class code to use the keyword arguments for required purposes like 
- A `task` keyword argument for differentiating between multiple ways of embedding like "passage" or "query" in an text embedding model
- A `source_language` or `target_language` keyword argument to use a predefined machine translation pipeline
- A `modality` keyword argument for choosing preprocessing pipeline for multimodal models like CLIP that can embed both images and text.

Multiple requests with the same task are batched together for more efficient usage of the device. The Dynamic Batching algorithm can take the following into account:
1. Time since batch was started (`INFERENCE_MAX_BATCH_WAIT_MS`)
2. Statically defined maximum batch size (`INFERENCE_MAX_BATCH_SIZE`)
3. Element length aware batching **(Not implemented)**
    - Sort items and create batches with items of matching length to reduce required padding tokens

```mermaid
flowchart LR;

subgraph main["Main Process"]
    subgraph API["API"]
        E1["/text"]
        E2["/image"]
    end
    S["Scheduler"]
end

subgraph procpool["Process Pool"]
    subgraph w1["Worker 1 (Child Process)"]
        m1["Model<br>Max Size: 8"]
    end
    subgraph w2["Worker 2 (Child Process)"]
        m2["Model<br>Max Size: 8"]
    end
end

a["Client 1"] -- Size: 11 --> E1
b["Client 2"] -- Size: 3 --> E1
c["Client 3"] -- Size: 5 --> E2
E1 -- Modality: Text, Size: 11 --> S
E1 -- Modality: Text, Size: 3 --> S
E2 -- Modality: Image, Size: 5 --> S
S == Modality: Text, Size: 8 ==> m1
S == Modality: Image, Size: 5 ==> m1
S == Modality: Text, Size: 6 ==> m2
```

## About Process Pools
On of the primary goals of this package is to simply using a model in a Python `ProcessPool`. Think of this as splitting the API web requests handling workload from the model inference workload into two "programs" (ie. processes). We can then using Python's `await` from the API process to wait for a inference task to finish in the model process. This allows other web requests like health checks, metric collection, Swagger documentation etc. to be handled even while a model inference task is being awaited.

If we did not do this, all other web requests would be blocked, making the API appear unresponsive to users of documentation and systems relying on health checks like Docker Compose/Swarm or HAProxy.

### Notes
Note that calling model inference using the process pool is queued if a model inference task is already running. We default to a single worker (can be configured). 

Also note that the inference task queue is in memory, so if the service crashes all requests are lost.

Python has both `ThreadPool` and `ProcessPool` but since most models itself uses multiple threads, it makes more sense to split by process.


