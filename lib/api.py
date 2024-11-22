# First-party
from pathlib import Path
import logging
from typing import Callable, Any, Tuple, Type, List, Dict, Iterable
from contextlib import asynccontextmanager
from time import perf_counter

# Third-party
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Gauge
# Own
from .model import InferenceModel
from .scheduler import Scheduler
from lib.model import InferenceModel, ModelError
from lib.api_models import HealthCheckModel
from lib.settings import SettingsLoader, BaseSettings
from lib.logging import EndpointFilter

# OpenAPI Tags
OPENAPI_TAGS_MODEL = ["Model"]
OPENAPI_TAGS_SYSTEM = ["System"]
tags_metadata = [
    {
        "name": OPENAPI_TAGS_MODEL[0],
        "description": "Endpoints custom to the model",
    },
    {
        "name": OPENAPI_TAGS_SYSTEM[0],
        "description": "Endpoints related to the shared API system",
    },
]

HEALTH_ENDPOINT_DESCRIPTION = """
## Description
Endpoint for checking if worker pool and API is up.
"""
class RequestDurationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):

        # Start timer
        start_time = perf_counter()

        # Do actual call
        response = await call_next(request) 

        # Measure and set header
        process_time = perf_counter() - start_time
        response.headers["X-Request-Duration"] = f"{process_time:.6f}"

        return response

class InferenceAPI(FastAPI):
    _scheduler: Scheduler
    logger: logging.Logger
    settings: BaseSettings

    def __init__(self, 
            model_type: Type[InferenceModel],
            redirect_to_docs = True,
            filter_log_paths = ["/health", "/metrics"],
            **kwargs
        ):
        super().__init__(lifespan=self.lifespan, docs_url=None, redoc_url=None, openapi_tags=tags_metadata, **kwargs)
        self.logger = logging.getLogger('uvicorn.error')
        self.settings = SettingsLoader.load(BaseSettings)

        # Create scheduler for model
        self._scheduler = Scheduler(model_type)

        # Add Prometheus
        self.instrumentator = Instrumentator()

        # Add static Swagger Docs UI files
        static_directory = Path(__file__).parent / "static"
        self.mount('/static', StaticFiles(directory=static_directory), name="static")

        # Add custom exception handler 
        self.add_exception_handler(ModelError, self.model_error_handler)

        # Add standard API routes
        self.add_api_route("/docs", self.docs, methods=["GET"], include_in_schema=False) 
        self.add_api_route("/health", self.health, methods=["GET"], tags=OPENAPI_TAGS_SYSTEM, 
                           summary="System health check endpoints",
                           description=HEALTH_ENDPOINT_DESCRIPTION)
        
        # Add root redirection to docs for convenience
        if redirect_to_docs:
            self.add_api_route("/", lambda: RedirectResponse(url='/docs'), methods=["GET"], include_in_schema=False)

        # Add filter to specific paths (e.g. health check endpoint)
        for path in filter_log_paths:
            logging.getLogger('uvicorn.access').addFilter(EndpointFilter(path=path))

        # Add HTTP middleware
        self.add_middleware(RequestDurationMiddleware)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Load the ML model
        await self._scheduler.start()

        # Setup Prometheus
        for instrumentation in self._scheduler.metrics.get_instrumentations():
            self.instrumentator.add(instrumentation)
        self.instrumentator.expose(self, tags=OPENAPI_TAGS_SYSTEM)

        # Let FastAPI take over
        self.logger.info("Starting API")
        yield

        # After FastAPI end
        self.logger.info("API shutdown")

        # Clean up the ML model and release the resources
        self._scheduler.stop()

    async def health(self) -> HealthCheckModel:
        if self.pool == None:
            raise HTTPException(status_code=500, detail="Pool is none!")
        return HealthCheckModel(running=True)

    async def docs(self):
        return get_swagger_ui_html(
            openapi_url=self.openapi_url,
            title=self.title,
            swagger_favicon_url=f'/static/favicon.png',
            swagger_js_url=f'/static/swagger-ui-bundle.js',
            swagger_css_url=f'/static/swagger-ui.css'
        )

    async def model_error_handler(self, request: Request, exc: ModelError):
        return JSONResponse(
            status_code=exc.http_status_code,
            content=exc.message,
        )

    async def submit_task(self, task_signature, data: Any):
        task_key = InferenceModel.get_task_key(task_signature)
        result = await self._scheduler.submit_tasks(task_name=task_key.task_name, data=[data])
        return result[0]

    async def submit_tasks(self, task_signature, data: Iterable[Any]):
        task_key = InferenceModel.get_task_key(task_signature)
        result = await self._scheduler.submit_tasks(task_name=task_key.task_name, data=data)
        return result