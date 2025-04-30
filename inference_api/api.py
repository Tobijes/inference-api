# First-party
import asyncio
from pathlib import Path
import logging
import json
from typing import Any
from contextlib import asynccontextmanager
from time import perf_counter
import os

# Third-party
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from prometheus_fastapi_instrumentator import Instrumentator

# Own
from .model import InferenceModel
from .scheduler import Scheduler
from inference_api.model import InferenceModel
from inference_api.api_models import HealthCheckModel
from inference_api.settings import SettingsLoader, BaseSettings
from inference_api.logging import EndpointFilter
from inference_api.storage import Storage
from inference_api.exceptions import APIHandledError, TaskCancelledError

VERSION = os.getenv("VERSION", "0.0.0")

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
class RequestInfoMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):

        # Start timer
        start_time = perf_counter()

        # Due to Starlette bug, Discussion #2094, we have to capture the disconnected function in first middleware
        request.state.is_disconnected = request.is_disconnected

        # Do actual call
        response = await call_next(request) 

        # Measure and set header
        process_time = perf_counter() - start_time
        response.headers["X-Request-Duration"] = f"{process_time:.6f}"
        response.headers["X-Request-Version"] = VERSION

        return response

class InferenceAPI(FastAPI):
    _scheduler: Scheduler
    logger: logging.Logger
    settings: BaseSettings
    storage: Storage

    def __init__(self,
            model_type: type[InferenceModel],
            redirect_to_docs = True,
            filter_log_paths = ["/health", "/metrics"],
            **kwargs
        ):
        super().__init__(
            lifespan=self.lifespan, 
            docs_url=None, 
            redoc_url=None, 
            openapi_tags=tags_metadata,
            version=VERSION,
            **kwargs)
        
        # Setup logging
        self.logger = logging.getLogger('uvicorn.error')

        # Load settings from model specified type
        self.settings = SettingsLoader.load_from_model(model_type)
        self.logger.info("Settings: %s", json.dumps(self.settings.__dict__, indent=2))

        # Create temporary storage space
        self.storage = Storage()

        # Create scheduler for model
        self._scheduler = Scheduler(model_type)

        # Add HTTP middleware
        self.add_middleware(RequestInfoMiddleware)

        # Add Prometheus
        self.instrumentator = Instrumentator()

        # Add static Swagger Docs UI files
        static_directory = Path(__file__).parent / "static"
        self.mount('/static', StaticFiles(directory=static_directory), name="static")

        # Add custom exception handler 
        self.add_exception_handler(APIHandledError, self.exception_handler)
        self.add_exception_handler(TaskCancelledError, self.task_cancelled_error_handler)

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
        if self._scheduler.pool is None:
            raise HTTPException(status_code=500, detail="Pool is none!")
        return HealthCheckModel(running=True)

    async def docs(self):
        return get_swagger_ui_html(
            openapi_url=self.openapi_url,
            title=self.title,
            swagger_favicon_url=f'/static/favicon-32x32.png',
            swagger_js_url=f'/static/swagger-ui-bundle.js',
            swagger_css_url=f'/static/swagger-ui.css'
        )

    async def exception_handler(self, request: Request, exc: APIHandledError):
        return JSONResponse(status_code=exc.http_status_code, content={"message": exc.message})
    
    async def task_cancelled_error_handler(self, request: Request, exc: TaskCancelledError):
        return Response(status_code=status.HTTP_204_NO_CONTENT)


    async def submit(self, request: Request, data: Any | list[Any], **kwargs):
        # Handle convenience of enablig both list of items and just a single item
        islist = isinstance(data, list)
        if not islist:
            data = [data]

        result = await self._scheduler.submit(
            is_cancelled=request.state.is_disconnected,
            data=data,
            **kwargs
        )

        # If convenience case used; return just the single item
        if not islist:
            result = result[0]
        
        return result