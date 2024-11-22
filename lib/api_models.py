from pydantic import BaseModel

class HealthCheckModel(BaseModel):
    running: bool