# Import routers from endpoint modules
from fastapi import APIRouter

# Create empty routers for now
# These would be implemented in their respective files
supervised_router = APIRouter()
unsupervised_router = APIRouter()
nlp_router = APIRouter()
cv_router = APIRouter()
timeseries_router = APIRouter()
rl_router = APIRouter()
deeplearning_router = APIRouter()
sensor_router = APIRouter()
tinyml_router = APIRouter()

# Import all implemented routers
# Uncomment these as they're implemented
# from .supervised_router import router as supervised_router
# from .unsupervised_router import router as unsupervised_router
# from .nlp_router import router as nlp_router
# from .cv_router import router as cv_router
# from .timeseries_router import router as timeseries_router
# from .rl_router import router as rl_router
# from .deeplearning_router import router as deeplearning_router
from .llm_router import router as llm_router
# from .sensor_router import router as sensor_router
# from .tinyml_router import router as tinyml_router

# Export all routers
__all__ = [
    "supervised_router",
    "unsupervised_router",
    "nlp_router",
    "cv_router",
    "timeseries_router",
    "rl_router",
    "deeplearning_router",
    "llm_router",
    "sensor_router",
    "tinyml_router",
] 