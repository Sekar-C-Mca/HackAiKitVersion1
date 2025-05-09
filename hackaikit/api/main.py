from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List, Dict, Any

from hackaikit.core.config_manager import ConfigManager
from hackaikit.core.module_manager import ModuleManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hackai-kit-api")

# Create FastAPI app with metadata
app = FastAPI(
    title="HackAI-Kit API",
    description="API for the HackAI-Kit, a comprehensive toolkit for AI/ML in hackathons",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. In production, specify origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers as app state
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing API...")
    app.state.config_manager = ConfigManager()
    app.state.module_manager = ModuleManager(config_manager=app.state.config_manager)
    logger.info(f"Discovered modules: {app.state.module_manager.discover_modules()}")

# Dependency to get the module manager
def get_module_manager():
    return app.state.module_manager

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to HackAI-Kit API"}

# Status endpoint
@app.get("/api/v1/status")
async def status():
    return {
        "status": "ok",
        "version": app.version
    }

# List available modules
@app.get("/api/v1/modules")
async def list_modules(module_manager: ModuleManager = Depends(get_module_manager)):
    modules = module_manager.get_module_info()
    return {
        "modules": modules,
        "count": len(modules)
    }

# Import and include module-specific routers
from hackaikit.api.endpoints import (
    supervised_router, 
    unsupervised_router,
    nlp_router,
    cv_router,
    timeseries_router,
    rl_router,
    deeplearning_router,
    llm_router,
    sensor_router,
    tinyml_router
)

# Include routers
app.include_router(supervised_router, prefix="/api/v1/supervised", tags=["Supervised Learning"])
app.include_router(unsupervised_router, prefix="/api/v1/unsupervised", tags=["Unsupervised Learning"])
app.include_router(nlp_router, prefix="/api/v1/nlp", tags=["NLP"])
app.include_router(cv_router, prefix="/api/v1/cv", tags=["Computer Vision"])
app.include_router(timeseries_router, prefix="/api/v1/timeseries", tags=["Time Series"])
app.include_router(rl_router, prefix="/api/v1/rl", tags=["Reinforcement Learning"])
app.include_router(deeplearning_router, prefix="/api/v1/deeplearning", tags=["Deep Learning"])
app.include_router(llm_router, prefix="/api/v1/llm", tags=["LLM Tools"])
app.include_router(sensor_router, prefix="/api/v1/sensor", tags=["Sensor & Signal"])
app.include_router(tinyml_router, prefix="/api/v1/tinyml", tags=["TinyML"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hackaikit.api.main:app", host="0.0.0.0", port=8000, reload=True) 