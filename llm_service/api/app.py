"""
FastAPI Application - Creates and configures the FastAPI application.

This module is responsible for:
- Application initialization
- Middleware configuration
- Startup/shutdown lifecycle management
- Model loading on startup
- Core API endpoints (health, list modules)
- Including module-specific routes (each module handles its own execution)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm_service.core.config import Settings, get_settings
from llm_service.core.model_loader import get_model_loader
from llm_service.core.inference_engine import get_inference_engine
from llm_service.modules import get_module_registry
from llm_service.modules.router import get_module_router

# Import module-specific routers
from llm_service.modules.best_practices import router as best_practices_router
from llm_service.modules.evaluation_service import router as evaluation_service_router

logger = logging.getLogger(__name__)

# Global app instance
_app: Optional[FastAPI] = None



# Response Models


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_id: str = Field(..., description="Loaded model identifier")


class ModuleInfo(BaseModel):
    """Information about a module."""
    module_id: str
    description: Optional[str] = None
    endpoints: list[str] = []


class ModuleListResponse(BaseModel):
    """Response for module listing."""
    modules: Dict[str, Any] = Field(..., description="Available modules with schemas")



# Lifespan Management


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Load the model
    - Shutdown: Clean up resources
    """
    settings = get_settings()
    
    # Startup
    logger.info("=" * 60)
    logger.info("LLM Service Starting Up")
    logger.info("=" * 60)
    logger.info(f"Model: {settings.model_id}")
    logger.info(f"Torch Dtype: {settings.vllm_dtype}")
    
    start_time = time.perf_counter()
    
    try:
        # Load model on startup (singleton pattern ensures single load)
        model_loader = get_model_loader(settings)
        model_loader.load_model()
        
        # Initialize inference engine
        inference_engine = get_inference_engine(model_loader, settings)
        
        # Initialize module router (for module registry access)
        module_router = get_module_router(inference_engine, get_module_registry())
        
        # Store in app state for access in endpoints
        app.state.model_loader = model_loader
        app.state.inference_engine = inference_engine
        app.state.module_router = module_router
        app.state.settings = settings
        
        load_time = time.perf_counter() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info("=" * 60)
        logger.info("LLM Service Ready")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("LLM Service Shutting Down")
        if hasattr(app.state, 'model_loader'):
            app.state.model_loader.unload()
        logger.info("Shutdown complete")



# Application Factory


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Optional settings override
        
    Returns:
        Configured FastAPI application
    """
    settings = settings or get_settings()
    
    app = FastAPI(
        title="LLM Service API",
        description="""
Production-ready LLM service for code evaluation and analysis.

## Features

- **Single Model Instance**: The model is loaded once and shared across all requests
- **Module-Based API**: Each module defines its own endpoints
- **Schema Validation**: All inputs and outputs are strictly validated

## Available Modules

### Best Practices (`best_practices`)
Evaluate code against coding best practice rules.

**Endpoints:**
- `GET /api/v1/modules/best_practices/rules` - Get predefined rules
- `POST /api/v1/modules/best_practices/evaluate` - Evaluate code against rules
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time-Ms"] = str(int(process_time * 1000))
        return response
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "error_type": "internal_error"
            }
        )
    
    
    # Core API Endpoints (health, list modules)
    
    
    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health Check",
        description="Check if the service is healthy and model is loaded"
    )
    async def health_check(request: Request) -> HealthResponse:
        """Health check endpoint."""
        model_loader = request.app.state.model_loader
        settings = request.app.state.settings
        
        return HealthResponse(
            status="healthy" if model_loader.is_loaded else "degraded",
            model_loaded=model_loader.is_loaded,
            model_id=settings.model_id
        )
    
    @app.get(
        "/api/v1/modules",
        response_model=ModuleListResponse,
        tags=["Modules"],
        summary="List Available Modules",
        description="Get a list of all available modules with their schemas"
    )
    async def list_modules(request: Request) -> ModuleListResponse:
        """List all available modules."""
        module_router = request.app.state.module_router
        modules = module_router.list_available_modules()
        return ModuleListResponse(modules=modules)
    
    
    # Include Module-Specific Routers
    # Each module handles its own /evaluate or execution endpoints
    
    
    app.include_router(best_practices_router, prefix="/api/v1/modules")
    app.include_router(evaluation_service_router, prefix="/api/v1/modules")
    
    return app


def get_app() -> FastAPI:
    """
    Get the global FastAPI application instance.
    
    Creates the app if it doesn't exist.
    
    Returns:
        FastAPI application
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app
