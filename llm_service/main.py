"""
LLM Service Entry Point - Main module for running the service (vLLM Backend).

Usage:
    python -m llm_service.main
    
Or with uvicorn:
    uvicorn llm_service.main:app --host 0.0.0.0 --port 8000
"""

import logging
import sys

import uvicorn

from llm_service.api import create_app
from llm_service.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# Set third-party loggers to WARNING
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Create app instance for uvicorn
app = create_app()


def main() -> None:
    """Run the LLM service."""
    settings = get_settings()
    
    logger.info("Starting LLM Service")
    logger.info(f"Host: {settings.api_host}")
    logger.info(f"Port: {settings.api_port}")
    logger.info(f"Workers: {settings.api_workers}")
    
    uvicorn.run(
        "llm_service.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,  # Keep at 1 to share model
        log_level=settings.log_level.lower(),
        reload=False,  # Disable reload in production
    )


if __name__ == "__main__":
    main()
