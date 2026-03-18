"""
Module Router - Provides module registry access for listing available modules.

Note: Each module handles its own execution via its routes.py file.
This router is only used for listing available modules.
"""

import logging
from typing import Any, Dict, Optional

from llm_service.core.inference_engine import InferenceEngine, get_inference_engine
from .base import ModuleRegistry, get_module_registry

logger = logging.getLogger(__name__)


class ModuleRouter:
    """
    Provides access to module registry for listing purposes.
    
    Each module handles its own execution via module-specific routes.
    """
    
    def __init__(
        self,
        inference_engine: Optional[InferenceEngine] = None,
        module_registry: Optional[ModuleRegistry] = None
    ) -> None:
        """
        Initialize the module router.
        
        Args:
            inference_engine: Shared inference engine. Uses global if None.
            module_registry: Module registry. Uses global if None.
        """
        self.inference_engine = inference_engine or get_inference_engine()
        self.module_registry = module_registry or get_module_registry()
    
    def list_available_modules(self) -> Dict[str, Any]:
        """
        List all available modules with their schemas.
        
        Returns:
            Dictionary of available modules
        """
        return self.module_registry.list_modules()


# Global router instance
_router: Optional[ModuleRouter] = None


def get_module_router(
    inference_engine: Optional[InferenceEngine] = None,
    module_registry: Optional[ModuleRegistry] = None
) -> ModuleRouter:
    """Get the global module router instance."""
    global _router
    if _router is None:
        _router = ModuleRouter(inference_engine, module_registry)
    return _router
