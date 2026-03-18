"""
Modules package - Task-specific implementations including prompt builders and validators.

To add a new module:
1. Create a new folder in this directory (e.g., my_module/)
2. Add __init__.py, module.py, schemas.py, helpers.py
3. Optionally add routes.py for custom endpoints
4. Register the module in module.py
5. Import the module in this file

See best_practices/ and evaluation_service/ for complete examples.
"""

from .base import BaseModule, ModuleRegistry, get_module_registry
from .best_practices import BestPracticesModule
from .evaluation_service import EvaluationServiceModule

__all__ = [
    "BaseModule",
    "ModuleRegistry",
    "get_module_registry",
    "BestPracticesModule",
    "EvaluationServiceModule",
]
