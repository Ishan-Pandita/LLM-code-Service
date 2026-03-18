"""
Base Module - Abstract base class and registry for all modules.

Defines the interface that all module implementations must follow and provides
the module registry for dynamic module resolution.

HOW TO ADD A NEW MODULE:
========================

1. Create a new file (e.g., llm_service/modules/my_module.py)

2. Define input/output schemas:
   ```python
   class MyModuleInput(ModuleInput):
       field1: str = Field(..., description="...")
       field2: int = Field(default=10)
   
   class MyModuleOutput(ModuleOutput):
       result: str
       score: float
   ```

3. Create the module class:
   ```python
   class MyModule(BaseModule):
       module_id = "my_module"          # Unique identifier
       input_schema = MyModuleInput
       output_schema = MyModuleOutput
       
       def build_system_prompt(self) -> str:
           return "You are an expert..."
       
       def build_user_prompt(self, payload: dict) -> str:
           return f"Process this: {payload['field1']}"
       
       def parse_output(self, raw_output: str) -> MyModuleOutput:
           # Parse JSON and return validated output
   ```

4. Register the module:
   ```python
   get_module_registry().register(MyModule)
   ```

5. Import in __init__.py:
   ```python
   from .my_module import MyModule
   ```

6. Done! Your module is now available via the API.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from llm_service.core.inference_engine import GenerationConfig, InferenceEngine, InferenceResult

logger = logging.getLogger(__name__)


class ModuleInput(BaseModel):
    """Base class for module input validation."""
    pass


class ModuleOutput(BaseModel):
    """Base class for module output validation."""
    pass


class BaseModule(ABC):
    """
    Abstract base class for all module implementations.
    
    Each module must implement:
    - module_id: Unique identifier for the module
    - input_schema: Pydantic model for validating inputs
    - output_schema: Pydantic model for validating outputs
    - build_system_prompt(): Returns the system message
    - build_user_prompt(payload): Returns the user message
    - parse_output(raw_output): Parses and validates the output
    """
    
    # Class-level attributes to be overridden
    module_id: str = ""
    input_schema: Type[ModuleInput] = ModuleInput
    output_schema: Type[ModuleOutput] = ModuleOutput
    
    def __init__(self, inference_engine: InferenceEngine) -> None:
        """
        Initialize the module.
        
        Args:
            inference_engine: Shared inference engine instance
        """
        self.inference_engine = inference_engine
    
    @abstractmethod
    def build_system_prompt(self) -> str:
        """
        Build the system prompt for this module.
        
        The system prompt defines the role and strict constraints for the model.
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def build_user_prompt(self, payload: Dict[str, Any]) -> str:
        """
        Build the user prompt from the payload.
        
        The user prompt contains the module-specific input.
        
        Args:
            payload: Module-specific input data
            
        Returns:
            User prompt string
        """
        pass
    
    @abstractmethod
    def parse_output(self, raw_output: str) -> ModuleOutput:
        """
        Parse and validate the raw model output.
        
        Args:
            raw_output: Raw text from the model
            
        Returns:
            Validated output conforming to output_schema
            
        Raises:
            ValueError: If output doesn't conform to schema
        """
        pass
    
    def get_generation_config(self) -> Optional[GenerationConfig]:
        """
        Get module-specific generation configuration.
        
        Override this to customize generation parameters for the module.
        
        Returns:
            GenerationConfig or None to use defaults
        """
        return None
    
    def validate_input(self, payload: Dict[str, Any]) -> ModuleInput:
        """
        Validate the input payload against the input schema.
        
        Args:
            payload: Raw input data
            
        Returns:
            Validated input model
            
        Raises:
            ValidationError: If input doesn't conform to schema
        """
        return self.input_schema.model_validate(payload)
    
    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the module with the given payload.
        
        This is the main entry point for module execution:
        1. Validates input
        2. Builds prompts
        3. Calls inference engine
        4. Parses and validates output
        
        Args:
            payload: Module-specific input data
            
        Returns:
            Dictionary with module output and metadata
            
        Raises:
            ValidationError: If input/output validation fails
            RuntimeError: If inference fails
        """
        logger.info(f"Executing module: {self.module_id}")
        
        # Validate input
        validated_input = self.validate_input(payload)
        logger.debug(f"Input validated: {validated_input}")
        
        # Build prompts
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(validated_input.model_dump())
        
        # Get generation config
        config = self.get_generation_config()
        
        # Execute inference
        result: InferenceResult = await self.inference_engine.generate_from_prompt(
            system_message=system_prompt,
            user_message=user_prompt,
            config=config
        )
        
        logger.debug(f"Raw output: {result.generated_text[:200]}...")
        
        # Parse and validate output
        parsed_output = self.parse_output(result.generated_text)
        
        return {
            "module_id": self.module_id,
            "output": parsed_output.model_dump(),
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
            }
        }


class ModuleRegistry:
    """
    Registry for module implementations.
    
    Provides dynamic module resolution by module_id and manages
    module instantiation with shared dependencies.
    """
    
    _instance: Optional["ModuleRegistry"] = None
    
    def __new__(cls) -> "ModuleRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modules: Dict[str, Type[BaseModule]] = {}
            cls._instance._instances: Dict[str, BaseModule] = {}
        return cls._instance
    
    def register(self, module_class: Type[BaseModule]) -> Type[BaseModule]:
        """
        Register a module class.
        
        Can be used as a decorator:
        
        @registry.register
        class MyModule(BaseModule):
            ...
            
        Args:
            module_class: Module class to register
            
        Returns:
            The module class (for decorator usage)
            
        Raises:
            ValueError: If module_id is not defined or already registered
        """
        if not module_class.module_id:
            raise ValueError(f"Module class {module_class.__name__} must define module_id")
        
        if module_class.module_id in self._modules:
            raise ValueError(f"Module '{module_class.module_id}' is already registered")
        
        self._modules[module_class.module_id] = module_class
        logger.info(f"Registered module: {module_class.module_id}")
        
        return module_class
    
    def get_module(
        self,
        module_id: str,
        inference_engine: InferenceEngine
    ) -> BaseModule:
        """
        Get a module instance by ID.
        
        Creates and caches module instances.
        
        Args:
            module_id: Module identifier
            inference_engine: Shared inference engine
            
        Returns:
            Module instance
            
        Raises:
            KeyError: If module_id is not registered
        """
        if module_id not in self._modules:
            available = list(self._modules.keys())
            raise KeyError(
                f"Unknown module: '{module_id}'. Available modules: {available}"
            )
        
        # Create instance if not cached
        if module_id not in self._instances:
            module_class = self._modules[module_id]
            self._instances[module_id] = module_class(inference_engine)
        
        return self._instances[module_id]
    
    def list_modules(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered modules with their schemas.
        
        Returns:
            Dictionary of module info keyed by module_id
        """
        return {
            module_id: {
                "module_id": module_id,
                "class": module_class.__name__,
                "input_schema": module_class.input_schema.model_json_schema(),
                "output_schema": module_class.output_schema.model_json_schema(),
            }
            for module_id, module_class in self._modules.items()
        }
    
    def clear(self) -> None:
        """Clear all registered modules and instances."""
        self._modules.clear()
        self._instances.clear()


# Global registry instance
_registry: Optional[ModuleRegistry] = None


def get_module_registry() -> ModuleRegistry:
    """Get the global module registry instance."""
    global _registry
    if _registry is None:
        _registry = ModuleRegistry()
    return _registry
