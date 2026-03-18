"""
Core module - Contains model loading, inference engine, and shared utilities.
"""

from .model_loader import ModelLoader, get_model_loader
from .inference_engine import InferenceEngine, get_inference_engine, ToolCallResult
from .config import Settings, get_settings
from .function_calling import (
    ToolDefinition,
    ToolParameter,
    FunctionCall,
    FunctionResult,
    FunctionRegistry,
    build_tools_system_prompt,
    parse_function_call,
    parse_multiple_function_calls,
    is_function_call_response,
)

__all__ = [
    # Model loading
    "ModelLoader",
    "get_model_loader",
    # Inference
    "InferenceEngine", 
    "get_inference_engine",
    "ToolCallResult",
    # Config
    "Settings",
    "get_settings",
    # Function Calling
    "ToolDefinition",
    "ToolParameter",
    "FunctionCall",
    "FunctionResult",
    "FunctionRegistry",
    "build_tools_system_prompt",
    "parse_function_call",
    "parse_multiple_function_calls",
    "is_function_call_response",
]
