"""
Function Calling Utilities

This module provides utilities for working with LLM function calling capabilities.
Works with any model that supports tool/function calling (e.g., GLM-4, Qwen, Llama, etc.).

Standard function calling format:
- Tool definitions are embedded in the system prompt
- Function calls use: {"role": "assistant", "metadata": function_name, "content": args_json}
- Tool results use: {"role": "observation", "content": result}
"""

import json
import re
import ast
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """A single parameter for a tool function."""
    name: str
    description: str
    type: str = "string"
    required: bool = True


@dataclass 
class ToolDefinition:
    """Definition of a tool/function that can be called by the model."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "description": param.description,
                "type": param.type
            }
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


@dataclass
class FunctionCall:
    """Represents a function call made by the model."""
    name: str
    arguments: Dict[str, Any]
    
    def to_message(self) -> Dict[str, Any]:
        """Convert to chat message format."""
        return {
            "role": "assistant",
            "metadata": self.name,
            "content": json.dumps(self.arguments, ensure_ascii=False)
        }


@dataclass
class FunctionResult:
    """Result from executing a function."""
    content: Any
    
    def to_message(self) -> Dict[str, Any]:
        """Convert to observation message format."""
        if isinstance(self.content, str):
            content_str = self.content
        else:
            content_str = json.dumps(self.content, ensure_ascii=False)
        
        return {
            "role": "observation",
            "content": content_str
        }


def build_tools_system_prompt(tools: List[ToolDefinition], base_prompt: str = "") -> str:
    """
    Build a system prompt with embedded tool definitions.
    
    Args:
        tools: List of tool definitions
        base_prompt: Base system prompt to prepend
        
    Returns:
        System prompt with tool definitions
    """
    if not tools:
        return base_prompt
    
    tool_section = "# Available Tools\n"
    
    for tool in tools:
        tool_dict = tool.to_dict()
        func = tool_dict["function"]
        tool_section += f"\n## {func['name']}\n\n"
        tool_section += json.dumps(func, ensure_ascii=False, indent=4)
        tool_section += "\n\nWhen calling this function, use JSON format for the arguments.\n"
    
    if base_prompt:
        return f"{base_prompt}\n\n{tool_section}"
    return tool_section


def parse_function_call(response_text: str) -> Optional[FunctionCall]:
    """
    Parse a function call from the model's response.
    
    Models typically output function calls in format:
    function_name
    {"arg1": "value1", "arg2": "value2"}
    
    Args:
        response_text: Raw response text from the model
        
    Returns:
        FunctionCall if found, None otherwise
    """
    response_text = response_text.strip()
    
    # Pattern to match: function_name followed by JSON
    pattern = re.compile(r'^([^\n`\{\}]+?)\n(\{.*?\})(?=\s*\n|$)', re.DOTALL)
    matches = pattern.findall(response_text)
    
    if not matches:
        return None
    
    func_name, args_str = matches[0]
    func_name = func_name.strip()
    
    # Try to parse the arguments as JSON
    try:
        parsed_args = json.loads(args_str)
    except json.JSONDecodeError:
        # Try ast.literal_eval as fallback
        try:
            parsed_args = ast.literal_eval(args_str)
        except:
            logger.warning(f"Failed to parse function arguments: {args_str}")
            return None
    
    return FunctionCall(name=func_name, arguments=parsed_args)


def parse_multiple_function_calls(response_text: str) -> List[FunctionCall]:
    """
    Parse multiple function calls from a response.
    
    Some models can call multiple functions in a single response.
    
    Args:
        response_text: Raw response text
        
    Returns:
        List of FunctionCall objects
    """
    calls = []
    
    # Split by assistant markers if present
    parts = response_text.split("<|assistant|>")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        fc = parse_function_call(part)
        if fc:
            calls.append(fc)
    
    # If no splits worked, try the whole text
    if not calls:
        fc = parse_function_call(response_text)
        if fc:
            calls.append(fc)
    
    return calls


def is_function_call_response(response_text: str) -> bool:
    """
    Check if the response contains a function call.
    
    Args:
        response_text: Model response text
        
    Returns:
        True if response appears to be a function call
    """
    return parse_function_call(response_text) is not None


class FunctionRegistry:
    """
    Registry for callable functions that can be invoked by the model.
    
    Allows registering Python functions that the model can call,
    and handles the execution of those functions.
    """
    
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}
    
    def register(
        self, 
        func: Callable, 
        definition: ToolDefinition
    ) -> None:
        """
        Register a function with its definition.
        
        Args:
            func: The callable function
            definition: Tool definition describing the function
        """
        self._functions[definition.name] = func
        self._definitions[definition.name] = definition
    
    def get_definitions(self) -> List[ToolDefinition]:
        """Get all registered tool definitions."""
        return list(self._definitions.values())
    
    def execute(self, call: FunctionCall) -> FunctionResult:
        """
        Execute a function call.
        
        Args:
            call: The function call to execute
            
        Returns:
            FunctionResult with the execution result
            
        Raises:
            KeyError: If function is not registered
        """
        if call.name not in self._functions:
            return FunctionResult(
                content={"error": f"Function '{call.name}' not found"}
            )
        
        try:
            func = self._functions[call.name]
            result = func(**call.arguments)
            return FunctionResult(content=result)
        except Exception as e:
            logger.exception(f"Error executing function {call.name}")
            return FunctionResult(
                content={"error": str(e)}
            )
    
    def has_function(self, name: str) -> bool:
        """Check if a function is registered."""
        return name in self._functions


# Pre-defined tools for code evaluation
EVALUATE_CODE_TOOL = ToolDefinition(
    name="evaluate_code",
    description="Evaluate student code against a specific coding rule and return the evaluation result",
    parameters=[
        ToolParameter(
            name="rule_id",
            description="The ID of the rule being evaluated"
        ),
        ToolParameter(
            name="status",
            description="PASS or FAIL - whether the code follows the rule",
            type="string"
        ),
        ToolParameter(
            name="confidence",
            description="HIGH, MEDIUM, or LOW - confidence in the evaluation",
            type="string"
        ),
        ToolParameter(
            name="evidence",
            description="Specific code reference or quote supporting the evaluation",
            type="string"
        ),
        ToolParameter(
            name="suggestion",
            description="Improvement suggestion if FAIL, null if PASS",
            type="string",
            required=False
        )
    ]
)

FIX_CODE_TOOL = ToolDefinition(
    name="fix_code",
    description="Fix a specific error in the code and return the fixed version",
    parameters=[
        ToolParameter(
            name="fixed_code",
            description="The complete fixed code"
        ),
        ToolParameter(
            name="original_snippet",
            description="The problematic code snippet (1-5 lines)"
        ),
        ToolParameter(
            name="fixed_snippet",
            description="The corrected code snippet (1-5 lines)"
        ),
        ToolParameter(
            name="explanation",
            description="Explanation of what was wrong and how it was fixed"
        ),
        ToolParameter(
            name="severity",
            description="Error severity from 1-10",
            type="integer"
        )
    ]
)
