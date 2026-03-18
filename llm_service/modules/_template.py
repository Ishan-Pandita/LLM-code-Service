"""
Module Template - Copy this file to create a new module.

Instructions:
1. Copy this file to llm_service/modules/your_module_name.py
2. Rename the classes (YourModuleInput, YourModuleOutput, YourModule)
3. Update module_id to a unique identifier
4. Define your input/output schemas
5. Implement build_system_prompt(), build_user_prompt(), parse_output()
6. Register: get_module_registry().register(YourModule)
7. Add import to llm_service/modules/__init__.py
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from llm_service.core.inference_engine import GenerationConfig, InferenceEngine
from .base import BaseModule, ModuleInput, ModuleOutput, get_module_registry

logger = logging.getLogger(__name__)



# STEP 1: Define your input schema

class YourModuleInput(ModuleInput):
    """
    Input schema for your module.
    
    Define all required and optional fields here.
    Use Pydantic Field() for validation and documentation.
    """
    
    # Required field example
    input_text: str = Field(
        ...,  # ... means required
        min_length=1,
        max_length=10000,
        description="The main input text to process"
    )
    
    # Optional field with default
    option1: str = Field(
        default="default_value",
        description="An optional configuration option"
    )
    
    # Optional field without default
    option2: Optional[List[str]] = Field(
        default=None,
        description="An optional list of items"
    )
    
    # Literal for enum-like validation
    mode: Literal["fast", "accurate", "balanced"] = Field(
        default="balanced",
        description="Processing mode"
    )



# STEP 2: Define your output schema

class YourModuleOutput(ModuleOutput):
    """
    Output schema for your module.
    
    This MUST match what the LLM is instructed to output in the system prompt.
    """
    
    result: str = Field(
        ...,
        description="The main result"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    details: List[str] = Field(
        default_factory=list,
        description="Additional details"
    )



# STEP 3: Implement the module

class YourModule(BaseModule):
    """
    Your module implementation.
    
    Change 'YourModule' to a descriptive name like:
    - CodeExplanationModule
    - CodeRefactoringModule
    - DocumentationGeneratorModule
    """
    
    # IMPORTANT: Unique identifier for API access
    module_id = "your_module"  # API will use: {"module": "your_module", ...}
    
    # Link the schemas
    input_schema = YourModuleInput
    output_schema = YourModuleOutput
    
    def __init__(self, inference_engine: InferenceEngine) -> None:
        super().__init__(inference_engine)
    
    def build_system_prompt(self) -> str:
        """
        Build the system prompt.
        
        This tells the LLM:
        - What role to assume
        - What constraints to follow
        - EXACTLY what JSON schema to output
        
        IMPORTANT: The JSON schema here MUST match YourModuleOutput!
        """
        return """You are an expert assistant. Your role is to [DESCRIBE YOUR MODULE'S PURPOSE].

STRICT REQUIREMENTS:
1. You MUST respond with ONLY a valid JSON object
2. No markdown, no explanations, no additional text
3. Your response must exactly match the required JSON schema

REQUIRED JSON SCHEMA:
{
    "result": "<main result string>",
    "confidence": <float between 0 and 1>,
    "details": ["<detail string 1>", "<detail string 2>"]
}

[ADD ANY ADDITIONAL INSTRUCTIONS FOR THE LLM HERE]"""

    def build_user_prompt(self, payload: Dict[str, Any]) -> str:
        """
        Build the user prompt from the validated payload.
        
        The payload dict contains all fields from YourModuleInput.
        """
        prompt_parts = [
            f"Process the following input in {payload['mode']} mode:",
            "",
            payload['input_text'],
        ]
        
        if payload.get('option1') and payload['option1'] != "default_value":
            prompt_parts.append(f"\nOption: {payload['option1']}")
        
        if payload.get('option2'):
            prompt_parts.append(f"\nItems: {', '.join(payload['option2'])}")
        
        prompt_parts.append("\nProvide your response as JSON following the schema.")
        
        return "\n".join(prompt_parts)
    
    def get_generation_config(self) -> GenerationConfig:
        """
        Optional: Customize generation parameters for this module.
        
        Lower temperature = more consistent outputs (good for JSON)
        Higher temperature = more creative outputs
        """
        return GenerationConfig(
            max_new_tokens=1024,
            temperature=0.3,  # Low for consistent JSON
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            do_sample=True,
        )
    
    def parse_output(self, raw_output: str) -> YourModuleOutput:
        """
        Parse and validate the LLM's output.
        
        This method:
        1. Extracts JSON from the raw text
        2. Parses the JSON
        3. Validates against YourModuleOutput schema
        """
        json_str = self._extract_json(raw_output)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Invalid JSON in model output: {e}")
        
        try:
            return YourModuleOutput.model_validate(data)
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            raise ValueError(f"Output does not match schema: {e}")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()
        
        # Try markdown code block
        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        match = re.search(json_pattern, text)
        if match:
            return match.group(1)
        
        # Try standalone JSON
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return text[brace_start:brace_end + 1]
        
        return text



# STEP 4: Register the module (uncomment when ready to use)

# get_module_registry().register(YourModule)


# STEP 5: Add to __init__.py (add this line):

# from .your_module import YourModule
