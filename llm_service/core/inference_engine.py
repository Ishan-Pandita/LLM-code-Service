"""
Inference Engine Module - Shared inference logic using Async vLLM.

Provides a unified interface for text generation using the vLLM engine.
Handles tokenization, generation with SamplingParams, and output processing
following production best practices.
"""

import logging
import threading
import uuid
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vllm import SamplingParams
from transformers import PreTrainedTokenizer

# vLLM async engine
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .config import Settings, get_settings
from .model_loader import ModelLoader, get_model_loader

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    
    Encapsulates all generation parameters with sensible defaults.
    These map to vLLM's SamplingParams.
    """
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    do_sample: bool = False
    num_return_sequences: int = 1
    stop_token_ids: Optional[List[int]] = None
    
    def to_sampling_params(self) -> SamplingParams:
        """Convert to vLLM SamplingParams."""
        # vLLM uses temperature=0 for greedy, otherwise sampling
        if not self.do_sample:
            temperature = 0.0
            top_p = 1.0
            top_k = -1  # Disable top_k for greedy
        else:
            temperature = self.temperature
            top_p = self.top_p
            top_k = self.top_k if self.top_k > 0 else -1
        
        return SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=self.repetition_penalty,
            n=self.num_return_sequences,
            stop_token_ids=self.stop_token_ids,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "top_k": self.top_k if self.do_sample else None,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "num_return_sequences": self.num_return_sequences,
        }


@dataclass
class InferenceResult:
    """
    Result of an inference operation.
    
    Attributes:
        generated_text: The generated text (prompt removed)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens processed
    """
    generated_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_text": self.generated_text,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            }
        }


class InferenceEngine:
    """
    Shared inference engine for all tasks using Async vLLM.
    
    Provides thread-safe text generation using the shared vLLM instance.
    vLLM handles batching and concurrent requests internally through
    continuous batching, so handling async requests efficiently is key.
    
    Pipeline:
    1. Build chat messages
    2. Apply chat template
    3. Generate with Async vLLM
    4. Return clean generated text
    """
    
    _instance: Optional["InferenceEngine"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs) -> "InferenceEngine":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize the inference engine.
        
        Args:
            model_loader: Model loader instance. If None, uses global instance.
            settings: Application settings. If None, uses default settings.
        """
        if InferenceEngine._initialized:
            return
            
        with InferenceEngine._lock:
            if InferenceEngine._initialized:
                return
            
            self.settings = settings or get_settings()
            self.model_loader = model_loader or get_model_loader()
            
            InferenceEngine._initialized = True
    
    @property
    def llm(self) -> AsyncLLMEngine:
        """Get the vLLM engine."""
        return self.model_loader.llm
    
    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the loaded tokenizer."""
        return self.model_loader.tokenizer
    
    def build_messages(
        self,
        system_message: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build chat messages in the standard format.
        
        Args:
            system_message: System instruction defining role and constraints
            user_message: User's task-specific input
            history: Optional conversation history
            
        Returns:
            List of message dictionaries in chat format
        """
        messages = [{"role": "system", "content": system_message}]
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs
    ) -> str:
        """
        Apply the tokenizer's chat template to messages.
        
        Args:
            messages: List of message dictionaries
            add_generation_prompt: Whether to add the generation prompt
            **kwargs: Additional arguments to pass to the template
            
        Returns:
            Formatted prompt string
        """
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs
        )
        return formatted
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        chat_template_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> InferenceResult:
        """
        Generate text from chat messages using Async vLLM.
        
        This is the main entry point for text generation. It:
        1. Applies the chat template
        2. Generates with vLLM's optimized inference asynchronously
        3. Returns clean generated text
        
        Args:
            messages: Chat messages in standard format
            config: Generation configuration
            chat_template_params: Optional parameters for chat template
            request_id: Optional custom request ID
            
        Returns:
            InferenceResult with generated text and token counts
            
        Raises:
            ValueError: If input exceeds maximum length
            RuntimeError: If generation fails
        """
        config = config or self._get_default_config()
        chat_template_params = chat_template_params or {}
        
        # Apply chat template
        prompt = self.apply_chat_template(messages, **chat_template_params)
        
        # Count input tokens for validation and reporting
        input_ids = self.tokenizer.encode(prompt)
        input_length = len(input_ids)
        
        # Validate input length
        if input_length >= self.settings.max_model_len:
            raise ValueError(
                f"Input length ({input_length}) exceeds maximum "
                f"({self.settings.max_model_len})"
            )
        
        logger.debug(f"Generating with input length: {input_length}")
        
        # Set stop tokens if not provided
        if config.stop_token_ids is None:
            config.stop_token_ids = [self.tokenizer.eos_token_id]
        
        # Convert to vLLM SamplingParams
        sampling_params = config.to_sampling_params()
        
        # Generate with Async vLLM
        request_id = request_id or str(uuid.uuid4())
        results_generator = self.llm.generate(prompt, sampling_params, request_id)
        
        final_output = None
        current_text = ""
        
        try:
            async for request_output in results_generator:
                final_output = request_output
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e
        
        # Extract result from first (and only) output
        if final_output:
            generated_text = final_output.outputs[0].text
            output_length = len(final_output.outputs[0].token_ids)
        else:
            generated_text = ""
            output_length = 0
            
        logger.debug(f"Generated {output_length} new tokens")
        
        return InferenceResult(
            generated_text=generated_text.strip(),
            input_tokens=input_length,
            output_tokens=output_length,
            total_tokens=input_length + output_length
        )
    
    async def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        chat_template_params: Optional[Dict[str, Any]] = None
    ) -> List[InferenceResult]:
        """
        Generate text for multiple message sets efficiently.
        
        Uses asyncio.gather to run multiple async generations concurrently.
        vLLM's internal continuous batching will optimize the execution.
        
        Args:
            messages_batch: List of message lists
            config: Generation configuration (shared across batch)
            chat_template_params: Optional parameters for chat template
            
        Returns:
            List of InferenceResult objects
        """
        
        logger.debug(f"Batch generating for {len(messages_batch)} prompts")
        
        tasks = [
            self.generate(messages, config, chat_template_params)
            for messages in messages_batch
        ]
        
        return await asyncio.gather(*tasks)
    
    async def generate_from_prompt(
        self,
        system_message: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
        history: Optional[List[Dict[str, str]]] = None,
        chat_template_params: Optional[Dict[str, Any]] = None
    ) -> InferenceResult:
        """
        Convenience method to generate from system and user messages.
        
        Args:
            system_message: System instruction
            user_message: User input
            config: Generation configuration
            history: Optional conversation history
            chat_template_params: Optional parameters for chat template
            
        Returns:
            InferenceResult with generated text
        """
        messages = self.build_messages(system_message, user_message, history)
        return await self.generate(messages, config, chat_template_params)
    
    def _get_default_config(self) -> GenerationConfig:
        """Get default generation config from settings."""
        return GenerationConfig(
            max_new_tokens=self.settings.default_max_new_tokens,
            temperature=self.settings.default_temperature,
            top_p=self.settings.default_top_p,
            top_k=self.settings.default_top_k,
            repetition_penalty=self.settings.default_repetition_penalty,
        )
    
    
    # Function Calling Support
    
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List["ToolDefinition"],
        config: Optional[GenerationConfig] = None,
        max_tool_calls: int = 5,
        chat_template_params: Optional[Dict[str, Any]] = None
    ) -> "ToolCallResult":
        """
        Generate with function calling support.
        
        Works with any model that supports tool calling. This method:
        1. Adds tool definitions to system prompt
        2. Generates response
        3. Parses any function calls
        4. Returns structured result
        
        Args:
            messages: Chat messages (system prompt will have tools added)
            tools: List of ToolDefinition objects
            config: Generation configuration
            max_tool_calls: Maximum number of tool calls allowed
            chat_template_params: Optional parameters for chat template
            
        Returns:
            ToolCallResult with parsed response and any function calls
        """
        from .function_calling import (
            ToolDefinition, 
            build_tools_system_prompt,
            parse_multiple_function_calls,
            FunctionCall
        )
        
        config = config or self._get_default_config()
        
        # Modify system message to include tool definitions
        enhanced_messages = []
        for msg in messages:
            if msg["role"] == "system":
                enhanced_content = build_tools_system_prompt(
                    tools, 
                    msg.get("content", "")
                )
                enhanced_messages.append({
                    "role": "system",
                    "content": enhanced_content
                })
            else:
                enhanced_messages.append(msg)
        
        # Generate response
        result = await self.generate(enhanced_messages, config, chat_template_params)
        
        # Parse any function calls from the response
        function_calls = parse_multiple_function_calls(result.generated_text)
        
        # Extract non-function-call text
        response_text = result.generated_text
        for fc in function_calls:
            # Remove function call from response text
            pattern = f"{fc.name}\n" + re.escape(json.dumps(fc.arguments))
            response_text = re.sub(pattern, "", response_text, count=1)
        response_text = response_text.strip()
        
        return ToolCallResult(
            text_response=response_text,
            function_calls=function_calls,
            raw_response=result.generated_text,
            usage={
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens
            }
        )
    
    def add_observation(
        self,
        messages: List[Dict[str, Any]],
        function_name: str,
        result: Any
    ) -> List[Dict[str, Any]]:
        """
        Add a function result (observation) to the conversation.
        
        Uses the "observation" role for function results.
        
        Args:
            messages: Current message list
            function_name: Name of the function that was called
            result: Result from the function execution
            
        Returns:
            Updated message list with observation added
        """
        messages = messages.copy()
        
        # First add the function call as assistant message
        if isinstance(result, str):
            content = result
        else:
            content = json.dumps(result, ensure_ascii=False)
        
        # Add observation
        messages.append({
            "role": "observation",
            "content": content
        })
        
        return messages


@dataclass
class ToolCallResult:
    """
    Result from a generation with tool calling.
    
    Attributes:
        text_response: Non-function-call text from the response
        function_calls: List of function calls made by the model
        raw_response: Original raw response text
        usage: Token usage statistics
    """
    text_response: str
    function_calls: List["FunctionCall"]
    raw_response: str
    usage: Dict[str, int]
    
    @property
    def has_function_calls(self) -> bool:
        """Check if the response contains function calls."""
        return len(self.function_calls) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_response": self.text_response,
            "function_calls": [
                {"name": fc.name, "arguments": fc.arguments}
                for fc in self.function_calls
            ],
            "raw_response": self.raw_response,
            "usage": self.usage
        }


# Import json and re for the new methods
import json
import re


# Module-level singleton accessor
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine(
    model_loader: Optional[ModelLoader] = None,
    settings: Optional[Settings] = None
) -> InferenceEngine:
    """
    Get the global inference engine instance.
    
    Args:
        model_loader: Optional model loader override
        settings: Optional settings override
        
    Returns:
        InferenceEngine: Singleton inference engine instance
    """
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine(model_loader, settings)
    return _inference_engine
