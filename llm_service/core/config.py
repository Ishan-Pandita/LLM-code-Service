"""
Configuration module for LLM Service with vLLM backend.

Centralizes all configuration using Pydantic Settings for environment variable support
and validation. Optimized for vLLM production deployment.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables with the same name
    (case-insensitive).
    """
    
    # Model Configuration
    model_id: str = Field(
        default="zai-org/GLM-4.7-Flash",
        description="Hugging Face model identifier"
    )
    model_revision: str = Field(
        default="main",
        description="Model revision/branch to use"
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code in the model"
    )
    
    # vLLM Engine Configuration
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism (set to GPU count for multi-GPU)"
    )
    gpu_memory_utilization: float = Field(
        default=0.90,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use (0.90 = 90%)"
    )
    max_model_len: int = Field(
        default=32768,
        ge=128,
        description="Maximum sequence length (context window)"
    )
    vllm_dtype: str = Field(
        default="auto",
        description="Data type for vLLM: 'auto', 'float16', 'bfloat16', 'float32'"
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization method: 'awq', 'gptq', 'squeezellm', 'fp8', or None"
    )
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graphs (useful for debugging or compatibility)"
    )

    @field_validator("quantization", mode="before")
    @classmethod
    def _empty_quantization_to_none(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable prefix caching for faster repeated prompts (recommended)"
    )
    
    # Generation Defaults
    default_max_new_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Default maximum new tokens to generate"
    )
    default_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature (0.0 for deterministic output)"
    )
    default_top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Default nucleus sampling probability"
    )
    default_top_k: int = Field(
        default=-1,
        ge=-1,
        description="Default top-k sampling parameter (-1 to disable)"
    )
    default_repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Default repetition penalty"
    )
    
    # Caching Configuration
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching model weights"
    )
    
    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port"
    )
    api_workers: int = Field(
        default=1,
        ge=1,
        description="Number of API workers (keep at 1 for shared vLLM engine)"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Request Limits
    request_timeout: float = Field(
        default=300.0,
        ge=1.0,
        description="Request timeout in seconds"
    )
    
    # External Services
    compiler_base_url: str = Field(
        default="http://localhost:32358",
        description="Base URL for the compiler service"
    )
    
    class Config:
        env_prefix = "LLM_SERVICE_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
