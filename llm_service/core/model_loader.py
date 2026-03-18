"""
Model Loader Module - Centralized model loading with vLLM (Async).

This module ensures the vLLM engine is initialized exactly once and shared
across all tasks. Uses Async vLLM for true non-blocking inference with:
- Continuous batching for high throughput
- PagedAttention for efficient memory usage
- Tensor parallelism for multi-GPU
- Prefix caching for faster repeated prompts
"""

import logging
import threading
import re
from typing import Optional, Tuple

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Thread-safe singleton model loader for Async vLLM.
    
    Ensures the vLLM engine and tokenizer are loaded exactly once and provides
    thread-safe access to both. vLLM handles all quantization and memory 
    management internally.
    
    Attributes:
        llm: The vLLM AsyncLLMEngine instance
        tokenizer: The loaded tokenizer
    """
    
    _instance: Optional["ModelLoader"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls, settings: Optional[Settings] = None) -> "ModelLoader":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize the model loader.
        
        Args:
            settings: Application settings. If None, uses default settings.
        """
        # Skip if already initialized
        if ModelLoader._initialized:
            return
            
        with ModelLoader._lock:
            if ModelLoader._initialized:
                return
                
            self.settings = settings or get_settings()
            self._llm: Optional[AsyncLLMEngine] = None
            self._tokenizer: Optional[PreTrainedTokenizer] = None
            self._model_loaded: bool = False
            self._load_lock: threading.Lock = threading.Lock()
            
            ModelLoader._initialized = True
    
    def load_model(self) -> Tuple[AsyncLLMEngine, PreTrainedTokenizer]:
        """
        Load the Async vLLM engine and tokenizer.
        
        Thread-safe method that ensures the model is loaded exactly once.
        Subsequent calls return the cached engine and tokenizer.
        
        Returns:
            Tuple[AsyncLLMEngine, PreTrainedTokenizer]: vLLM engine and tokenizer
            
        Raises:
            RuntimeError: If model loading fails
        """
        if self._model_loaded:
            return self._llm, self._tokenizer
        
        with self._load_lock:
            # Double-checked locking
            if self._model_loaded:
                return self._llm, self._tokenizer
            
            logger.info(f"Loading model with Async vLLM: {self.settings.model_id}")
            logger.info(f"Tensor parallel size: {self.settings.tensor_parallel_size}")
            logger.info(f"GPU memory utilization: {self.settings.gpu_memory_utilization}")
            logger.info(f"Max model length: {self.settings.max_model_len}")
            
            try:
                # Load tokenizer first (for chat template support)
                logger.info("Loading tokenizer...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_id,
                    revision=self.settings.model_revision,
                    trust_remote_code=self.settings.trust_remote_code,
                    cache_dir=self.settings.cache_dir,
                )
                
                # Ensure pad token is set
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    logger.info("Set pad_token to eos_token")
                
                # Build vLLM engine args
                print(f"DEBUG: Model ID: {self.settings.model_id}")
                print(f"DEBUG: Quantization: {self.settings.quantization}")
                print(f"DEBUG: Dtype: {self.settings.vllm_dtype}")
                print(f"DEBUG: Prefix Caching: {self.settings.enable_prefix_caching}")
                
                logger.info(f"Model ID: {self.settings.model_id}")
                logger.info(f"Quantization: {self.settings.quantization}")
                logger.info(f"Dtype: {self.settings.vllm_dtype}")
                logger.info(f"Prefix Caching: {self.settings.enable_prefix_caching}")
                
                engine_args = self._build_engine_args(self.settings.gpu_memory_utilization)
                
                # Load vLLM engine
                logger.info("Loading Async vLLM engine (this may take a few minutes)...")
                try:
                    self._llm = AsyncLLMEngine.from_engine_args(engine_args)
                except Exception as first_error:
                    # Common on constrained GPUs: free memory is below requested utilization.
                    fallback_util = self._suggest_fallback_gpu_utilization(str(first_error))
                    if fallback_util is None:
                        raise
                    
                    logger.warning(
                        "Initial vLLM startup failed due to GPU memory headroom. "
                        f"Retrying with gpu_memory_utilization={fallback_util:.2f} "
                        f"(was {self.settings.gpu_memory_utilization:.2f})."
                    )
                    retry_args = self._build_engine_args(fallback_util)
                    self._llm = AsyncLLMEngine.from_engine_args(retry_args)
                
                logger.info(f"Async vLLM engine loaded successfully")
                
                self._model_loaded = True
                return self._llm, self._tokenizer
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Model loading failed: {e}") from e

    def _build_engine_args(self, gpu_memory_utilization: float) -> AsyncEngineArgs:
        """Create AsyncEngineArgs with a specific GPU memory utilization."""
        return AsyncEngineArgs(
            model=self.settings.model_id,
            revision=self.settings.model_revision,
            trust_remote_code=self.settings.trust_remote_code,
            tensor_parallel_size=self.settings.tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self.settings.max_model_len,
            dtype=self.settings.vllm_dtype,
            download_dir=self.settings.cache_dir,
            enforce_eager=self.settings.enforce_eager,
            enable_prefix_caching=self.settings.enable_prefix_caching
            if hasattr(self.settings, 'enable_prefix_caching')
            else False,
            quantization=self.settings.quantization,
        )

    def _suggest_fallback_gpu_utilization(self, error_text: str) -> Optional[float]:
        """
        Suggest a safer GPU memory utilization for vLLM startup retries.

        Returns None if the error does not match the expected memory-headroom pattern.
        """
        # vLLM error format:
        # "Free memory on device cuda:0 (3.2/4.0 GiB) ... utilization (0.85, 3.4 GiB)."
        if "Free memory on device" not in error_text or "desired GPU memory utilization" not in error_text:
            return None

        match = re.search(r"\(([\d.]+)/([\d.]+) GiB\)", error_text)
        if not match:
            return None

        free_gb = float(match.group(1))
        total_gb = float(match.group(2))
        if total_gb <= 0:
            return None

        # Keep a little headroom below the observed free-memory ratio.
        free_ratio = free_gb / total_gb
        suggested = min(self.settings.gpu_memory_utilization - 0.05, free_ratio - 0.02)

        # Avoid unsafe/invalid values.
        if suggested < 0.5:
            suggested = 0.5
        if suggested >= self.settings.gpu_memory_utilization:
            return None

        return round(suggested, 2)
    
    @property
    def llm(self) -> AsyncLLMEngine:
        """Get the vLLM engine, loading if necessary."""
        if not self._model_loaded:
            self.load_model()
        return self._llm
    
    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the loaded tokenizer, loading if necessary."""
        if not self._model_loaded:
            self.load_model()
        return self._tokenizer
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded
    
    def unload(self) -> None:
        """
        Unload the model and free GPU memory.
        
        Primarily used for testing or graceful shutdown.
        """
        with self._load_lock:
            if self._llm is not None:
                # AsyncLLMEngine doesn't have a simple __del__ cleanup like LLM might,
                # but we can at least remove the reference.
                import gc
                del self._llm
                self._llm = None
            
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            
            self._model_loaded = False
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
            
            logger.info("Model unloaded and memory cleared")


# Module-level singleton accessor
_model_loader: Optional[ModelLoader] = None


def get_model_loader(settings: Optional[Settings] = None) -> ModelLoader:
    """
    Get the global model loader instance.
    
    Args:
        settings: Optional settings override
        
    Returns:
        ModelLoader: Singleton model loader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(settings)
    return _model_loader
