"""
Evaluation Service Module - Iteratively fixes code errors using LLM and compiler.

This module takes code with errors or incomplete code, and:
1. Submits to compiler to detect errors
2. Uses LLM to fix ONE error at a time
3. Verifies fix with compiler
4. Repeats until no errors or max iterations

Endpoints:
- GET  /api/v1/modules/evaluation_service/languages  - Get supported languages
- GET  /api/v1/modules/evaluation_service/health     - Check compiler status
- POST /api/v1/modules/evaluation_service/evaluate   - Evaluate and fix code
- POST /api/v1/modules/evaluation_service/compile    - Just compile (no fix)
"""

from .module import EvaluationServiceModule
from .schemas import (
    EvaluationRequest,
    EvaluationResponse,
    ErrorInfo,
    FixInfo,
    LanguageInfo,
)
from .routes import router

__all__ = [
    "EvaluationServiceModule",
    "EvaluationRequest",
    "EvaluationResponse",
    "ErrorInfo",
    "FixInfo",
    "LanguageInfo",
    "router",
]
