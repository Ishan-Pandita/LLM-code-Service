"""
Evaluation Service Module - Custom API routes.

Provides module-specific endpoints:
- GET  /api/v1/modules/evaluation_service/languages  - Get supported languages
- GET  /api/v1/modules/evaluation_service/health     - Check compiler status
- POST /api/v1/modules/evaluation_service/evaluate   - Evaluate and fix code
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from .schemas import EvaluationRequest, EvaluationResponse, LanguageInfo
from .helpers import CompilerClient
from .module import EvaluationServiceModule

logger = logging.getLogger(__name__)

# Create router for this module
router = APIRouter(
    prefix="/evaluation_service",
    tags=["Evaluation Service"]
)


@router.get(
    "/languages",
    summary="Get Supported Languages",
    description="Get list of programming languages supported by the compiler"
)
async def get_languages():
    """
    Get all supported programming languages from the compiler service.
    
    Returns:
        List of language objects with id and name
    """
    try:
        compiler = CompilerClient()
        languages = await compiler.get_languages()
        
        if not languages:
            raise HTTPException(
                status_code=503,
                detail="Compiler service unavailable or returned no languages"
            )
        
        return {
            "languages": languages,
            "total": len(languages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate and Fix Code",
    description="Submit code with errors, get it fixed iteratively using LLM and verified by compiler"
)
async def evaluate_code(
    request: Request,
    evaluation_request: EvaluationRequest
) -> EvaluationResponse:
    """
    Evaluate code and iteratively fix any errors.
    
    Process:
    1. Submit code to compiler
    2. If error found, use LLM to fix ONE error
    3. Verify fix with compiler
    4. Repeat until no errors or max iterations reached
    
    Args:
        evaluation_request: The evaluation request containing:
            - code: The code to evaluate and fix
            - language_id: Language ID from /languages endpoint
            - language_name: Optional human-readable language name
            - max_iterations: Maximum fix attempts (default: 10)
            - stdin: Optional input for the program
            - expected_output: Optional expected output
            
    Returns:
        EvaluationResponse with:
            - success: Whether code was successfully fixed
            - total_errors_fixed: Number of errors fixed
            - fixed_code: The corrected code
            - fixes: List of all fixes with explanations and severity
    """
    try:
        # Get inference engine from app state
        inference_engine = request.app.state.inference_engine
        
        # Create module instance
        module = EvaluationServiceModule(inference_engine)
        
        # Run evaluation
        result = await module.evaluate_and_fix(
            code=evaluation_request.code,
            language_id=evaluation_request.language_id,
            language_name=evaluation_request.language_name,
            max_iterations=evaluation_request.max_iterations,
            stdin=evaluation_request.stdin,
            expected_output=evaluation_request.expected_output,
            expected_code=evaluation_request.expected_code,
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    summary="Check Compiler Health",
    description="Check if the compiler service is available"
)
async def check_compiler_health():
    """Check if the compiler service is healthy."""
    try:
        compiler = CompilerClient()
        languages = await compiler.get_languages()
        
        return {
            "status": "healthy" if languages else "unavailable",
            "compiler_url": compiler.base_url,
            "languages_available": len(languages) if languages else 0
        }
        
    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e)
        }
