"""
Best Practices Module - Custom API routes.

Provides module-specific endpoints:
- GET  /api/v1/modules/best_practices/rules     - Get predefined rules
- POST /api/v1/modules/best_practices/evaluate  - Evaluate code against rules
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

from .schemas import Rule, RulesResponse, EvaluationRequest, EvaluationResponse
from .helpers import load_predefined_rules, get_rules_by_ids
from .module import BestPracticesModule

logger = logging.getLogger(__name__)

# Create router for this module
router = APIRouter(
    prefix="/best_practices",
    tags=["Best Practices"]
)


@router.get(
    "/rules",
    response_model=RulesResponse,
    summary="Get Predefined Rules",
    description="Retrieve all predefined coding best practice rules"
)
async def get_rules(
    category: Optional[str] = None
) -> RulesResponse:
    """
    Get all predefined rules with optional filtering.
    
    Args:
        category: Optional category filter (e.g., 'readability', 'structure')
        
    Returns:
        List of rules matching the filters
    """
    rules = load_predefined_rules()
    
    # Apply filters
    if category:
        rules = [r for r in rules if r.category == category]
    
    return RulesResponse(
        rules=rules,
        total=len(rules)
    )


@router.get(
    "/rules/{rule_id}",
    response_model=Rule,
    summary="Get Rule by ID",
    description="Retrieve a specific rule by its ID"
)
async def get_rule_by_id(rule_id: str) -> Rule:
    """
    Get a specific rule by ID.
    
    Args:
        rule_id: The rule identifier (e.g., 'R1')
        
    Returns:
        The requested rule
        
    Raises:
        404: If rule not found
    """
    rules = get_rules_by_ids([rule_id])
    
    if not rules:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")
    
    return rules[0]


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate Code",
    description="Evaluate code against coding best practice rules"
)
async def evaluate_code(
    request: Request,
    evaluation_request: EvaluationRequest
) -> EvaluationResponse:
    """
    Evaluate code against best practice rules.
    
    Args:
        evaluation_request: The evaluation request containing:
            - language: Programming language
            - code: The code to evaluate
            - predefined_rules: Optional list of predefined rule IDs to use
            - custom_rules: Optional list of custom rules to add
            
    Returns:
        Evaluation results with status for each rule
        
    Raises:
        400: If no rules specified
        500: If evaluation fails
    """
    try:
        # Get the module instance from app state
        module_router = request.app.state.module_router
        inference_engine = request.app.state.inference_engine
        
        # Get or create module instance
        module = BestPracticesModule(inference_engine)
        
        # Convert custom rules to dict format if provided
        custom_rules_dict = None
        if evaluation_request.custom_rules:
            custom_rules_dict = [r.model_dump() for r in evaluation_request.custom_rules]
        
        # Run evaluation
        result = await module.evaluate(
            language=evaluation_request.language,
            code=evaluation_request.code,
            predefined_rule_ids=evaluation_request.predefined_rules,
            custom_rules=custom_rules_dict
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/categories",
    summary="Get Rule Categories",
    description="Get all available rule categories"
)
async def get_categories() -> dict:
    """Get all unique rule categories."""
    rules = load_predefined_rules()
    categories = list(set(r.category for r in rules if r.category))
    
    return {
        "categories": sorted(categories),
        "total": len(categories)
    }


@router.get(
    "/health",
    summary="Check Module Health",
    description="Check if the best practices module is healthy"
)
async def check_health(request: Request) -> dict:
    """Check if the module is healthy and ready to process requests."""
    try:
        # Check if rules are loadable
        rules = load_predefined_rules()
        rules_ok = len(rules) > 0
        
        # Check if inference engine is available
        inference_ok = hasattr(request.app.state, 'inference_engine') and request.app.state.inference_engine is not None
        
        # Check if model is loaded
        model_loaded = hasattr(request.app.state, 'model_loader') and request.app.state.model_loader.is_loaded
        
        status = "healthy" if (rules_ok and inference_ok and model_loaded) else "degraded"
        
        return {
            "status": status,
            "rules_loaded": rules_ok,
            "rules_count": len(rules) if rules_ok else 0,
            "inference_engine_ready": inference_ok,
            "model_loaded": model_loaded
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
