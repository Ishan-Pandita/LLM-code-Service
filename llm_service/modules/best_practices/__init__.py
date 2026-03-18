"""
Best Practices Module - Code quality evaluation for college assignments.

This module evaluates code against coding best practice rules and provides
structured feedback with pass/fail status and evidence.

Endpoints:
- GET  /api/v1/modules/best_practices/rules     - Get predefined rules
- POST /api/v1/modules/best_practices/evaluate  - Evaluate code
"""

from .module import BestPracticesModule
from .schemas import (
    Rule,
    RulesResponse,
    EvaluationRequest,
    EvaluationResponse,
    RuleResult,
)
from .routes import router

__all__ = [
    "BestPracticesModule",
    "Rule",
    "RulesResponse",
    "EvaluationRequest",
    "EvaluationResponse",
    "RuleResult",
    "router",
]
