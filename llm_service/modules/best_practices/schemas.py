"""
Best Practices Module - Pydantic schemas for input/output validation.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field



# Rule Schemas


class Rule(BaseModel):
    """A coding best practice rule."""
    
    id: str = Field(
        ...,
        description="Unique rule identifier (e.g., R1, R2)"
    )
    name: str = Field(
        ...,
        description="Human-readable rule name"
    )
    description: str = Field(
        ...,
        description="Detailed description of what the rule checks"
    )
    category: Optional[str] = Field(
        default=None,
        description="Rule category (e.g., readability, structure)"
    )


class RulesResponse(BaseModel):
    """Response schema for rules endpoint."""
    
    rules: List[Rule] = Field(
        ...,
        description="List of available rules"
    )
    total: int = Field(
        ...,
        description="Total number of rules"
    )



# Evaluation Schemas


class EvaluationRequest(BaseModel):
    """Request schema for code evaluation."""
    
    language: str = Field(
        ...,
        min_length=1,
        description="Programming language of the code"
    )
    code: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The code to evaluate"
    )
    predefined_rules: Optional[List[str]] = Field(
        default=None,
        description="List of predefined rule IDs to use (e.g., ['R1', 'R3'])."
    )
    custom_rules: Optional[List[Rule]] = Field(
        default=None,
        description="Additional custom rules defined by the user"
    )


class RuleResult(BaseModel):
    """Result for a single rule evaluation."""
    
    rule_id: str = Field(
        ...,
        description="The rule ID that was evaluated"
    )
    status: Literal["PASS", "FAIL"] = Field(
        ...,
        description="Whether the code passed or failed this rule"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="Confidence level of the evaluation"
    )
    evidence: str = Field(
        ...,
        description="Short quote or reference from code supporting the verdict"
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="Improvement suggestion if the rule failed"
    )


class EvaluationResponse(BaseModel):
    """Response schema for code evaluation."""
    
    overall_status: str = Field(
        ...,
        description="Overall evaluation status in format 'passed/total' (e.g., '8/10')"
    )
    rules: List[RuleResult] = Field(
        ...,
        description="Individual rule evaluation results"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional summary of the evaluation"
    )
