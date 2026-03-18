"""
Evaluation Service Module - Pydantic schemas for input/output validation.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field



# Input Schemas


class EvaluationRequest(BaseModel):
    """Request schema for code evaluation and fixing."""
    
    code: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The code to evaluate and fix (may contain errors or be incomplete)"
    )
    language_id: int = Field(
        ...,
        description="Language ID from the compiler service"
    )
    language_name: Optional[str] = Field(
        default=None,
        description="Human-readable language name (e.g., 'python', 'cpp')"
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of fix iterations"
    )
    stdin: Optional[str] = Field(
        default=None,
        description="Standard input for the program"
    )
    expected_output: Optional[str] = Field(
        default=None,
        description="Expected output for validation - helps detect logic errors"
    )
    expected_code: Optional[str] = Field(
        default=None,
        description="Reference implementation (correct code) - helps fix incomplete/wrong logic"
    )



# Error and Fix Schemas


class ErrorInfo(BaseModel):
    """Information about a detected error."""
    
    error_type: str = Field(
        ...,
        description="Type of error (e.g., 'SyntaxError', 'RuntimeError', 'CompileError', 'InternalError')"
    )
    message: str = Field(
        ...,
        description="Error message from compiler"
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Line number where error occurred"
    )
    is_infrastructure_error: bool = Field(
        default=False,
        description="Whether this is an infrastructure error (compiler service issue) rather than a code error"
    )



class FixInfo(BaseModel):
    """Information about a fix applied."""
    
    iteration: int = Field(
        ...,
        description="Which iteration this fix was applied"
    )
    error: ErrorInfo = Field(
        ...,
        description="The error that was fixed"
    )
    original_code_snippet: str = Field(
        ...,
        description="The problematic code snippet before fix"
    )
    fixed_code_snippet: str = Field(
        ...,
        description="The corrected code snippet after fix"
    )
    explanation: str = Field(
        ...,
        description="Explanation of what was wrong and how it was fixed"
    )
    severity: int = Field(
        ...,
        ge=1,
        le=10,
        description="Severity of the error (1=minor, 10=critical)"
    )



# Output Schemas


class EvaluationResponse(BaseModel):
    """Response schema for code evaluation."""
    
    success: bool = Field(
        ...,
        description="Whether the code was successfully fixed and runs without errors"
    )
    total_errors_fixed: int = Field(
        ...,
        description="Total number of errors that were fixed"
    )
    iterations_used: int = Field(
        ...,
        description="Number of iterations used to fix errors"
    )
    original_code: str = Field(
        ...,
        description="The original code submitted"
    )
    fixed_code: str = Field(
        ...,
        description="The final fixed code"
    )
    fixes: List[FixInfo] = Field(
        default_factory=list,
        description="List of all fixes applied"
    )
    final_output: Optional[str] = Field(
        default=None,
        description="Output of the fixed code after execution"
    )
    remaining_errors: Optional[List[ErrorInfo]] = Field(
        default=None,
        description="Any errors that could not be fixed (if max iterations reached)"
    )


class LanguageInfo(BaseModel):
    """Information about a supported language."""
    
    id: int
    name: str
