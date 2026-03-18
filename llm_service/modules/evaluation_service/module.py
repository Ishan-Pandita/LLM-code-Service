"""
Evaluation Service Module - Main module implementation.

Iteratively fixes code errors using LLM and verifies with compiler.
Uses one-error-at-a-time approach for better accuracy.
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional

from llm_service.core.inference_engine import GenerationConfig, InferenceEngine
from llm_service.modules.base import BaseModule, ModuleInput, ModuleOutput, get_module_registry

from .schemas import (
    EvaluationRequest,
    EvaluationResponse,
    ErrorInfo,
    FixInfo,
)
from .helpers import (
    CompilerClient,
    parse_compiler_error,
    build_fix_prompt,
    extract_json,
    validate_fix_output,
)

logger = logging.getLogger(__name__)


class EvaluationServiceInput(ModuleInput):
    """Input schema for evaluation service module."""
    
    code: str
    language_id: int
    language_name: Optional[str] = None
    max_iterations: int = 10
    stdin: Optional[str] = None
    expected_output: Optional[str] = None
    expected_code: Optional[str] = None


class EvaluationServiceOutput(ModuleOutput):
    """Output schema for evaluation service module."""
    
    success: bool
    total_errors_fixed: int
    iterations_used: int
    fixed_code: str
    fixes: List[Dict[str, Any]]


class EvaluationServiceModule(BaseModule):
    """
    Evaluation Service Module - Fixes code errors iteratively.
    
    Process:
    1. Submit code to compiler
    2. If error, use LLM to fix ONE error
    3. Verify fix with compiler
    4. Repeat until no errors or max iterations
    
    Benefits:
    - One error at a time = more accurate fixes
    - Compiler verification = guaranteed working code
    - Detailed fix history = educational value
    """
    
    module_id = "evaluation_service"
    input_schema = EvaluationServiceInput
    output_schema = EvaluationServiceOutput
    
    def __init__(self, inference_engine: InferenceEngine) -> None:
        super().__init__(inference_engine)
        self.compiler = CompilerClient()
    
    def build_system_prompt(self) -> str:
        """Not used directly - using helpers."""
        return ""
    
    def build_user_prompt(self, payload: Dict[str, Any]) -> str:
        """Not used directly - using helpers."""
        return ""
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation config optimized for code fixing."""
        return GenerationConfig(
            max_new_tokens=4096,  # Larger for complete code output
            temperature=0.0,      # Deterministic
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            do_sample=False,
        )
    
    async def fix_single_error(
        self,
        code: str,
        error: ErrorInfo,
        language_name: str,
        expected_code: Optional[str] = None,
        expected_output: Optional[str] = None,
        use_function_calling: bool = True
    ) -> Dict[str, Any]:
        """
        Use LLM to fix a single error in the code.
        
        Uses function calling for more reliable structured output,
        with fallback to JSON parsing.
        
        Args:
            code: Current code with error
            error: The error to fix
            language_name: Programming language name
            expected_code: Reference implementation (correct code)
            expected_output: Expected output for the program
            use_function_calling: Whether to use function calling
            
        Returns:
            Fix information dictionary
        """
        logger.info(f"Fixing error: {error.error_type} - {error.message[:100]}...")
        
        config = self.get_generation_config()
        
        # Try function calling first
        if use_function_calling:
            try:
                from .helpers import (
                    get_fix_code_tool,
                    build_fix_messages_with_tools,
                    parse_fix_function_call
                )
                
                # Build tool-aware messages
                messages = build_fix_messages_with_tools(
                    code, error, language_name,
                    expected_code=expected_code,
                    expected_output=expected_output
                )
                tool = get_fix_code_tool()
                
                # Generate with tools
                result = await self.inference_engine.generate_with_tools(
                    messages,
                    tools=[tool],
                    config=config
                )
                
                # Check if model used function calling
                if result.has_function_calls:
                    for fc in result.function_calls:
                        if fc.name == "submit_fix":
                            parsed = parse_fix_function_call(fc)
                            logger.debug(f"Got function call fix result")
                            return parsed
                
                # If no function call, fall through to fallback
                logger.debug("No function call for fix, trying text extraction")
                
            except Exception as e:
                logger.warning(f"Function calling failed for fix, using fallback: {e}")
        
        # Fallback: Traditional JSON parsing
        system_msg, user_msg = build_fix_prompt(
            code, error, language_name,
            expected_code=expected_code,
            expected_output=expected_output
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        result = await self.inference_engine.generate(messages, config)
        
        try:
            json_str = extract_json(result.generated_text)
            parsed = json.loads(json_str)
            validate_fix_output(parsed)
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse fix output: {e}")
            raise ValueError(f"Failed to parse LLM fix output: {e}")
    
    async def evaluate_and_fix(
        self,
        code: str,
        language_id: int,
        language_name: Optional[str] = None,
        max_iterations: int = 10,
        stdin: Optional[str] = None,
        expected_output: Optional[str] = None,
        expected_code: Optional[str] = None,
    ) -> EvaluationResponse:
        """
        Evaluate code and iteratively fix errors.
        
        Args:
            code: Code to evaluate
            language_id: Compiler language ID
            language_name: Human-readable language name
            max_iterations: Maximum fix iterations
            stdin: Optional input for execution
            expected_output: Expected output (helps detect logic errors)
            expected_code: Reference implementation (helps fix incomplete/wrong code)
            
        Returns:
            EvaluationResponse with all fixes and final code
        """
        original_code = code
        current_code = code
        fixes: List[FixInfo] = []
        iteration = 0
        
        # Default language name if not provided
        if not language_name:
            language_name = "code"
        
        logger.info(f"Starting evaluation with max {max_iterations} iterations")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[Iteration {iteration}/{max_iterations}] Submitting to compiler...")
            
            # Submit to compiler (run in threadpool to avoid blocking loop)
            # Assuming compiler.submit_code is synchronous requests-based
            # Submit to compiler (async)
            result = await self.compiler.submit_code(
                source_code=current_code,
                language_id=language_id,
                stdin=stdin,
                expected_output=expected_output,
                wait=True
            )
            
            # Check for API error
            if "error" in result and not result.get("status"):
                logger.error(f"Compiler API error: {result['error']}")
                return EvaluationResponse(
                    success=False,
                    total_errors_fixed=len(fixes),
                    iterations_used=iteration,
                    original_code=original_code,
                    fixed_code=current_code,
                    fixes=fixes,
                    remaining_errors=[ErrorInfo(
                        error_type="APIError",
                        message=str(result.get("error", "Compiler service unavailable")),
                        line_number=None
                    )]
                )
            
            # Parse error from result
            error = parse_compiler_error(result)
            
            # If no error, we're done!
            if error is None:
                logger.info(f"Success! Code runs without errors after {len(fixes)} fixes")
                return EvaluationResponse(
                    success=True,
                    total_errors_fixed=len(fixes),
                    iterations_used=iteration,
                    original_code=original_code,
                    fixed_code=current_code,
                    fixes=fixes,
                    final_output=result.get("stdout", "")
                )
            
            logger.info(f"Error found: {error.error_type}")
            
            # Check if this is an infrastructure error (not fixable by LLM)
            if getattr(error, 'is_infrastructure_error', False):
                logger.error(f"Infrastructure error detected: {error.message}")
                return EvaluationResponse(
                    success=False,
                    total_errors_fixed=len(fixes),
                    iterations_used=iteration,
                    original_code=original_code,
                    fixed_code=current_code,
                    fixes=fixes,
                    remaining_errors=[error]
                )

            
            # Use LLM to fix the error (pass expected_code and expected_output for context)
            try:
                fix_result = await self.fix_single_error(
                    current_code, error, language_name,
                    expected_code=expected_code,
                    expected_output=expected_output
                )
                
                # Create fix info
                fix_info = FixInfo(
                    iteration=iteration,
                    error=error,
                    original_code_snippet=fix_result.get("original_snippet", ""),
                    fixed_code_snippet=fix_result.get("fixed_snippet", ""),
                    explanation=fix_result.get("explanation", ""),
                    severity=fix_result.get("severity", 5)
                )
                fixes.append(fix_info)
                
                # Update code
                current_code = fix_result.get("fixed_code", current_code)
                logger.info(f"Applied fix (severity: {fix_info.severity})")
                
            except Exception as e:
                logger.error(f"Failed to fix error: {e}")
                return EvaluationResponse(
                    success=False,
                    total_errors_fixed=len(fixes),
                    iterations_used=iteration,
                    original_code=original_code,
                    fixed_code=current_code,
                    fixes=fixes,
                    remaining_errors=[error]
                )
        
        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached")
        
        # Do one final check
        # Do one final check
        result = await self.compiler.submit_code(
            source_code=current_code,
            language_id=language_id,
            stdin=stdin,
            wait=True
        )
        final_error = parse_compiler_error(result)
        
        return EvaluationResponse(
            success=final_error is None,
            total_errors_fixed=len(fixes),
            iterations_used=iteration,
            original_code=original_code,
            fixed_code=current_code,
            fixes=fixes,
            final_output=result.get("stdout", "") if final_error is None else None,
            remaining_errors=[final_error] if final_error else None
        )
    
    def parse_output(self, raw_output: str) -> EvaluationServiceOutput:
        """Parse output from raw LLM response."""
        json_str = extract_json(raw_output)
        parsed = json.loads(json_str)
        
        return EvaluationServiceOutput(
            success=parsed.get("success", False),
            total_errors_fixed=parsed.get("total_errors_fixed", 0),
            iterations_used=parsed.get("iterations_used", 0),
            fixed_code=parsed.get("fixed_code", ""),
            fixes=parsed.get("fixes", [])
        )
    
    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute via base module interface."""
        validated = self.validate_input(payload)
        
        result = await self.evaluate_and_fix(
            code=validated.code,
            language_id=validated.language_id,
            language_name=validated.language_name,
            max_iterations=validated.max_iterations,
            stdin=validated.stdin,
            expected_output=validated.expected_output,
        )
        
        return {
            "module_id": self.module_id,
            "output": result.model_dump(),
            "usage": {}
        }


# Register the module
get_module_registry().register(EvaluationServiceModule)
