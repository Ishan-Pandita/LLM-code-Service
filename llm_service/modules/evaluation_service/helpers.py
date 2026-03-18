"""
Evaluation Service Module - Helper functions for compiler integration and prompts.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .schemas import ErrorInfo

from llm_service.core.config import get_settings

logger = logging.getLogger(__name__)


class CompilerClient:
    """
    Client for interacting with the Judge0-compatible compiler service.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        if base_url is None:
            try:
                base_url = get_settings().compiler_base_url
            except Exception:
                base_url = "http://localhost:32358"
                
        self.base_url = base_url
        self.submissions_url = f"{base_url}/submissions"
        self.languages_url = f"{base_url}/languages"
        self.timeout = timeout
    
    async def get_languages(self) -> List[Dict[str, Any]]:
        """Get list of supported languages."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.languages_url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get languages: {e}")
            return []
    
    def get_languages_sync(self) -> List[Dict[str, Any]]:
        """Get list of supported languages (synchronous)."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self.languages_url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get languages: {e}")
            return []
    
    async def submit_code(
        self,
        source_code: str,
        language_id: int,
        stdin: Optional[str] = None,
        expected_output: Optional[str] = None,
        wait: bool = True
    ) -> Dict[str, Any]:
        """
        Submit code for compilation and execution.
        
        Args:
            source_code: The source code to compile/run
            language_id: Language ID from the compiler service
            stdin: Optional standard input
            expected_output: Optional expected output for validation
            wait: Whether to wait for results (Note: always uses async mode internally)
            
        Returns:
            Submission result with status and output/error
        """
        payload = {
            "source_code": source_code,
            "language_id": language_id,
        }
        
        if stdin:
            payload["stdin"] = stdin
        if expected_output:
            payload["expected_output"] = expected_output
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Always use wait=false (async mode) and poll for results
                # This is because some Judge0 deployments have issues with wait=true
                response = await client.post(
                    f"{self.submissions_url}?base64_encoded=false&wait=false",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                # Poll for result since we're using async submission
                if "token" in result:
                    result = await self._poll_result(client, result["token"])
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to submit code: {e}")
            return {"error": str(e)}
    
    async def _poll_result(
        self,
        client: httpx.AsyncClient,
        token: str,
        max_attempts: int = 30,
        interval: float = 1.0
    ) -> Dict[str, Any]:
        """Poll for submission result."""
        import asyncio
        for _ in range(max_attempts):
            try:
                response = await client.get(
                    f"{self.submissions_url}/{token}?base64_encoded=false"
                )
                response.raise_for_status()
                result = response.json()
                
                # Check if processing is complete
                status_id = result.get("status", {}).get("id", 0)
                if status_id >= 3:  # 3+ means processing complete
                    return result
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Failed to poll result: {e}")
                return {"error": str(e)}
        
        return {"error": "Timeout waiting for result"}
    
    async def get_result(self, token: str) -> Dict[str, Any]:
        """Get result for a submission by token."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.submissions_url}/{token}?base64_encoded=false"
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get result: {e}")
            return {"error": str(e)}


def parse_compiler_error(result: Dict[str, Any]) -> Optional[ErrorInfo]:
    """
    Parse compiler result to extract error information.
    
    Args:
        result: Compiler submission result
        
    Returns:
        ErrorInfo if there's an error, None otherwise
    """
    status = result.get("status", {})
    status_id = status.get("id", 0)
    
    # Status IDs:
    # 1: In Queue, 2: Processing, 3: Accepted
    # 4: Wrong Answer, 5: Time Limit Exceeded
    # 6: Compilation Error, 7-12: Various errors
    # 13: Internal Error (infrastructure issue, not code error)
    # 14: Exec Format Error
    
    if status_id == 3:  # Accepted - no error
        return None
    
    # Get error details
    error_message = ""
    error_type = "UnknownError"
    line_number = None
    is_infrastructure_error = False
    
    # Handle Internal Error (status 13) - this is NOT a code error
    # It's an infrastructure issue with the compiler service
    if status_id == 13:
        error_type = "InternalError"
        error_message = result.get("message", "") or "Compiler service internal error"
        is_infrastructure_error = True
        logger.warning(f"Compiler service internal error: {error_message}")
    elif status_id == 6:  # Compilation Error
        error_type = "CompileError"
        error_message = result.get("compile_output", "") or result.get("message", "")
    elif result.get("stderr"):
        error_type = "RuntimeError"
        error_message = result.get("stderr", "")
    elif status_id == 4:
        error_type = "WrongAnswer"
        error_message = f"Expected: {result.get('expected_output', 'N/A')}, Got: {result.get('stdout', 'N/A')}"
    elif status_id == 5:
        error_type = "TimeLimitExceeded"
        error_message = "Execution took too long"
    else:
        error_type = status.get("description", "UnknownError")
        error_message = result.get("message", "") or result.get("stderr", "") or "Unknown error"
    
    # Try to extract line number from error message (only for code errors)
    if not is_infrastructure_error:
        line_match = re.search(r'line (\d+)', error_message, re.IGNORECASE)
        if line_match:
            line_number = int(line_match.group(1))
    
    error_info = ErrorInfo(
        error_type=error_type,
        message=error_message.strip(),
        line_number=line_number
    )
    
    # Mark infrastructure errors so they can be handled differently
    error_info.is_infrastructure_error = is_infrastructure_error
    
    return error_info


def build_fix_prompt(
    code: str,
    error: ErrorInfo,
    language_name: str,
    expected_code: Optional[str] = None,
    expected_output: Optional[str] = None
) -> Tuple[str, str]:
    """
    Build system and user prompts for fixing a single error.
    
    Args:
        code: The current code with error
        error: The error to fix
        language_name: Programming language name
        expected_code: Reference implementation (correct code)
        expected_output: Expected output for the program
        
    Returns:
        Tuple of (system_message, user_message)
    """
    # Optimized system prompt
    # Key: Expert identity + debugging methodology + minimal change philosophy
    system_msg = """You are an expert code debugger specializing in fixing errors in academic programming assignments.

<identity>
- Senior software engineer with expertise in debugging across all major languages
- Patient teacher who preserves student coding style while fixing issues
- Precise problem-solver who makes minimal, targeted changes
</identity>

<debugging_methodology>
When fixing errors, follow this exact process:
1. ANALYZE: Parse the error message to understand the failure type
2. LOCATE: Pinpoint the exact line(s) causing the error
3. DIAGNOSE: Determine root cause (syntax, logic, runtime, etc.)
4. FIX: Apply the smallest possible change to resolve the issue
5. VALIDATE: Mentally verify the fix doesn't break other code
</debugging_methodology>

<minimal_change_principle>
Critical: Make the SMALLEST change necessary to fix the error.
- Do NOT refactor unrelated code
- Do NOT add features or optimizations
- Do NOT change coding style
- Do NOT remove or rewrite working code
- ONLY touch what's broken
</minimal_change_principle>

<severity_classification>
Assign severity 1-10 based on error complexity:

Syntax Errors (1-3):
  1 = Trivial typo (missing semicolon, extra comma)
  2 = Basic syntax (missing colon, quotes, parenthesis)
  3 = Scope/bracket mismatch

Declaration/Type Errors (4-6):
  4 = Undefined variable or missing declaration
  5 = Type mismatch or wrong argument types
  6 = Missing import or module not found

Runtime/Logic Errors (7-10):
  7 = Control flow error (infinite loop, wrong condition)
  8 = Runtime exception (null reference, index error)
  9 = Logic error (wrong algorithm, incorrect output)
  10 = Major structural problem (incomplete implementation)
</severity_classification>

<output_requirements>
Output EXACTLY ONE valid JSON object. No text before or after.
Schema:
{
  "fixed_code": "string - complete fixed code, ready to run",
  "original_snippet": "string - the broken code (1-5 lines)",
  "fixed_snippet": "string - the corrected code (1-5 lines)",
  "explanation": "string - what was wrong and how you fixed it",
  "severity": integer - 1 to 10 based on classification
}
</output_requirements>

<critical_rules>
- Always output complete, runnable fixed_code
- Never add markdown formatting around JSON
- Keep explanation concise but informative
- Preserve all working parts of the original code
</critical_rules>"""

    # Build structured error information
    line_info = f"\nLine: {error.line_number}" if error.line_number else ""

    # Build optional context sections
    context_sections = ""
    
    if expected_output:
        context_sections += f"""
<expected_output>
{expected_output}
</expected_output>
Note: Use this to verify your fix produces the correct result."""

    if expected_code:
        context_sections += f"""

<reference_solution language="{language_name}">
{expected_code}
</reference_solution>
Note: Use this to understand the intended logic. Do not copy verbatim."""

    # Optimized user prompt
    # Key: Clear error context + structured code + explicit action
    user_msg = f"""<task>
Fix the error in this {language_name} code.
</task>

<error type="{error.error_type}">
{error.message}{line_info}
</error>

<broken_code language="{language_name}">
{code}
</broken_code>
{context_sections}

<action>
1. Analyze the error message to understand what went wrong
2. Find the exact location causing the error
3. Apply a minimal fix to resolve it
4. Output complete fixed code as JSON
</action>"""

    return system_msg, user_msg


def extract_json(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks."""
    text = text.strip()
    
    # Try markdown code block
    json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1)
    
    # Try standalone JSON
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start != -1 and end > start:
        return text[start:end]
    
    raise ValueError("No valid JSON found in model output")


def validate_fix_output(parsed: Dict[str, Any]) -> None:
    """Validate the LLM fix output."""
    required_fields = ["fixed_code", "original_snippet", "fixed_snippet", "explanation", "severity"]
    
    for field in required_fields:
        if field not in parsed:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate severity
    severity = parsed.get("severity")
    if not isinstance(severity, int) or severity < 1 or severity > 10:
        parsed["severity"] = 5  # Default to medium severity



# Function Calling Support


def get_fix_code_tool():
    """
    Get the tool definition for code fixing.
    
    This leverages native function calling capabilities
    for more reliable structured output.
    
    Returns:
        ToolDefinition for code fixing
    """
    from llm_service.core.function_calling import ToolDefinition, ToolParameter
    
    return ToolDefinition(
        name="submit_fix",
        description="Submit the fixed code with explanation",
        parameters=[
            ToolParameter(
                name="fixed_code",
                description="The complete fixed code that resolves the error",
                type="string",
                required=True
            ),
            ToolParameter(
                name="original_snippet",
                description="The problematic code snippet (1-5 lines)",
                type="string",
                required=True
            ),
            ToolParameter(
                name="fixed_snippet",
                description="The corrected code snippet (1-5 lines)",
                type="string",
                required=True
            ),
            ToolParameter(
                name="explanation",
                description="Clear explanation of what was wrong and how it was fixed",
                type="string",
                required=True
            ),
            ToolParameter(
                name="severity",
                description="Error severity from 1-10 (1=simple typo, 10=major structural error)",
                type="integer",
                required=True
            )
        ]
    )


def build_fix_messages_with_tools(
    code: str,
    error: ErrorInfo,
    language_name: str,
    expected_code: Optional[str] = None,
    expected_output: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Build messages for function calling mode.
    
    Uses a streamlined prompt since output structure is enforced
    by the function calling mechanism.
    
    Args:
        code: The code with error
        error: Error information
        language_name: Programming language
        expected_code: Optional reference implementation
        expected_output: Optional expected output
        
    Returns:
        List of message dictionaries
    """
    # Function calling optimized system prompt
    # Simpler since structure is enforced by the tool schema
    system_msg = """You are an expert code debugger.

<task>
Fix ONE specific error in student code, then call the submit_fix function with your solution.
</task>

<debugging_process>
1. Analyze the error message to understand what failed
2. Locate the exact line(s) causing the error
3. Apply the SMALLEST change to fix it
4. Verify mentally that the fix doesn't break other code
5. Call submit_fix with the complete fixed code
</debugging_process>

<severity_guide>
1-3: Syntax (typo, missing punctuation, brackets)
4-6: Declaration (undefined var, type mismatch, missing import)
7-8: Runtime (null ref, index error, division by zero)
9-10: Logic (wrong output, incomplete implementation)
</severity_guide>

<critical>
- Preserve student's code style
- Make MINIMAL changes only
- Do NOT refactor or add features
</critical>"""

    # Build error info
    line_info = f"\nLine: {error.line_number}" if error.line_number else ""

    # Build context sections
    context = ""
    if expected_output:
        context += f"""
<expected_output>
{expected_output}
</expected_output>"""

    if expected_code:
        context += f"""
<reference language="{language_name}">
{expected_code}
</reference>"""

    # Function calling optimized user prompt
    user_msg = f"""<error type="{error.error_type}">
{error.message}{line_info}
</error>

<code language="{language_name}">
{code}
</code>
{context}

Fix the error and call submit_fix with your solution."""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def parse_fix_function_call(function_call) -> Dict[str, Any]:
    """
    Parse a function call result into fix result format.
    
    Args:
        function_call: FunctionCall object from the model
        
    Returns:
        Fix result dictionary
    """
    args = function_call.arguments
    
    severity = args.get("severity", 5)
    if isinstance(severity, str):
        try:
            severity = int(severity)
        except ValueError:
            severity = 5
    
    return {
        "fixed_code": args.get("fixed_code", ""),
        "original_snippet": args.get("original_snippet", ""),
        "fixed_snippet": args.get("fixed_snippet", ""),
        "explanation": args.get("explanation", ""),
        "severity": max(1, min(10, severity))  # Clamp to 1-10
    }

